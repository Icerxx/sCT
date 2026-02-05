import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import os
import datetime
from tqdm import tqdm
import argparse

from meddino import MedDINOGenerator
from utils.dataset import SynthRADDataset
from utils.loss import AnatomyAwareLoss, CharbonnierLoss

# global configs
ROOT = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_PATH = os.path.join(ROOT, "model.pth")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# hyperparameters
def parse_args():
    parser = argparse.ArgumentParser(description="Train MedDINO model")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--unfreeze", action="store_true", help="Freeze encoder (ViT backbone) during training")
    parser.add_argument("--gpu", type=int, default=0, help="Specify GPU ID to use (default: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def train(log_dir, save_dir, args, device):
    writer = SummaryWriter(log_dir=log_dir)

    # 1. Dataset & Loader
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_ds = SynthRADDataset(ROOT, split='train', img_size=args.img_size, augment=True)
    val_ds = SynthRADDataset(ROOT, split='val', img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=8, 
                              pin_memory=True, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, 
                            pin_memory=True, worker_init_fn=worker_init_fn)

    # 2. Model
    model = MedDINOGenerator(img_size=args.img_size, pretrained_path=PRETRAINED_PATH).to(device)

    if args.unfreeze:
        print(">>> Mode: Full Fine-tuning (Encoder + Decoder).")

    else:
        print(">>> Mode: Freezing Encoder (Backbone). Training Decoder only.")
        for param in model.vit_encoder.parameters():
            param.requires_grad = False
    
    # 3. Loss & Optimizer
    criterion = AnatomyAwareLoss(
        global_l1_weight=1.0,
        masked_l1_weight=0,
        masked_grad_weight=0
        ).to(device)
    val_criterion = CharbonnierLoss().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params/1e6:.2f}M | Trainable: {trainable_count/1e6:.2f}M")

    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    scaler = GradScaler()
    
    # 4. Loop
    best_val_loss = float('inf')
    global_step = 0

    fixed_vis_batch = next(iter(val_loader))
    fixed_cbct = fixed_vis_batch['cbct'].to(device)
    fixed_ct = fixed_vis_batch['ct'].to(device)
    
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")
        
        for batch in loop:
            cbct = batch['cbct'].to(device)  # (B, 3, H, W)
            ct = batch['ct'].to(device)  # (B, 1, H, W)
            seg = batch['seg'].to(device)  # (B, 1, H, W) Long Tensor
            
            optimizer.zero_grad()
            # Forward
            with autocast("cuda"):
                pred_ct = model(cbct)
                loss, loss_dict = criterion(pred_ct, ct, seg_mask=seg)
            
            # Backward
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            writer.add_scalar('Loss/Train_Step', loss.item(), global_step)
            writer.add_scalar('Loss/Step_L1', loss_dict['global_l1'], global_step)
            writer.add_scalar('Loss/Step_masked_L1', loss_dict['soft_l1'], global_step)
            writer.add_scalar('Loss/Step_masked_Grad', loss_dict['bone_grad'], global_step)
            # writer.add_scalar('Loss/Step_Perceptual', loss_dict['perceptual'], global_step)

            if global_step % 1000 == 0:
                with torch.no_grad():
                    vis_input = cbct[:, 0:1, :, :] * 0.5 + 0.5
                    vis_pred = pred_ct * 0.5 + 0.5
                    vis_label = ct * 0.5 + 0.5
                    vis_seg = seg

                    img_grid = torch.cat([vis_input, vis_pred, vis_label, vis_seg], dim=0)
                    gird = make_grid(img_grid, nrow=cbct.shape[0], padding=2, normalize=False)
                    writer.add_image('Train/Input_Pred_GT_Seg', gird, global_step)

            global_step += 1
            
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        # 获取当前 LR 并记录
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate', current_lr, epoch)
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                cbct = batch['cbct'].to(device)
                ct = batch['ct'].to(device)
                # seg = batch['seg'].to(device)

                with autocast("cuda"):
                    pred = model(cbct)
                    v_loss = val_criterion(pred, ct)
                    val_loss += v_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.4e}")
        writer.add_scalar('Loss/Val_Epoch', avg_val_loss, epoch)
        
        # --- 可视化 (每个 Epoch 记录一次图片) ---
        with torch.no_grad():
            model.eval()
            pred_vis = model(fixed_cbct)
            # 拼接显示: Input(只取第1通道) | Prediction | Ground Truth
            # fixed_cbct 是 (B, 3, H, W)，我们取 (B, 1, H, W) 用于灰度显示
            vis_input = fixed_cbct[:, 0:1, :, :] 
            
            # 将数值从 [-1, 1] 映射回 [0, 1] 以便 TensorBoard 正确显示
            img_grid = torch.cat([vis_input * 0.5 + 0.5, 
                                  pred_vis * 0.5 + 0.5, 
                                  fixed_ct * 0.5 + 0.5], dim=0)
            
            grid = make_grid(img_grid, nrow=fixed_cbct.shape[0], padding=2, normalize=False)
            
            writer.add_image('Visual/Input_Pred_GT', grid, epoch)

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("Saved Best Model!")

        torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))

        if epoch + 1 == 30:
            torch.save(model.state_dict(), os.path.join(save_dir, "epoch_30.pth"))

    writer.close()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join(ROOT, "log", f"{timestamp}_seed{args.seed}")
    SAVE_DIR = os.path.join(ROOT, "checkpoints", f"{timestamp}_seed{args.seed}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    train(LOG_DIR, SAVE_DIR, args, device)
