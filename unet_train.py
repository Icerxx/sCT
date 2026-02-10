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
from unet import get_unet, get_unetpp
from swinunet import SwinTransformerSys
from transunet import VisionTransformer, CONFIG

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
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# hyperparameters
def parse_args():
    parser = argparse.ArgumentParser(description="Train MedDINO model (Iteration-based)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")

    # [修改] 训练长度控制改为 max_steps
    parser.add_argument("--max_steps", type=int, default=100000, help="Total number of training steps")
    parser.add_argument("--val_interval", type=int, default=1000, help="Validation interval (steps)")
    parser.add_argument("--save_interval", type=int, default=10000, help="Checkpoint save interval (steps)")

    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    # parser.add_argument("--unfreeze", action="store_true", help="Unfreeze encoder (ViT backbone) during training")
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

    # 建议加上 persistent_workers=True 以减少 epoch 切换时的 overhead
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True,
                              generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)

    # 2. Model
    # model = MedDINOGenerator(img_size=args.img_size, pretrained_path=PRETRAINED_PATH).to(device)
    # model = get_unet(device)
    # model = get_unetpp(device)
    # model = SwinTransformerSys(img_size=args.img_size, window_size=8, num_classes=1).to(device)
    model = VisionTransformer(CONFIG, img_size=256, num_classes=CONFIG.n_classes).to(device)

    # if args.unfreeze:
    #     print(">>> Mode: Full Fine-tuning (Encoder + Decoder).")
    # else:
    #     print(">>> Mode: Freezing Encoder (Backbone). Training Decoder only.")
    #     for param in model.vit_encoder.parameters():
    #         param.requires_grad = False

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
    print(f"Total Params: {total_params / 1e6:.2f}M | Trainable: {trainable_count / 1e6:.2f}M")

    optimizer = AdamW(trainable_params, lr=2e-4, weight_decay=1e-5)

    # [修改] T_max 设为总步数，实现更平滑的 Per-Step Decay
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, total_steps=args.max_steps, pct_start=0.05,
                                                    anneal_strategy='cos', final_div_factor=1000)
    # scaler = GradScaler()

    # 4. Loop
    best_val_loss = float('inf')

    # 固定可视化样本
    fixed_vis_batch = next(iter(val_loader))
    fixed_cbct = fixed_vis_batch['cbct'].to(device)
    fixed_ct = fixed_vis_batch['ct'].to(device)
    # fixed_seg = fixed_vis_batch['seg'].to(device) # 也拿出来看看

    # 初始化迭代器
    train_iter = iter(train_loader)

    # 进度条
    progress_bar = tqdm(range(1, args.max_steps + 1), desc="Training")

    running_loss = 0.0  # 用于计算 val_interval 期间的平均 loss

    for step in progress_bar:
        model.train()

        # --- 无限数据流逻辑 ---
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)  # Epoch 结束，重置迭代器
            batch = next(train_iter)
        # ---------------------

        cbct = batch['cbct'].to(device)
        ct = batch['ct'].to(device)
        seg = batch['seg'].to(device)

        optimizer.zero_grad()

        # Forward
        # with autocast("cuda"):
        pred_ct = model(cbct)
        loss, loss_dict = criterion(pred_ct, ct, seg_mask=seg)

        # Backward
        # scaler.scale(loss).backward()
        # prev_scale = scaler.get_scale()
        # scaler.step(optimizer)
        # scaler.update()
        # new_scale = scaler.get_scale()
        loss.backward()
        optimizer.step()

        # # [修改] Step-wise Scheduler Update
        # if new_scale >= prev_scale:
        #     scheduler.step()
        scheduler.step()

        # Logging
        current_loss = loss.item()
        running_loss += current_loss

        # 实时 loss
        progress_bar.set_postfix(loss=f"{current_loss:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.4e}")

        writer.add_scalar('Loss/Train_Step', current_loss, step)
        writer.add_scalar('Loss/Step_L1', loss_dict['global_l1'], step)
        writer.add_scalar('Loss/Step_masked_L1', loss_dict['soft_l1'], step)
        writer.add_scalar('Loss/Step_masked_Grad', loss_dict['bone_grad'], step)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], step)

        # --- Training Visualization ---
        if step % args.val_interval == 0:  # 每次验证时也可进行训练可视化
            with torch.no_grad():
                # 取当前 batch 的部分数据进行可视化
                vis_input = cbct[:, 0:1, :, :] * 0.5 + 0.5  # 归一化到 [0, 1]
                vis_pred = pred_ct * 0.5 + 0.5
                vis_label = ct * 0.5 + 0.5
                vis_seg = seg

                # 拼接可视化图像
                img_grid = torch.cat([vis_input, vis_pred, vis_label, vis_seg], dim=0)
                grid = make_grid(img_grid, nrow=vis_input.shape[0], padding=2, normalize=False)
                writer.add_image(' Train/Input_Pred_GT_Seg', grid, step)

        # --- Validation & Visualization ---
        if step % args.val_interval == 0:
            model.eval()
            val_loss = 0.0

            # 计算区间平均训练 Loss
            avg_train_interval_loss = running_loss / args.val_interval
            writer.add_scalar('Loss/Train_Avg_Interval', avg_train_interval_loss, step)
            running_loss = 0.0  # Reset

            with torch.no_grad():
                for val_batch in val_loader:
                    v_cbct = val_batch['cbct'].to(device)
                    v_ct = val_batch['ct'].to(device)

                    # with autocast("cuda"):
                    v_pred = model(v_cbct)
                    v_l = val_criterion(v_pred, v_ct)
                    val_loss += v_l.item()

            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar('Loss/Val_Step', avg_val_loss, step)

            print(
                f"\n[Step {step}/{args.max_steps}] Train Loss: {avg_train_interval_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            # Visualization
            with torch.no_grad():
                # 为了防止可视化时 OOM，单独跑一次 fixed batch
                with autocast("cuda"):
                    pred_vis = model(fixed_cbct)

                vis_input = fixed_cbct[:, 0:1, :, :] * 0.5 + 0.5
                vis_pred = pred_vis * 0.5 + 0.5
                vis_label = fixed_ct * 0.5 + 0.5
                # # seg 归一化到 [0,1] 方便显示 (假设 seg max label 不大，或者除以最大值)
                # vis_seg_disp = fixed_seg.float() / (fixed_seg.max() + 1e-6)

                img_grid = torch.cat([vis_input, vis_pred, vis_label], dim=0)
                grid = make_grid(img_grid, nrow=fixed_cbct.shape[0], padding=2, normalize=False)
                writer.add_image('Validation/Input_Pred_GT', grid, step)

            # Save Best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print("  -> Saved Best Model!")

        # --- Periodic Checkpoint ---
        if step % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"step_{step}.pth"))

    # End of Training
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join(ROOT, "log", f"{timestamp}_seed{args.seed}_steps")
    SAVE_DIR = os.path.join(ROOT, "checkpoints", f"{timestamp}_seed{args.seed}_steps")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train(LOG_DIR, SAVE_DIR, args, device)