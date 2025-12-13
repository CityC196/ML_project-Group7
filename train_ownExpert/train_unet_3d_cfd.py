import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==============================================================================
# [用户配置区域]
# ==============================================================================
INPUT_TRAIN_DATA = "../data/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"
INPUT_VAL_DATA   = "../data/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"
OUTPUT_DIR       = "../ExpertModels/ExpertModels_own/cfd_3d_unet"
# ==============================================================================

# ==========================================
# 1. 模型定义
# ==========================================
class CompactUNet3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompactUNet3d, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c), nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c), nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(in_channels, 16)
        self.pool = nn.MaxPool3d(2)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.bottleneck = conv_block(64, 128)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = conv_block(128 + 64, 64)
        self.dec2 = conv_block(64 + 32, 32)
        self.dec1 = conv_block(32 + 16, 16)
        self.final = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        b = self.bottleneck(p3)
        d3 = self.up(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)

# ==========================================
# 2. 配置
# ==========================================
CONFIG = {
    "exp_name": "CFD_3D_Compact_8x8x8_InMemory",
    "train_data": INPUT_TRAIN_DATA,
    "val_data": INPUT_VAL_DATA,
    "save_dir": OUTPUT_DIR,
    "rdc": 8, 
    "history_steps": 3,
    "batch_size": 256, 
    "epochs": 100, 
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 0,
    "model_save_path": os.path.join(OUTPUT_DIR, "new_cfd_3d_unet.pt")
}
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 3. 数据集
# ==========================================
class CFD3DInMemoryDataset(Dataset):
    def __init__(self, filename, rdc=1, history_steps=3, mode='train'):
        self.rdc = rdc
        self.hist = history_steps
        self.mode = mode
        self.data_cache = [] 
        
        print(f"  -> Loading {mode} data: {filename}")
        try:
            with h5py.File(filename, "r") as f:
                num_samples = f['density'].shape[0]
                split_idx = int(num_samples * 0.9)
                indices = range(split_idx) if mode == 'train' else range(split_idx, num_samples)
                sl = slice(None, None, self.rdc)
                
                for idx in tqdm(indices, desc=f"Loading"):
                    rho = f['density'][idx, :, sl, sl, sl]
                    vx  = f['Vx'][idx, :, sl, sl, sl]
                    vy  = f['Vy'][idx, :, sl, sl, sl]
                    vz  = f['Vz'][idx, :, sl, sl, sl]
                    p   = f['pressure'][idx, :, sl, sl, sl]
                    sample_data = np.stack([rho, vx, vy, vz, p], axis=-1)
                    sample_data = np.transpose(sample_data, (0, 4, 1, 2, 3)).astype(np.float32)
                    self.data_cache.append(sample_data)
        except Exception as e:
            print(f"Error loading data: {e}. Generating dummy data for testing.")
            for _ in range(10):
                self.data_cache.append(np.random.randn(20, 5, 16, 16, 16).astype(np.float32))

    def __len__(self):
        return len(self.data_cache) * 5 

    def __getitem__(self, idx):
        real_idx = idx % len(self.data_cache)
        full_seq = self.data_cache[real_idx]
        max_start = full_seq.shape[0] - self.hist - 1
        t_start = np.random.randint(0, max_start + 1)
        seq_slice = full_seq[t_start : t_start + self.hist + 1]
        
        x_seq = seq_slice[:self.hist]
        y_seq = seq_slice[self.hist]
        x_in = x_seq.reshape(-1, x_seq.shape[2], x_seq.shape[3], x_seq.shape[4])
        return torch.from_numpy(x_in), torch.from_numpy(y_seq)

# ==========================================
# 4. 可视化 (2x3 Grid)
# ==========================================
def visualize_prediction(model, val_loader, device, save_dir, num_samples=3):
    print(f"\n>>> Generating Standardized Plots (2x3 Grid)...")
    model.eval()
    target_channel = 1 # Vx
    
    count = 0
    with torch.no_grad():
        for x, y in val_loader:
            if count >= num_samples: break
            
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            # Numpy Conversion [C, D, H, W]
            x_np = x[0].cpu().numpy()
            y_np = y[0].cpu().numpy()
            pred_np = pred[0].cpu().numpy()
            
            # --- 解析 Input History (t-3, t-2, t-1) ---
            # Input Shape: [Hist*5, D, H, W], Hist=3
            # Frame 1 (t-3): Channels 0-4 -> Vx is idx 1
            # Frame 2 (t-2): Channels 5-9 -> Vx is idx 6
            # Frame 3 (t-1): Channels 10-14 -> Vx is idx 11
            frame_indices = [1, 6, 11]
            input_frames = []
            
            depth = y_np.shape[1]
            mid_z = depth // 2
            
            for idx in frame_indices:
                slice_3d = x_np[idx]
                input_frames.append(slice_3d[mid_z, :, :]) # 取中间切片
            
            # 计算 Input Average
            avg_input = np.mean(input_frames, axis=0)
            
            # Target & Pred
            img_target = y_np[target_channel, mid_z, :, :]
            img_pred   = pred_np[target_channel, mid_z, :, :]
            
            # --- 绘图 (2x3 Grid) ---
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            vmin = min(img_target.min(), img_pred.min())
            vmax = max(img_target.max(), img_pred.max())
            
            # Row 1: Inputs
            titles_r1 = ["Input Frame 1 (t-3)", "Input Frame 2 (t-2)", "Input Frame 3 (t-1)"]
            for i in range(3):
                ax = axes[0, i]
                im = ax.imshow(input_frames[i], cmap='jet', vmin=vmin, vmax=vmax)
                ax.set_title(titles_r1[i])
                plt.colorbar(im, ax=ax)

            # Row 2: Avg, Truth, Pred
            # 2-1: Average
            ax = axes[1, 0]
            im = ax.imshow(avg_input, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title("Average Input")
            plt.colorbar(im, ax=ax)
            
            # 2-2: Ground Truth
            ax = axes[1, 1]
            im = ax.imshow(img_target, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title("Ground Truth (t)")
            plt.colorbar(im, ax=ax)
            
            # 2-3: Prediction
            ax = axes[1, 2]
            im = ax.imshow(img_pred, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title("Prediction (t)")
            plt.colorbar(im, ax=ax)
            
            plt.suptitle(f"3D CFD Prediction (Vx Slice) - Sample {count}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"std_vis_sample_{count}.png"))
            plt.close()
            count += 1
            print(f"  -> Saved visualization: std_vis_sample_{count}.png")

# ==========================================
# 5. Main
# ==========================================
def train():
    device = torch.device(CONFIG["device"])
    print(f"🚀 [Start] {CONFIG['exp_name']}")
    
    train_ds = CFD3DInMemoryDataset(CONFIG["train_data"], rdc=CONFIG["rdc"], history_steps=CONFIG["history_steps"], mode='train')
    val_ds = CFD3DInMemoryDataset(CONFIG["val_data"], rdc=CONFIG["rdc"], history_steps=CONFIG["history_steps"], mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0)
    
    x_sample, y_sample = train_ds[0]
    in_c, out_c = x_sample.shape[0], y_sample.shape[0]
    model = CompactUNet3d(in_channels=in_c, out_channels=out_c).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_loss = float('inf')
    
    print(">>> 开始训练...")
    # [修复] 恢复了完整的训练循环
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", unit="batch")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        print(f"   [Result] Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': CONFIG
            }, CONFIG["model_save_path"])
            print(f"   ✅ Best Model Saved: {CONFIG['model_save_path']}")
    
    # 训练结束后，加载最佳模型进行可视化
    if os.path.exists(CONFIG["model_save_path"]):
        checkpoint = torch.load(CONFIG["model_save_path"])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n>>> Loaded Best Model from Epoch {checkpoint['epoch']}")
    
    visualize_prediction(model, val_loader, device, CONFIG["save_dir"], num_samples=2)
    print("🎉 Done!")

if __name__ == "__main__":
    train()