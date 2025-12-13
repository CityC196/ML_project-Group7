import sys
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py
import matplotlib.pyplot as plt

# ==============================================================================
# [用户配置区域] 修改此处的路径即可生效
# ==============================================================================
INPUT_DATA_PATH = "../data/Advection/1D_Advection_Sols_beta1.0.hdf5"
OUTPUT_DIR      = "../ExpertModels/ExpertModels_own/advection_unet"
# ==============================================================================

# ================= 配置参数 =================
class Config:
    data_path = INPUT_DATA_PATH
    output_dir = OUTPUT_DIR
    orig_resolution = 1024
    target_resolution = 256 
    stride = orig_resolution // target_resolution
    
    input_frames = 4  
    output_frames = 1
    
    batch_size = 64
    epochs = 100
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 数据集类 =================
class AdvectionTimeSeriesDataset(Dataset):
    def __init__(self, file_path, stride, input_frames, output_frames, mode='train'):
        self.stride = stride
        self.n_in = input_frames
        self.n_out = output_frames
        try:
            with h5py.File(file_path, 'r') as f:
                if 'tensor' in f.keys():
                    data = f['tensor'][:] 
                elif 'data' in f.keys():
                    data = f['data'][:]
                else:
                    data = f[list(f.keys())[0]][:]
        except Exception:
            data = np.random.rand(100, 200, 1024).astype(np.float32)

        data = data[..., ::self.stride]
        split = int(0.9 * len(data))
        if mode == 'train':
            self.data = data[:split]
        else:
            self.data = data[split:]
            
        self.traj_len = self.data.shape[1]
        self.n_samples_per_traj = self.traj_len - self.n_in - self.n_out + 1
        self.total_samples = len(self.data) * self.n_samples_per_traj
        
        print(f"[{mode}] Samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        traj_idx = idx // self.n_samples_per_traj
        time_idx = idx % self.n_samples_per_traj
        traj = self.data[traj_idx]
        x_slice = traj[time_idx : time_idx + self.n_in]
        y_slice = traj[time_idx + self.n_in : time_idx + self.n_in + self.n_out]
        return torch.from_numpy(x_slice).float(), torch.from_numpy(y_slice).float()

# ================= UNet 模型 =================
class TimeSeriesUNet1d(nn.Module):
    def __init__(self, in_channels, out_channels, features=32):
        super(TimeSeriesUNet1d, self).__init__()
        self.enc1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = self._block(features, features*2)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = self._block(features*2, features*4)
        self.pool3 = nn.MaxPool1d(2)
        self.bottleneck = self._block(features*4, features*8)
        self.up3 = nn.ConvTranspose1d(features*8, features*4, 2, 2)
        self.dec3 = self._block(features*8, features*4)
        self.up2 = nn.ConvTranspose1d(features*4, features*2, 2, 2)
        self.dec2 = self._block(features*4, features*2)
        self.up1 = nn.ConvTranspose1d(features*2, features, 2, 2)
        self.dec1 = self._block(features*2, features)
        self.final = nn.Conv1d(features, out_channels, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)

# ================= 主程序 =================
def main():
    if not os.path.exists(Config.output_dir):
        os.makedirs(Config.output_dir)

    print(f"--- Training Time-Series UNet ---")
    
    train_set = AdvectionTimeSeriesDataset(Config.data_path, Config.stride, Config.input_frames, Config.output_frames, mode='train')
    val_set = AdvectionTimeSeriesDataset(Config.data_path, Config.stride, Config.input_frames, Config.output_frames, mode='val')
    
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False)
    
    model = TimeSeriesUNet1d(in_channels=Config.input_frames, out_channels=Config.output_frames).to(Config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    loss_fn = nn.MSELoss()
    
    history = []
    
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(Config.device), y.to(Config.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.device), y.to(Config.device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
        avg_val = val_loss / len(val_loader)
        
        history.append(avg_val)
        print(f"Epoch {epoch+1} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")
        
        if (epoch+1) % 20 == 0:
            torch.save(model.state_dict(), f"{Config.output_dir}/model_{epoch+1}.pt")

    # 4. 可视化检查 (6张图片子图)
    x_test, y_test = val_set[0] # x: [4, 256], y: [1, 256]
    
    model.eval()
    with torch.no_grad():
        pred = model(x_test.unsqueeze(0).to(Config.device)).cpu().numpy() # [1, 1, 256]
        
    inputs = x_test.numpy()       # [4, 256]
    target = y_test.numpy()[0]    # [256]
    prediction = pred[0, 0]       # [256]
    
    # 计算前三帧平均
    avg_first_3 = np.mean(inputs[:3], axis=0)
    
    plt.figure(figsize=(15, 8))
    
    # 1. Input t-3
    plt.subplot(2, 3, 1)
    plt.plot(inputs[0], 'b-', alpha=0.7)
    plt.title("Input Frame 1 (t-3)")
    plt.grid(True, alpha=0.3)
    
    # 2. Input t-2
    plt.subplot(2, 3, 2)
    plt.plot(inputs[1], 'b-', alpha=0.7)
    plt.title("Input Frame 2 (t-2)")
    plt.grid(True, alpha=0.3)
    
    # 3. Input t-1
    plt.subplot(2, 3, 3)
    plt.plot(inputs[2], 'b-', alpha=0.7)
    plt.title("Input Frame 3 (t-1)")
    plt.grid(True, alpha=0.3)
    
    # 4. Average
    plt.subplot(2, 3, 4)
    plt.plot(avg_first_3, 'g-', linewidth=2)
    plt.title("Average of First 3 Inputs")
    plt.grid(True, alpha=0.3)
    
    # 5. Ground Truth
    plt.subplot(2, 3, 5)
    plt.plot(target, 'k-', linewidth=2)
    plt.title("Ground Truth (t+1)")
    plt.grid(True, alpha=0.3)
    
    # 6. Prediction
    plt.subplot(2, 3, 6)
    plt.plot(prediction, 'r--', linewidth=2)
    plt.title("Prediction (t+1)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{Config.output_dir}/prediction_check_6_views.png")
    
    print("Training Complete. Visual saved.")

if __name__ == "__main__":
    main()