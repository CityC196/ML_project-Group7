"""
位置: test_codes/test_expert_unet_diff_sorp.py
功能: 测试 Diff-Sorp U-Net (支持损坏检测 + 自动降采样适配 64分辨率模型)
"""
import os
import sys
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置 =================
CONFIG = {
    # 在这里切换你想测试的模型
    "model_path": "../ExpertModels/1D_diff-sorp_NA__Unet-PF-20.pt", 
    # "model_path": "../ExpertModels/1D_diff-sorp_NA__Unet-PF-20.pt", 
    
    "data_path": "../data/1D_diff-sorp_NA_NA.h5",
    "output_dir": "../OUTPUT/diff_sorp_unet_test",
    "in_channels": 10,
    "out_channels": 1,
    "init_features": 32,
    "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

sys.path.append("..") 
try:
    from pdebench.models.unet.unet import UNet1d
except ImportError:
    sys.exit("❌ 错误: 无法导入 pdebench。")

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ================= 数据加载 (含自动降采样) =================
def load_diff_sorp_data(filename, seed_idx, target_resolution=None):
    print(f">>> Loading data from {filename} ...")
    with h5py.File(filename, "r") as f:
        seed_str = str(seed_idx).zfill(4)
        if seed_str not in f: seed_str = list(f.keys())[0]
        group = f[seed_str]
        t = np.array(group["grid"]["t"], dtype=np.float32)
        x = np.array(group["grid"]["x"], dtype=np.float32)
        u_data = np.array(group["data"], dtype=np.float32) 
        
        if u_data.shape[0] == len(x) and u_data.shape[1] == len(t):
            u_data = np.transpose(u_data, (1, 0, 2))
            
    # --- 自动降采样逻辑 ---
    current_res = u_data.shape[1]
    if target_resolution and current_res != target_resolution:
        print(f"⚠️ Resolution Mismatch: Data({current_res}) vs Model({target_resolution})")
        print(f"   -> Performing Downsampling...")
        
        # 计算步长 (例如 1024 -> 64, step=16)
        step = current_res // target_resolution
        if current_res % target_resolution != 0:
            print("   ⚠️ Warning: 不能整除，可能会有截断偏差")
            
        # 执行切片降采样
        u_data = u_data[:, ::step, :]
        x = x[::step]
        print(f"   -> New Data Shape: {u_data.shape}")

    return u_data, x, t

# ================= 主流程 =================
def main():
    device = torch.device(CONFIG["device"])
    
    # 1. 智能判断模型分辨率
    target_res = None
    if "64_resolution" in CONFIG["model_path"]:
        target_res = 64
        print("💡 Detected '64_resolution' model, setting target resolution to 64.")
    
    # 2. 加载数据
    try:
        u_data, x_grid, t_grid = load_diff_sorp_data(CONFIG["data_path"], CONFIG["seed"], target_res)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 3. 加载模型
    print(f">>> Loading U-Net Model: {CONFIG['model_path']}")
    model = UNet1d(CONFIG["in_channels"], CONFIG["out_channels"], CONFIG["init_features"]).to(device)
    
    if not os.path.exists(CONFIG["model_path"]):
        print(f"❌ 模型文件不存在: {CONFIG['model_path']}")
        return

    try:
        checkpoint = torch.load(CONFIG["model_path"], map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("✅ 模型权重加载成功!")
    except RuntimeError as e:
        print(f"❌ [严重错误] 权重加载失败: {e}")
        print("💡 原因可能是文件损坏，或者你的 Git LFS 下载不完整。")
        return
    except Exception as e:
        print(f"❌ 文件读取错误: {e}")
        return

    model.eval()

    # 4. 推理 (滑动窗口)
    window_size = CONFIG["in_channels"]
    total_steps = len(t_grid)
    preds = []
    
    # 填入历史
    history_buffer = u_data[:window_size, :, 0]
    for i in range(window_size): preds.append(history_buffer[i])
    current_input = torch.tensor(history_buffer).unsqueeze(0).float().to(device) # [1, 10, 64]

    print(f">>> Predicting...")
    with torch.no_grad():
        for t in range(window_size, total_steps):
            prediction = model(current_input) # [1, 1, 64]
            pred_frame = prediction.cpu().numpy()[0, 0]
            preds.append(pred_frame)
            current_input = torch.cat([current_input[:, 1:, :], prediction], dim=1)

    preds = np.array(preds)
    truths = u_data[:, :, 0]
    mse = np.mean((preds[window_size:] - truths[window_size:]) ** 2)
    print(f"✅ MSE: {mse:.4e}")

    # 5. 绘图
    plt.figure(figsize=(15, 5))
    snap_indices = [15, 50, 90]
    for i, idx in enumerate(snap_indices):
        if idx >= total_steps: break
        plt.subplot(1, 3, i+1)
        plt.plot(x_grid, truths[idx], 'k--', label='GT')
        plt.plot(x_grid, preds[idx], 'r-', label='Pred')
        plt.title(f"t={t_grid[idx]:.1f}")
        if i==0: plt.legend()
    
    save_path = os.path.join(CONFIG["output_dir"], "unet_64_result.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()