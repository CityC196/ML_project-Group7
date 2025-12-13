"""
位置: test_codes/test_expert_unet_3d_cfd.py
功能: 测试 3D CFD U-Net 模型 (修复输入通道数问题，启用历史滑动窗口)
"""
import os
import sys
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")

try:
    from pdebench.models.unet.unet import UNet3d
except ImportError:
    print("❌ 无法导入 UNet3d")
    sys.exit(1)

# ==========================================
# 1. 配置区域
# ==========================================
CONFIG = {
    "model_path": "../ExpertModels/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train_Unet-PF-20.pt",
    "data_path": "../data/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5",
    "output_dir": "../OUTPUT/cfd_3d_unet_test",
    "rdc": 4,
    "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # [关键修复] 历史步数
    # 模型期望 50 通道输入，物理量有 5 个 -> 50 / 5 = 10 步历史
    "history_steps": 10 
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==========================================
# 2. 数据加载
# ==========================================
def load_data_3d(filename, seed_idx, rdc=4):
    print(f">>> Loading data from {filename} (Reduce {rdc}x)...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")

    try:
        with h5py.File(filename, "r") as f:
            rho = f['density'][seed_idx] 
            vx = f['Vx'][seed_idx]
            vy = f['Vy'][seed_idx]
            vz = f['Vz'][seed_idx]
            p = f['pressure'][seed_idx]
            
            raw_data = np.stack([rho, vx, vy, vz, p], axis=-1) 
            u_down = raw_data[:, ::rdc, ::rdc, ::rdc, :]
            # [Time, Channel, D, H, W]
            u_torch = np.transpose(u_down, (0, 4, 1, 2, 3))
            return u_torch
    except Exception as e:
        print(f"❌ HDF5 读取错误: {e}")
        raise e

# ==========================================
# 3. 主流程
# ==========================================
def main():
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # --- A. 加载数据 ---
    try:
        # u_data shape: [Time, 5, D, H, W]
        u_data = load_data_3d(CONFIG["data_path"], CONFIG["seed"], CONFIG["rdc"])
        print(f"Data Loaded. Shape: {u_data.shape}")
    except Exception as e:
        print(f"❌ 程序终止: {e}")
        return

    n_vars = u_data.shape[1]      # 5
    total_time = u_data.shape[0]  # 21
    hist_steps = CONFIG["history_steps"] # 10

    # 检查数据是否足够长
    if total_time <= hist_steps:
        print(f"❌ 数据时间步 ({total_time}) 少于所需的历史步数 ({hist_steps})，无法运行。")
        return

    # --- B. 初始化模型 ---
    # Input Channels = 5 * 10 = 50
    # Output Channels = 5 (预测下一步)
    in_channels = n_vars * hist_steps
    out_channels = n_vars
    
    print(f">>> Initializing UNet3d (in={in_channels}, out={out_channels})...")
    model = UNet3d(in_channels=in_channels, out_channels=out_channels).to(device)

    if not os.path.exists(CONFIG["model_path"]):
        print(f"❌ 模型文件不存在: {CONFIG['model_path']}")
        return

    # --- C. 加载权重 ---
    try:
        checkpoint = torch.load(CONFIG["model_path"], map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print("✅ 权重加载成功!")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        print(f"提示: 请检查 history_steps 是否正确 (当前设置: {hist_steps})")
        return

    model.eval()

    # --- D. 预测 (滑动窗口) ---
    print(f">>> Starting prediction from t={hist_steps} to {total_time-1}...")
    
    preds = []
    # 填充前 10 帧 (Ground Truth) 作为“热身”
    # preds 列表存储 numpy 数组 [5, D, H, W]
    for i in range(hist_steps):
        preds.append(u_data[i])

    with torch.no_grad():
        # 从第 10 步开始预测 (因为需要 0~9 作为输入)
        for t in range(hist_steps, total_time):
            # 1. 构造输入: 取过去 10 帧
            # 列表切片: [t-10, t-9, ..., t-1]
            input_frames = preds[t - hist_steps : t] 
            
            # 2. 拼接通道: list of [5, D, H, W] -> [50, D, H, W]
            input_tensor = np.concatenate(input_frames, axis=0)
            
            # 3. 增加 Batch 维 -> [1, 50, D, H, W]
            input_tensor = torch.tensor(input_tensor).unsqueeze(0).float().to(device)
            
            # 4. 模型预测 -> [1, 5, D, H, W]
            pred_next = model(input_tensor)
            
            # 5. 存入结果 (用于下一步输入)
            preds.append(pred_next.cpu().numpy()[0])

    preds = np.array(preds) # [Time, 5, D, H, W]
    truths = u_data

    # --- E. 误差与可视化 ---
    # 只计算预测部分 (从 hist_steps 开始) 的 MSE
    valid_preds = preds[hist_steps:]
    valid_truths = truths[hist_steps:]
    
    if len(valid_preds) > 0:
        mse = np.mean((valid_preds - valid_truths) ** 2)
        print(f"✅ Prediction MSE (t={hist_steps}~end): {mse:.4e}")
    else:
        print("⚠️ 没有产生预测数据 (时间步不足)")

    # 切片可视化
    print(">>> Generating slices...")
    try:
        c_idx = 0 
        z_idx = truths.shape[2] // 2 
        
        # 选取 3 个时间点: 开始预测时, 中间, 结束
        # 注意: t 是全局时间索引
        vis_indices = [hist_steps, (hist_steps + total_time)//2, total_time-1]
        
        plt.figure(figsize=(15, 6))
        for i, t_idx in enumerate(vis_indices):
            if t_idx >= total_time: continue

            # GT
            plt.subplot(2, 3, i+1)
            plt.imshow(truths[t_idx, c_idx, z_idx, :, :], cmap='jet')
            plt.title(f"GT (t={t_idx})")
            plt.axis('off'); plt.colorbar()
            
            # Pred
            plt.subplot(2, 3, i+4)
            plt.imshow(preds[t_idx, c_idx, z_idx, :, :], cmap='jet')
            plt.title(f"Pred (t={t_idx})")
            plt.axis('off'); plt.colorbar()

        plt.tight_layout()
        save_path = os.path.join(CONFIG["output_dir"], "cfd_3d_result.png")
        plt.savefig(save_path)
        print(f"✅ Plot saved to: {save_path}")
        
    except Exception as e:
        print(f"⚠️ 作图失败: {e}")

if __name__ == "__main__":
    main()