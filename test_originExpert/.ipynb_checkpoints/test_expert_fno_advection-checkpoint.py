"""
位置: test/test_expert_fno_advection.py
功能: 测试 1D Advection FNO 模型 (修复数据加载为空/常数的问题)
"""
import os
import sys
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
CONFIG = {
    "model_path": "../ExpertModels/ExpertModels_official/1D_Advection_Sols_beta1.0_FNO.pt",
    "data_path": "../data/1D_Advection_Sols_beta1.0.hdf5", 
    "output_dir": "OUTPUT/advection_fno_test",
    
    # 根据之前的报错修正的模型参数
    "modes": 12,
    "width": 20,
    "initial_step": 10,
    "num_channels": 1,
    
    # 🔴 [关键修改] 改用 Seed 100，避开可能的空样本 Seed 0
    "seed": 100, 
    
    # 调试参数
    "time_stride": 1,   # 保持为1，如果预测速度不对再调整
    "normalize_grid": True,
    
    "device": "cpu"
}

sys.path.append("..") 
try:
    from pdebench.models.fno.fno import FNO1d
except ImportError:
    sys.exit("❌ 错误: 无法导入 pdebench。")

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ================= 数据加载 =================
def find_file_with_ext_fallback(filepath):
    if os.path.exists(filepath): return filepath
    base, ext = os.path.splitext(filepath)
    alt = base + '.hdf5' if ext == '.h5' else base + '.h5'
    if os.path.exists(alt): return alt
    raise FileNotFoundError(f"❌ 找不到文件: {filepath}")

def load_advection_data(filename, seed_idx, time_stride=1):
    filename = find_file_with_ext_fallback(filename)
    print(f">>> Loading data from {filename} ...")

    with h5py.File(filename, "r") as f:
        # 优先读取 tensor
        if "tensor" in f:
            dset = f["tensor"]
            total_samples = dset.shape[0]
            
            # 🔄 自动寻找非空样本逻辑
            # 如果指定的 seed 数据标准差为 0 (常数)，则向后寻找
            for probe_seed in range(seed_idx, min(seed_idx + 10, total_samples)):
                temp_data = np.array(dset[probe_seed], dtype=np.float32)
                if temp_data.std() > 1e-5: # 数据有波动
                    if probe_seed != seed_idx:
                        print(f"⚠️ Seed {seed_idx} 看起来是空数据(std=0)，自动切换到 Seed {probe_seed}")
                    u_data = temp_data
                    seed_idx = probe_seed # 更新当前使用的 seed
                    break
            else:
                # 如果循环结束都没 break，说明全是空的，只能硬着头皮用
                print("❌ 警告: 连续10个样本都是常数! 数据集可能损坏。")
                u_data = np.array(dset[seed_idx], dtype=np.float32)

            # 读取坐标
            if "x-coordinate" in f:
                x = np.array(f["x-coordinate"], dtype=np.float32)
                t = np.array(f["t-coordinate"], dtype=np.float32)
            else:
                # 尝试从 grid 读取
                 x = np.array(f["grid"]["x"], dtype=np.float32)
                 t = np.array(f["grid"]["t"], dtype=np.float32)
        
        elif "data" in f:
             # 备用逻辑
             u_data = np.array(f["data"], dtype=np.float32)
             x = np.array(f["grid"]["x"], dtype=np.float32)
             t = np.array(f["grid"]["t"], dtype=np.float32)
        else:
             raise KeyError(f"Unknown H5 keys: {list(f.keys())}")

    # --- 1. 维度修正 (强制对齐) ---
    # Advection 标准: Spatial=1024. 
    # 我们根据 1024 这个特征值来锁定维度。
    
    spatial_dim_std = 1024
    
    if u_data.ndim == 3: u_data = u_data.squeeze(-1)
    
    print(f"   Raw Sample Shape: {u_data.shape}")
    
    # 逻辑: 只要有一个维度是 1024，就把那个维度当作 Spatial，放到第0维
    if u_data.shape[0] == spatial_dim_std:
        print("   -> Dim 0 is Spatial (1024). Keep as (Spatial, Time).")
    elif u_data.shape[1] == spatial_dim_std:
        print("   -> Dim 1 is Spatial (1024). Transpose to (Spatial, Time).")
        u_data = u_data.T
    else:
        print("⚠️ 无法通过 1024 特征识别维度，尝试匹配 Grid 长度...")
        if u_data.shape[0] == len(x):
            pass
        elif u_data.shape[1] == len(x):
            u_data = u_data.T
    
    # 此时 u_data 必须是 (Spatial, Time)
    # 截断 Grid 以匹配数据
    if len(t) != u_data.shape[1]:
        print(f"⚠️ Trimming t: {len(t)} -> {u_data.shape[1]}")
        t = t[:u_data.shape[1]]
        
    # --- 2. 检查数据有效性 ---
    print(f"📊 Data Stats: Min={u_data.min():.4f}, Max={u_data.max():.4f}, Std={u_data.std():.4f}")
    if u_data.std() < 1e-5:
        print("❌❌❌ 严重错误: 加载的数据仍然是常数！请检查 HDF5 文件是否损坏。")
        sys.exit(1)

    # --- 3. 时间步长 ---
    if time_stride > 1:
        u_data = u_data[:, ::time_stride]
        t = t[::time_stride]

    return u_data, x, t

# ================= 主流程 =================
def main():
    device = torch.device(CONFIG["device"])
    print(f"🚀 Running on device: {device}")

    try:
        u_data, x_grid, t_grid = load_advection_data(CONFIG["data_path"], CONFIG["seed"], CONFIG["time_stride"])
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
        
    # Grid 归一化
    if CONFIG["normalize_grid"]:
        x_grid = (x_grid - x_grid.min()) / (x_grid.max() - x_grid.min())

    # 模型初始化
    model = FNO1d(CONFIG["num_channels"], CONFIG["modes"], CONFIG["width"], CONFIG["initial_step"]).to(device)
    
    if os.path.exists(CONFIG["model_path"]):
        checkpoint = torch.load(CONFIG["model_path"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        print("✅ 模型权重加载成功!")
    else:
        print(f"❌ 模型文件不存在: {CONFIG['model_path']}")
        return

    model.eval()

    # 推理
    initial_step = CONFIG["initial_step"]
    time_dim = u_data.shape[1]
    
    grid_tensor = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device).contiguous()
    input_full = torch.tensor(u_data, dtype=torch.float32).unsqueeze(0).to(device).contiguous()
    current_input = input_full[:, :, :initial_step].contiguous()

    preds = [u_data[:, i] for i in range(initial_step)]

    print(f">>> Predicting Advection (Steps {initial_step} -> {time_dim})...")
    with torch.no_grad():
        for t in range(initial_step, time_dim):
            pred = model(current_input.contiguous(), grid_tensor).squeeze(-2) 
            preds.append(pred[0, :, 0].cpu().numpy())
            current_input = torch.cat([current_input[:, :, 1:], pred], dim=-1)

    # 评估
    preds = np.array(preds).T
    # 截断对齐
    min_len = min(preds.shape[1], u_data.shape[1])
    preds = preds[:, :min_len]
    u_data = u_data[:, :min_len]

    mse = np.mean((preds[:, initial_step:] - u_data[:, initial_step:]) ** 2)
    print(f"📊 MSE: {mse:.4e}")

    # 绘图
    plt.figure(figsize=(15, 5))
    
    # 热力图对比
    plt.subplot(1, 3, 1)
    plt.imshow(u_data, aspect='auto', cmap='jet', origin='lower')
    plt.title("Ground Truth (Heatmap)")
    plt.ylabel("Spatial x")
    plt.xlabel("Time t")
    
    plt.subplot(1, 3, 2)
    plt.imshow(preds, aspect='auto', cmap='jet', origin='lower')
    plt.title(f"Prediction (MSE={mse:.2e})")
    plt.xlabel("Time t")
    
    # 波形切片 (重要: 检查是否为常数)
    plt.subplot(1, 3, 3)
    # 画初始、中间、结尾三个时刻
    t_indices = [initial_step, time_dim // 2, time_dim - 5]
    colors = ['r', 'g', 'b']
    for i, t_idx in enumerate(t_indices):
        if t_idx < min_len:
            plt.plot(x_grid, u_data[:, t_idx], '--', color=colors[i], label=f'GT t={t_idx}', alpha=0.6)
            plt.plot(x_grid, preds[:, t_idx], '-', color=colors[i], label=f'Pred t={t_idx}')
            
    plt.title("Wave Profile Snapshots")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(CONFIG["output_dir"], "advection_result.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ Saved: {save_path}")

if __name__ == "__main__":
    main()