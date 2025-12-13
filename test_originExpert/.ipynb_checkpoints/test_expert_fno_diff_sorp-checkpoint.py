"""
位置: test/test_expert_fno_diff_sorp.py
功能: 测试 1D Diff-Sorp 的 FNO 专家模型 (修复 cuFFT 错误与内存连续性问题)
"""
import os
import sys
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
CONFIG = {
    "model_path": "../ExpertModels/ExpertModels_official/1D_diff-sorp_NA__FNO.pt",
    "data_path": "../data/1D_diff-sorp_NA_NA.h5",
    "output_dir": "OUTPUT/diff_sorp_fno_test",
    
    "modes": 16,
    "width": 64,
    "initial_step": 10,
    "num_channels": 1,
    
    "target_resolution": 64,  
    "seed": 0,
    
    # 🔴 修改这里：强制使用 cpu，绕过 CUDA FFT 错误
    "device": "cpu" 
    # "device": "cuda" if torch.cuda.is_available() else "cpu"
}

sys.path.append("..") 
try:
    from pdebench.models.fno.fno import FNO1d
except ImportError:
    sys.exit("❌ 错误: 无法导入 pdebench。")

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ================= 智能数据加载 =================
def load_diff_sorp_data(filename, seed_idx, target_resolution=None):
    print(f">>> Loading data from {filename} ...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} 不存在")

    with h5py.File(filename, "r") as f:
        seed_str = str(seed_idx).zfill(4)
        if seed_str not in f: 
            seed_str = list(f.keys())[0]
            print(f"⚠️ Seed {seed_idx} not found, using {seed_str} instead.")
            
        group = f[seed_str]
        t = np.array(group["grid"]["t"], dtype=np.float32)
        x = np.array(group["grid"]["x"], dtype=np.float32)
        u_data = np.array(group["data"], dtype=np.float32) 
        
        # --- 1. 维度修正 ---
        if u_data.ndim == 3:
            u_data = u_data.squeeze(-1)
            
        nx = len(x)
        nt = len(t)
        
        print(f"   Raw Data Shape: {u_data.shape}, Grid x: {nx}, Grid t: {nt}")
        
        if u_data.shape[0] == nt and u_data.shape[1] == nx:
            print("   -> Detected (Time, Spatial), Transposing to (Spatial, Time)...")
            u_data = u_data.T
        elif u_data.shape[0] == nx and u_data.shape[1] == nt:
            print("   -> Detected (Spatial, Time), No transpose needed.")
        else:
            raise ValueError(f"❌ 数据维度 {u_data.shape} 与 x({nx}), t({nt}) 不匹配!")

    # --- 2. 空间降采样 ---
    current_res = u_data.shape[0] # Spatial
    if target_resolution and current_res != target_resolution:
        print(f"⚠️ Resolution Mismatch: Spatial({current_res}) vs Model({target_resolution})")
        
        step = current_res // target_resolution
        print(f"   -> Performing Spatial Downsampling (step={step})...")
        
        # 使用 np.copy 确保降采样后的数组在内存中是连续的
        u_data = np.ascontiguousarray(u_data[::step, :])
        x = x[::step]
        
        print(f"   -> New Data Shape: {u_data.shape}")

    return u_data, x, t

# ================= 主流程 =================
def main():
    device = torch.device(CONFIG["device"])
    print(f"🚀 Running on device: {device}")

    # 1. 加载数据
    try:
        u_data, x_grid, t_grid = load_diff_sorp_data(CONFIG["data_path"], CONFIG["seed"], CONFIG["target_resolution"])
    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return
        
    # [关键] 检查 NaNs
    if np.isnan(u_data).any():
        print("❌ 错误: 输入数据包含 NaN，这会导致 cuFFT 崩溃。")
        return

    # 2. 初始化模型
    model = FNO1d(
        num_channels=CONFIG["num_channels"],
        modes=CONFIG["modes"],
        width=CONFIG["width"],
        initial_step=CONFIG["initial_step"]
    ).to(device)

    # 3. 加载权重
    model_path = CONFIG["model_path"]
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("✅ 模型权重加载成功!")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    model.eval()

    # 4. 准备推理
    initial_step = CONFIG["initial_step"]
    time_dim = u_data.shape[1]
    
    # 构造 Grid: [1, S, 1]
    # 使用 .contiguous() 确保内存连续
    grid_tensor = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device).contiguous()
    
    # 构造 Input: [1, S, T]
    input_full = torch.tensor(u_data, dtype=torch.float32).unsqueeze(0).to(device).contiguous()
    
    # 取前 initial_step 步作为初始历史
    current_input = input_full[:, :, :initial_step].contiguous()

    preds = []
    # 填入历史
    for i in range(initial_step):
        preds.append(u_data[:, i])

    print(f">>> Starting Autoregressive Inference (Steps: {initial_step} -> {time_dim})...")
    
    # 清空缓存，给 FFT 腾出干净的环境
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    try:
        with torch.no_grad():
            for t in range(initial_step, time_dim):
                # 再次强制 contiguous，防止 cat/slice 产生碎片
                curr_in_contig = current_input.contiguous()
                
                # FNO Forward
                # 输出: [1, S, 1, 1] -> squeeze -> [1, S, 1]
                prediction = model(curr_in_contig, grid_tensor).squeeze(-2) 
                
                # 存结果
                pred_val = prediction[0, :, 0].cpu().numpy()
                preds.append(pred_val)
                
                # 更新历史: 丢弃最早的一帧，加入预测的一帧
                # [1, S, 10] -> [1, S, 9] + [1, S, 1]
                current_input = torch.cat([current_input[:, :, 1:], prediction], dim=-1)

                if t % 20 == 0:
                    print(f"   Step {t}/{time_dim}", end="\r")

    except RuntimeError as e:
        print(f"\n❌ 推理过程中发生错误: {e}")
        if "CUFFT" in str(e):
            print("💡 提示: CUFFT 错误通常与内存布局有关。如果问题持续，尝试在 CPU 上运行 (CONFIG['device']='cpu') 以排除 GPU 驱动问题。")
        return

    print(f"\n✅ Inference Done. Total frames: {len(preds)}")
    
    # 5. 后处理与评估
    preds = np.array(preds).T 
    
    # 对齐长度
    min_len = min(preds.shape[1], u_data.shape[1])
    preds = preds[:, :min_len]
    u_data_trunc = u_data[:, :min_len]

    # 计算 MSE
    mse = np.mean((preds[:, initial_step:] - u_data_trunc[:, initial_step:]) ** 2)
    print(f"📊 MSE: {mse:.4e}")

    # 6. 绘图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plot_indices = [int(min_len*0.2), int(min_len*0.5), int(min_len*0.8)]
    for idx in plot_indices:
        if idx < min_len:
            plt.plot(x_grid, u_data_trunc[:, idx], 'k--', alpha=0.5, label='GT' if idx==plot_indices[0] else None)
            plt.plot(x_grid, preds[:, idx], '-', label=f't={t_grid[idx]:.1f}')
    plt.title(f"Snapshots (MSE={mse:.2e})")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.imshow(u_data_trunc, aspect='auto', cmap='jet', origin='lower')
    plt.title("Ground Truth")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(preds, aspect='auto', cmap='jet', origin='lower')
    plt.title("FNO Prediction")
    plt.xlabel("t")
    plt.colorbar()
    
    save_path = os.path.join(CONFIG["output_dir"], "diff_sorp_fno_result.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ Saved plot to {save_path}")

if __name__ == "__main__":
    main()