"""
位置: test_codes/test_expert_unet_advection.py
功能: 测试 Advection U-Net 专家模型 (滑动窗口预测)
输出: ../OUTPUT/advection_unet_test/
"""
import os
import sys
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置区域
# ==========================================
CONFIG = {
    # 模型路径
    "model_path": "../ExpertModels/1D_Advection_Sols_beta1.0_Unet-PF-20.pt",
    
    # 数据集路径
    "data_path": "../data/1D_Advection_Sols_beta1.0.hdf5",
    
    # 输出目录
    "output_dir": "../OUTPUT/advection_unet_test",
    
    # 【重要】模型参数
    # 如果报错 "size mismatch"，请根据报错信息修改这里
    # 常见值: 1, 4, 10, 12
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

# ==========================================
# 3. 数据加载 (含 Advection 专用修复)
# ==========================================
def load_advection_data(filename, seed_idx):
    print(f">>> Loading data from {filename} ...")
    with h5py.File(filename, "r") as f:
        # 读取 Grid
        x = np.array(f["x-coordinate"], dtype=np.float32)
        t = np.array(f["t-coordinate"], dtype=np.float32)
        
        # 读取 Data
        if "tensor" in f:
            u_raw = np.array(f["tensor"][seed_idx], dtype=np.float32)
        elif "data" in f:
            u_raw = np.array(f["data"][seed_idx], dtype=np.float32)
        else:
            raise ValueError("Cannot find 'tensor' or 'data'")

        # === 维度自动对齐 (同 PINN 逻辑) ===
        # 目标: [Time, Space, Channel] -> (Nt, Nx, 1)
        
        # 1. 先转成 [Space, Time]
        if u_raw.shape == (len(t), len(x)):
            u_matrix = u_raw.T
        elif u_raw.shape == (len(x), len(t)):
            u_matrix = u_raw
        else:
            # 暴力截断
            min_t = min(len(t), u_raw.shape[0], u_raw.shape[1])
            t = t[:min_t]
            if u_raw.shape[0] == len(x):
                u_matrix = u_raw[:, :min_t]
            else:
                u_matrix = u_raw[:min_t, :].T
        
        # 2. 转成 U-Net 需要的 [Time, Space, 1]
        # u_matrix 是 [Space, Time] -> 转置为 [Time, Space]
        u_final = u_matrix.T 
        # 增加 Channel 维
        u_final = u_final[..., None] # (Nt, Nx, 1)
        
        print(f"    Final Data Shape: {u_final.shape}")
        return u_final, x, t

# ==========================================
# 4. 主测试流程
# ==========================================
def main():
    device = torch.device(CONFIG["device"])
    
    # A. 加载数据
    try:
        u_data, x_grid, t_grid = load_advection_data(CONFIG["data_path"], CONFIG["seed"])
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # B. 加载模型
    print(f">>> Loading U-Net (In={CONFIG['in_channels']})...")
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
        print(f"\n❌ 权重加载失败: {e}")
        print("💡 提示: 请检查 'size mismatch' 信息。")
        print(f"   如果提示 shape [32, X, 3]，请将脚本中的 CONFIG['in_channels'] 改为 X。")
        return

    model.eval()

    # C. 滑动窗口预测
    window_size = CONFIG["in_channels"]
    total_steps = len(t_grid)
    
    if total_steps <= window_size:
        print("❌ 时间步太少，无法运行滑动窗口。")
        return

    preds = []
    # 填充历史
    # u_data: [Time, Space, 1]
    history_buffer = u_data[:window_size, :, 0] # [Window, Space]
    for i in range(window_size): preds.append(history_buffer[i])
    
    # 构造初始 Tensor: [1, Window, Space]
    current_input = torch.tensor(history_buffer).unsqueeze(0).float().to(device)

    print(f">>> Predicting from step {window_size} to {total_steps}...")
    with torch.no_grad():
        for t in range(window_size, total_steps):
            # U-Net 1D forward: Input [Batch, Channel, Space] -> Output [Batch, Channel, Space]
            prediction = model(current_input) 
            
            # 取结果
            pred_frame = prediction.cpu().numpy()[0, 0] # [Space]
            preds.append(pred_frame)
            
            # 更新窗口: 移除最旧(0)，追加最新
            # cat dim=1 (Channel维是时间)
            current_input = torch.cat([current_input[:, 1:, :], prediction], dim=1)

    preds = np.array(preds)
    truths = u_data[:, :, 0]
    
    # D. 误差与绘图
    valid_steps = preds.shape[0]
    mse = np.mean((preds[window_size:] - truths[window_size:valid_steps]) ** 2)
    print(f"✅ Prediction MSE: {mse:.4e}")

    plt.figure(figsize=(15, 5))
    # 选取 25%, 50%, 75% 时间点
    indices = [int(total_steps*0.25), int(total_steps*0.5), int(total_steps*0.75)]
    
    for i, idx in enumerate(indices):
        if idx >= valid_steps: break
        
        plt.subplot(1, 3, i+1)
        plt.plot(x_grid, truths[idx], 'k--', label='GT')
        plt.plot(x_grid, preds[idx], 'r-', label='Pred')
        plt.title(f"t={t_grid[idx]:.2f}")
        plt.ylim([-1.2, 1.2]) # Advection 范围通常在 -1 到 1
        if i==0: plt.legend()
    
    save_path = os.path.join(CONFIG["output_dir"], "advection_unet_result.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Result saved: {save_path}")

if __name__ == "__main__":
    main()