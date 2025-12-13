"""
位置: test/test_expert_fno_3d_cfd.py
功能: 测试 3D CFD 的 FNO 专家模型
修复: 修正 modes=12 以匹配 checkpoint 权重。
"""
import os
import sys
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 (已根据报错修正) =================
CONFIG = {
    "model_path": "../ExpertModels/ExpertModels_official/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train_FNO.pt",
    "data_path": "../data/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", 
    "output_dir": "OUTPUT/cfd_3d_fno_test",
    
    # [关键修正] 根据报错信息调整参数
    "modes1": 12, # 原来是 8 -> Checkpoint 是 12
    "modes2": 12,
    "modes3": 12,
    "width": 20,  # 匹配
    "initial_step": 10,
    "num_channels": 5, 
    
    "seed": 0,
    "device": "cpu"
}

sys.path.append("..") 
try:
    from pdebench.models.fno.fno import FNO3d
except ImportError:
    sys.exit("❌ 错误: 无法导入 pdebench。")

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ================= 智能文件加载 =================
def find_file_with_ext_fallback(filepath):
    if os.path.exists(filepath): return filepath
    base, ext = os.path.splitext(filepath)
    alt = base + '.hdf5' if ext == '.h5' else base + '.h5'
    if os.path.exists(alt): return alt
    # 模糊匹配
    dirname = os.path.dirname(filepath)
    if os.path.exists(dirname):
        for f in os.listdir(dirname):
            if "3D_CFD" in f and (f.endswith('.h5') or f.endswith('.hdf5')):
                return os.path.join(dirname, f)
    raise FileNotFoundError(f"❌ 找不到文件: {filepath}")

def load_3d_cfd_data(filename, seed_idx):
    filename = find_file_with_ext_fallback(filename)
    print(f">>> Loading 3D data from {filename} ...")

    with h5py.File(filename, "r") as f:
        if "train" in f:
            group = f["train"]
        else:
            group = f

        try:
            rho = np.array(group["density"][seed_idx], dtype=np.float32)
            p   = np.array(group["pressure"][seed_idx], dtype=np.float32)
            vx  = np.array(group["Vx"][seed_idx], dtype=np.float32)
            vy  = np.array(group["Vy"][seed_idx], dtype=np.float32)
            vz  = np.array(group["Vz"][seed_idx], dtype=np.float32)
        except IndexError:
            print(f"⚠️ Seed {seed_idx} 越界，使用索引 0")
            rho = np.array(group["density"][0], dtype=np.float32)
            p   = np.array(group["pressure"][0], dtype=np.float32)
            vx  = np.array(group["Vx"][0], dtype=np.float32)
            vy  = np.array(group["Vy"][0], dtype=np.float32)
            vz  = np.array(group["Vz"][0], dtype=np.float32)

        u_data = np.stack([rho, p, vx, vy, vz], axis=-1)
        
        nt, nx, ny, nz, nc = u_data.shape
        print(f"   Data Shape: {u_data.shape} (Time, X, Y, Z, C)")
        
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        
    return u_data, (x, y, z)

# ================= 主流程 =================
def main():
    device = torch.device(CONFIG["device"])
    print(f"🚀 Running on device: {device}")
    print(f"📋 Config: Modes={CONFIG['modes1']}, Width={CONFIG['width']}")

    try:
        u_data, (x, y, z) = load_3d_cfd_data(CONFIG["data_path"], CONFIG["seed"])
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 初始化模型
    model = FNO3d(
        num_channels=CONFIG["num_channels"],
        modes1=CONFIG["modes1"], modes2=CONFIG["modes2"], modes3=CONFIG["modes3"],
        width=CONFIG["width"],
        initial_step=CONFIG["initial_step"]
    ).to(device)

    # 加载权重
    if os.path.exists(CONFIG["model_path"]):
        try:
            checkpoint = torch.load(CONFIG["model_path"], map_location=device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print("✅ 模型权重加载成功!")
        except RuntimeError as e:
            print(f"❌ 权重尺寸不匹配: {e}")
            print("💡 请检查 CONFIG 中的 modes/width。")
            return
        except Exception as e:
            print(f"❌ 加载错误: {e}")
            return
    else:
        print(f"❌ 模型文件未找到: {CONFIG['model_path']}")
        return

    model.eval()

    # 构造 Grid
    X, Y, Z_grid = np.meshgrid(x, y, z, indexing='ij')
    grid_numpy = np.stack([X, Y, Z_grid], axis=-1)
    grid_tensor = torch.tensor(grid_numpy, dtype=torch.float32).unsqueeze(0).to(device).contiguous()

    # 准备推理 (测试 5 步)
    test_steps = 5 
    start_t = CONFIG["initial_step"]
    
    input_list = []
    for i in range(start_t):
        input_list.append(u_data[i]) 
    
    current_input_np = np.concatenate(input_list, axis=-1)
    current_input = torch.tensor(current_input_np, dtype=torch.float32).unsqueeze(0).to(device).contiguous()

    print(f">>> Predicting 3D CFD (Testing {test_steps} steps)...")
    
    preds = [] 
    truths = []
    
    with torch.no_grad():
        for t in range(test_steps):
            target_t = start_t + t
            
            prediction = model(current_input.contiguous(), grid_tensor).squeeze(-2)
            
            pred_np = prediction[0].cpu().numpy()
            preds.append(pred_np)
            truths.append(u_data[target_t])
            
            c_dim = CONFIG["num_channels"]
            current_input = torch.cat([current_input[..., c_dim:], prediction], dim=-1)
            
            print(f"   Step {t+1}/{test_steps} done.", end="\r")

    print("\n✅ Inference Complete.")

    # 可视化
    z_slice = len(z) // 2
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(truths[-1][:, :, z_slice, 0], cmap='jet', origin='lower')
    plt.title(f"GT Density (z={z_slice})")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(preds[-1][:, :, z_slice, 0], cmap='jet', origin='lower')
    plt.title(f"Pred Density (z={z_slice})")
    plt.colorbar()
    
    save_path = os.path.join(CONFIG["output_dir"], "cfd_3d_result.png")
    plt.savefig(save_path)
    print(f"🖼️ Saved slice view to {save_path}")

if __name__ == "__main__":
    main()