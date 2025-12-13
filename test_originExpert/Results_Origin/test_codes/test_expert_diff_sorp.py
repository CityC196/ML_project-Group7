"""
位置: test_codes/test_expert_diff_sorp.py
功能: 测试 ExpertModels 文件夹下的官方 PINN 专家模型
输出: ../OUTPUT/diff_sorp_expert/
"""
import os
import sys
import h5py
import numpy as np
import torch
import deepxde as dde
import matplotlib.pyplot as plt

# ===========================
# 1. 配置路径 (Configuration)
# ===========================
CONFIG = {
    # 专家模型权重路径 (相对路径：上一级目录 -> ExpertModels)
    "checkpoint_path": "../ExpertModels/1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt",
    
    # 数据集路径 (相对路径：上一级目录 -> data)
    # 请务必核对文件名是否包含 _0001，如 "1D_diff-sorp_NA_NA_0001.h5"
    "data_path": "../data/1D_diff-sorp_NA_NA.h5", 
    
    # 输出路径 (修改为根目录下的 OUTPUT 文件夹)
    "output_dir": "../OUTPUT/diff_sorp_expert",
    "seed": 0
}

# 确保输出子目录存在
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ===========================
# 2. PDE 定义 (Diff-Sorp)
# ===========================
def pde_diffusion_sorption(x, y):
    D = 5e-4
    por = 0.29
    rho_s = 2880
    k_f = 3.5e-4
    n_f = 0.874

    # DeepXDE Hessian/Jacobian
    du1_xx = dde.grad.hessian(y, x, i=0, j=0)
    du1_t = dde.grad.jacobian(y, x, i=0, j=1)

    u1 = y
    # 加上 1e-6 避免数值不稳定
    retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (u1 + 1e-6) ** (n_f - 1)

    return du1_t - D / retardation_factor * du1_xx

# ===========================
# 3. 数据加载辅助函数
# ===========================
def load_eval_data(filename, seed_idx=0):
    print(f">>> Loading data from {filename}...")
    try:
        with h5py.File(filename, "r") as f:
            seed_str = str(seed_idx).zfill(4)
            if seed_str not in f:
                keys = list(f.keys())
                seed_str = keys[0] if keys else None
                print(f"Warning: Seed {seed_idx} not found, using {seed_str}")
            
            if seed_str is None:
                raise ValueError("No keys found in HDF5 file.")

            group = f[seed_str]
            x = np.array(group["grid"]["x"], dtype=np.float32)
            t = np.array(group["grid"]["t"], dtype=np.float32)
            u_true = np.array(group["data"], dtype=np.float32) 
            
            if u_true.ndim == 3 and u_true.shape[0] == len(t):
                u_true = np.transpose(u_true, (1, 0, 2))
            
            return x, t, u_true
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

# ===========================
# 4. 主函数
# ===========================
def main():
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 500.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    
    data = dde.data.TimePDE(
        geomtime, pde_diffusion_sorption, [ic, bc],
        num_domain=100, num_boundary=100, num_initial=100
    )

    net = dde.nn.FNN([2] + [40] * 6 + [1], "tanh", "Glorot normal")

    def transform_output(x, y):
        return torch.relu(y)
    net.apply_output_transform(transform_output)

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    print(f">>> Loading weights from {CONFIG['checkpoint_path']}...")
    if not os.path.exists(CONFIG['checkpoint_path']):
        print(f"Error: 权重文件不存在: {CONFIG['checkpoint_path']}")
        return

    try:
        model.restore(CONFIG['checkpoint_path'], verbose=1)
        print(">>> 权重加载成功！")
    except Exception as e:
        print(f"Error restoring model: {e}")
        return

    x_grid, t_grid, u_true = load_eval_data(CONFIG["data_path"], CONFIG["seed"])
    
    snapshots = [50.0, 250.0, 450.0]
    
    plt.figure(figsize=(15, 5))
    total_mse = 0
    
    for i, t_val in enumerate(snapshots):
        t_idx = (np.abs(t_grid - t_val)).argmin()
        real_t = t_grid[t_idx]
        
        X_plot = x_grid[:, None]
        T_plot = np.ones_like(X_plot) * real_t
        X_input = np.hstack((X_plot, T_plot))
        
        u_pred = model.predict(X_input)
        u_real = u_true[:, t_idx, 0]
        
        mse = np.mean((u_pred.flatten() - u_real.flatten())**2)
        total_mse += mse
        
        plt.subplot(1, 3, i+1)
        plt.plot(x_grid, u_real, 'k--', label='Ground Truth', linewidth=2)
        plt.plot(x_grid, u_pred, 'r-', label='Expert Pred', linewidth=1.5)
        plt.title(f"t = {real_t:.1f} (MSE={mse:.2e})")
        plt.xlabel("x")
        plt.ylabel("u")
        if i == 0: plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], "expert_test_plot.png")
    plt.savefig(save_path)
    print(f"\n>>> 测试完成！")
    print(f">>> 平均 MSE: {total_mse / 3:.4e}")
    print(f">>> 结果图已保存至: {save_path}")

if __name__ == "__main__":
    main()