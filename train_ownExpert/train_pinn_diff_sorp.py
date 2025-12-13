import os
import time
import argparse
import h5py
import numpy as np
import torch
import deepxde as dde
import matplotlib.pyplot as plt

# ==============================================================================
# [用户配置区域] 修改此处的路径即可生效
# ==============================================================================
INPUT_DATA_PATH = "../data/1D_diff-sorp_NA_NA.h5"
# 输出目录的前缀 (最终目录会加上 seed, 例如: ../training_results/diff_sorp_seed_0)
OUTPUT_DIR_BASE = "../ExpertModels/ExpertModels_own/diff_sorp_pinn_seed" 
# ==============================================================================

# ================= 参数解析 =================
parser = argparse.ArgumentParser(description='Train PINN for Diff-Sorp')
parser.add_argument('--seed', type=int, default=0, help='Index of the sample to train on (e.g. 0)')
parser.add_argument('--epochs', type=int, default=15000, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
args = parser.parse_args()

# ================= 配置 =================
CONFIG = {
    # 引用顶部的变量
    "data_path": INPUT_DATA_PATH, 
    "output_dir": f"{OUTPUT_DIR_BASE}{args.seed}",
    "val_ratio": 0.3,
}
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ================= PDE 定义 =================
def pde_diffusion_sorption(x, y):
    D = 5e-4
    por = 0.29
    rho_s = 2880
    k_f = 3.5e-4
    n_f = 0.874
    
    du1_xx = dde.grad.hessian(y, x, i=0, j=0)
    du1_t = dde.grad.jacobian(y, x, i=0, j=1)

    u1 = y
    retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (u1 + 1e-6) ** (n_f - 1)

    return du1_t - D / retardation_factor * du1_xx

# ================= 数据加载 =================
def load_data(filename, seed_idx):
    print(f"Loading data from {filename}, Seed: {seed_idx} ...")
    with h5py.File(filename, "r") as f:
        seed_str = str(seed_idx).zfill(4)
        if seed_str not in f:
            # 尝试获取第一个key作为兜底
            keys = list(f.keys())
            if keys:
                seed_str = keys[0]
                print(f"Warning: Seed {seed_idx} not found, using {seed_str}")
            else:
                raise ValueError(f"Seed {seed_str} not found and file is empty!")
            
        seed_group = f[seed_str]
        
        x = np.array(seed_group["grid"]["x"], dtype=np.float32)
        t = np.array(seed_group["grid"]["t"], dtype=np.float32)
        u_data = np.array(seed_group["data"], dtype=np.float32) 
        
        X, T = np.meshgrid(x, t, indexing='ij') 
        
        if u_data.shape[0] == len(t): 
             u_data = np.transpose(u_data, (1, 0, 2))
        
        input_data = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        output_data = u_data.flatten()[:, None]
        
        return input_data, output_data, x, t

# ================= 主流程 =================
def main():
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 500.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)
    bc_d = dde.icbc.DirichletBC(
        geomtime, lambda x: 1.0, lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0)
    )
    def operator_bc(inputs, outputs, X):
        D = 5e-4
        du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        return outputs - D * du_x
    bc_d2 = dde.icbc.OperatorBC(
        geomtime, operator_bc, lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0)
    )

    input_data, output_data, grid_x, grid_t = load_data(CONFIG["data_path"], args.seed)
    
    num_samples = int(len(input_data) * CONFIG["val_ratio"])
    idx = np.random.choice(len(input_data), num_samples, replace=False)
    bc_data = dde.icbc.PointSetBC(input_data[idx], output_data[idx])

    data = dde.data.TimePDE(
        geomtime, pde_diffusion_sorption, [ic, bc_d, bc_d2, bc_data],
        num_domain=1000, num_boundary=1000, num_initial=5000,
    )
    
    net = dde.nn.FNN([2] + [40] * 6 + [1], "tanh", "Glorot normal")
    net.apply_output_transform(lambda x, y: torch.relu(y))

    model = dde.Model(data, net)
    model.compile("adam", lr=args.lr)
    
    print(f"\n>>> Start Training Seed {args.seed} (Epochs={args.epochs})...")
    ckpt_path = os.path.join(CONFIG["output_dir"], "model.ckpt")
    
    # 【修改点】period=5000, 每5000步保存一次
    checker = dde.callbacks.ModelCheckpoint(ckpt_path, save_better_only=True, period=5000)
    
    start_time = time.time()
    # display_every=1000 保持不变，方便你在屏幕上看进度
    # 这里的 train 只是演示，如果您已经训练好，可以注释掉这行直接 predict
    losshistory, _ = model.train(epochs=args.epochs, display_every=1000, callbacks=[checker])
    print(f">>> Training finished in {time.time() - start_time:.2f}s")

    dde.utils.save_loss_history(losshistory, os.path.join(CONFIG["output_dir"], "loss.dat"))

    # ==================================================
    # [修改] 3个时刻的对比图 (True vs Pred)
    # ==================================================
    print(">>> Generating 3-Snapshot Comparison Plots...")
    plt.figure(figsize=(18, 5)) # 宽一点，放3张子图
    
    # 选取3个典型时刻: 早期(50), 中期(250), 晚期(450)
    # 假设总时长是 500
    snapshots = [50.0, 250.0, 450.0]
    u_matrix = output_data.reshape(len(grid_x), len(grid_t))
    
    for i, t_val in enumerate(snapshots):
        # 1. 找到最接近的时间索引
        t_idx = (np.abs(grid_t - t_val)).argmin()
        real_t = grid_t[t_idx]
        
        # 2. 构造测试输入 (x, t)
        # x 从 0 到 1, t 固定为 real_t
        X_test = np.hstack((grid_x[:, None], np.ones_like(grid_x[:, None]) * real_t))
        
        # 3. 预测
        u_pred = model.predict(X_test)
        u_true = u_matrix[:, t_idx]
        
        # 4. 绘图
        plt.subplot(1, 3, i+1)
        plt.plot(grid_x, u_true, 'k-', linewidth=2, label='Ground Truth')
        plt.plot(grid_x, u_pred, 'r--', linewidth=2, label='PINN Prediction')
        
        plt.title(f"Time: {real_t:.1f}s")
        plt.xlabel("Position (x)")
        if i == 0:
            plt.ylabel("Concentration (u)")
            plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle(f"Diffusion-Sorption PINN Results (Seed {args.seed})", fontsize=16)
    plt.tight_layout()
    save_file = os.path.join(CONFIG["output_dir"], "result_3_snapshots.png")
    plt.savefig(save_file)
    print(f"Result saved to {save_file}")

if __name__ == "__main__":
    main()