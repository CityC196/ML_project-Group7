import os
import time
import argparse
import h5py
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
import torch

# ==============================================================================
# [用户配置区域] 修改此处的路径即可生效
# ==============================================================================
INPUT_DATA_PATH = "../data/1D_Advection_Sols_beta1.0.hdf5"
# 输出目录的前缀 (最终目录会加上 seed, 例如: ../training_results/advection_seed_0)
OUTPUT_DIR_BASE = "../ExpertModels/ExpertModels_own/advection_pinn_seed"
# ==============================================================================

# ================= 参数解析 =================
parser = argparse.ArgumentParser(description='Train PINN for Advection')
parser.add_argument('--seed', type=int, default=0, help='Index of the sample')
parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter')
parser.add_argument('--epochs', type=int, default=20000, help='Training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
args = parser.parse_args()

# ================= 配置 =================
CONFIG = {
    # 引用顶部的变量
    "data_path": INPUT_DATA_PATH, 
    "output_dir": f"{OUTPUT_DIR_BASE}{args.seed}",
    "val_ratio": 0.5,
}
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ================= PDE 定义 =================
def pde_advection(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    return dy_t + args.beta * dy_x

# ================= 数据加载 =================
def load_data(filename, seed_idx):
    print(f"Loading data from {filename}, Seed: {seed_idx} ...")
    with h5py.File(filename, "r") as f:
        x = np.array(f["x-coordinate"], dtype=np.float32)
        t = np.array(f["t-coordinate"], dtype=np.float32)
        
        if "tensor" in f:
            u_raw = np.array(f["tensor"][seed_idx], dtype=np.float32)
        elif "data" in f:
            u_raw = np.array(f["data"][seed_idx], dtype=np.float32)
        else:
            raise ValueError("Key 'tensor' or 'data' not found")
            
        if u_raw.shape == (len(t), len(x)):
            u_matrix = u_raw.T
        elif u_raw.shape == (len(x), len(t)):
            u_matrix = u_raw
        else:
            min_t = min(len(t), u_raw.shape[0], u_raw.shape[1])
            t = t[:min_t]
            if u_raw.shape[0] == len(x):
                u_matrix = u_raw[:, :min_t]
            else:
                u_matrix = u_raw[:min_t, :].T

        print(f"  -> Final Matrix Shape: {u_matrix.shape}")
        
        X, T = np.meshgrid(x, t, indexing='ij') 
        input_data = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        output_data = u_matrix.flatten()[:, None]
        
        return input_data, output_data, x, t, u_matrix

# ================= 主流程 =================
def main():
    input_data, output_data, grid_x, grid_t, u_matrix = load_data(CONFIG["data_path"], args.seed)
    t_max = float(grid_t.max())
    
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, t_max)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # BC & IC
    bc = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)

    # IC (t=0)
    t0_idx = 0
    X_ic = np.hstack((grid_x[:, None], np.zeros_like(grid_x[:, None]))) 
    u_ic = u_matrix[:, t0_idx][:, None]
    ic_data = dde.icbc.PointSetBC(X_ic, u_ic)

    # 内部数据
    num_samples = int(len(input_data) * CONFIG["val_ratio"])
    idx = np.random.choice(len(input_data), num_samples, replace=False)
    bc_data = dde.icbc.PointSetBC(input_data[idx], output_data[idx])

    data = dde.data.TimePDE(
        geomtime, pde_advection, [bc, ic_data, bc_data],
        num_domain=2000, num_boundary=2000, num_initial=2000,
    )

    # ==========================================
    # 关键修改：更换激活函数为 sin
    # ==========================================
    # "sin" 激活函数能让网络天然具备模拟波动的能力
    # 同时稍微加深网络层数
    net = dde.nn.FNN([2] + [64] * 5 + [1], "sin", "He normal") 
    
    model = dde.Model(data, net)
    
    # Loss Weights: [PDE, BC, IC, Data]
    # 继续保持强数据约束
    loss_weights = [1.0, 1.0, 100.0, 100.0]
    
    model.compile("adam", lr=args.lr, loss_weights=loss_weights)

    print(f"\n>>> Start Training (Sine Activation) Seed {args.seed}...")
    ckpt_path = os.path.join(CONFIG["output_dir"], "model.ckpt")
    checker = dde.callbacks.ModelCheckpoint(ckpt_path, save_better_only=True, period=5000)
    
    start_time = time.time()
    # 这里的 train 只是演示，如果您已经训练好，可以注释掉这行直接 predict
    losshistory, _ = model.train(epochs=args.epochs, display_every=1000, callbacks=[checker])
    print(f">>> Training finished in {time.time() - start_time:.2f}s")
    
    dde.utils.save_loss_history(losshistory, os.path.join(CONFIG["output_dir"], "loss.dat"))

    # ==================================================
    # [修改] 3个时刻的对比图 (True vs Pred)
    # ==================================================
    print(">>> Generating 3-Snapshot Comparison Plots...")
    plt.figure(figsize=(18, 5)) # 宽一点，放3张子图
    
    max_t_idx = len(grid_t) - 1
    # 选取 25%, 50%, 75% 三个时间点
    indices = [int(max_t_idx * 0.25), int(max_t_idx * 0.5), int(max_t_idx * 0.75)]
    
    for i, t_idx in enumerate(indices):
        real_t = grid_t[t_idx]
        
        # 构造输入
        X_test = np.hstack((grid_x[:, None], np.ones_like(grid_x[:, None]) * real_t))
        
        # 预测
        u_pred = model.predict(X_test)
        u_true = u_matrix[:, t_idx]
        
        # 绘图
        plt.subplot(1, 3, i+1)
        plt.plot(grid_x, u_true, 'k-', linewidth=2, label='Ground Truth')
        plt.plot(grid_x, u_pred, 'r--', linewidth=2, label='PINN Prediction')
        
        plt.title(f"Time: {real_t:.2f}s")
        plt.xlabel("Position (x)")
        # 统一 Y 轴范围，方便观察波形移动
        plt.ylim([u_matrix.min() - 0.2, u_matrix.max() + 0.2])
        
        if i == 0:
            plt.ylabel("u(x,t)")
            plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"Advection PINN Results (Seed {args.seed})", fontsize=16)
    plt.tight_layout()
    save_file = os.path.join(CONFIG["output_dir"], "result_3_snapshots.png")
    plt.savefig(save_file)
    print(f"Result saved to {save_file}")

if __name__ == "__main__":
    main()