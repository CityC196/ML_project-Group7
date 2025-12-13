"""
位置: test_codes/test_expert_advection.py
功能: 测试 ExpertModels 文件夹下的 1D Advection (beta=1.0) PINN 专家模型
修复: 自动处理 Time/Data 维度不一致 (202 vs 201) 的问题
"""
import os
import sys
import h5py
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

# ===========================
# 1. 配置路径 (Configuration)
# ===========================
CONFIG = {
    "checkpoint_path": "../ExpertModels/1D_Advection_Sols_beta1.0_PINN.pt-15000.pt",
    "data_path": "../data/1D_Advection_Sols_beta1.0.hdf5", 
    "output_dir": "../OUTPUT/advection_expert_PINN",
    "beta": 1.0, 
    "seed": 0
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ===========================
# 2. PDE 定义 (Advection)
# ===========================
def pde_advection(x, y):
    beta = CONFIG["beta"]
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    return dy_t + beta * dy_x

# ===========================
# 3. 数据加载辅助函数 (修复版)
# ===========================
def load_eval_data(filename, seed_idx=0):
    print(f">>> Loading data from {filename} ...")
    if not os.path.exists(filename):
        print(f"Error: 数据文件不存在: {filename}")
        sys.exit(1)
        
    try:
        with h5py.File(filename, "r") as f:
            # 读取网格
            x = np.array(f["x-coordinate"], dtype=np.float32)
            t = np.array(f["t-coordinate"], dtype=np.float32)
            
            # 读取 Tensor 数据
            if "tensor" in f:
                u_raw = np.array(f["tensor"][seed_idx], dtype=np.float32)
            elif "data" in f:
                u_raw = np.array(f["data"][seed_idx], dtype=np.float32)
            else:
                raise ValueError("Cannot find 'tensor' or 'data' in HDF5")
            
            print(f"    Grid shapes -> x: {x.shape}, t: {t.shape}")
            print(f"    Raw data shape: {u_raw.shape}")

            # === 维度修复逻辑 ===
            # 目标: 将 u_true 统一为 [Space, Time] (例如 1024, 201)
            # 同时修正 t 数组的长度以匹配数据
            
            nt_data = -1
            
            # 情况 A: [Time, Space] (例如 201, 1024)
            # 判断依据: 第2个维度等于 x 的长度 (1024)
            if u_raw.shape[1] == len(x):
                print("    检测到格式 [Time, Space]，正在转置...")
                u_true = u_raw.T  # 转置后变 [Space, Time] (1024, 201)
                nt_data = u_true.shape[1]
                
            # 情况 B: [Space, Time] (例如 1024, 201)
            # 判断依据: 第1个维度等于 x 的长度
            elif u_raw.shape[0] == len(x):
                print("    检测到格式 [Space, Time]，无需转置。")
                u_true = u_raw
                nt_data = u_true.shape[1]
                
            else:
                raise ValueError(f"无法匹配空间维度! Data: {u_raw.shape}, x: {len(x)}")
            
            # === 时间维度对齐 ===
            if len(t) != nt_data:
                print(f"    警告: 时间网格长度 ({len(t)}) 与数据时间步 ({nt_data}) 不一致!")
                if len(t) > nt_data:
                    print(f"    -> 自动截断时间网格至 {nt_data} 个点。")
                    t = t[:nt_data]
                else:
                    raise ValueError("数据时间步多于时间网格，无法自动对齐。")
            
            return x, t, u_true

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

# ===========================
# 4. 主函数
# ===========================
def main():
    # 4.1 几何与时间域 (为了兼容性，时间域稍微设大一点也没关系)
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 2.0) 
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # 4.2 虚拟 BC
    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    bc = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
    
    data = dde.data.TimePDE(
        geomtime, pde_advection, [ic, bc],
        num_domain=100, num_boundary=100, num_initial=100
    )

    # 4.3 网络结构
    net = dde.nn.FNN([2] + [40] * 6 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    # 4.4 加载权重
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

    # 4.5 获取数据 (已修复维度)
    x_grid, t_grid, u_true = load_eval_data(CONFIG["data_path"], CONFIG["seed"])
    
    # 4.6 预测与绘图
    # 动态选择时间索引，确保不超过数据范围
    max_t_idx = len(t_grid) - 1
    # 取 25%, 50%, 75% 处的时间点
    snapshot_indices = [int(max_t_idx * 0.25), int(max_t_idx * 0.5), int(max_t_idx * 0.75)]
    
    plt.figure(figsize=(15, 5))
    total_mse = 0
    
    for i, t_idx in enumerate(snapshot_indices):
        real_t = t_grid[t_idx]
        
        # 构造输入坐标 (Space, 1)
        # x_grid: (1024,) -> (1024, 1)
        X_plot = x_grid[:, None]
        # T_plot: (1024, 1) 全是 real_t
        T_plot = np.ones_like(X_plot) * real_t
        X_input = np.hstack((X_plot, T_plot))
        
        # 预测 -> (1024, 1)
        u_pred = model.predict(X_input)
        
        # 提取真值
        # u_true 是 [Space, Time] (1024, 201)
        # 取第 t_idx 列 -> (1024,)
        u_real = u_true[:, t_idx] 
        
        # 计算 MSE
        mse = np.mean((u_pred.flatten() - u_real.flatten())**2)
        total_mse += mse
        
        # 绘图
        plt.subplot(1, 3, i+1)
        plt.plot(x_grid, u_real, 'k--', label='Ground Truth', linewidth=2)
        plt.plot(x_grid, u_pred, 'r-', label='Expert Pred', linewidth=1.5)
        plt.title(f"t = {real_t:.2f} (MSE={mse:.2e})")
        plt.xlabel("x")
        plt.ylabel("u")
        if i == 0: plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], "advection_expert_test.png")
    plt.savefig(save_path)
    print(f"\n>>> 测试完成！")
    print(f">>> 平均 MSE: {total_mse / 3:.4e}")
    print(f">>> 结果图已保存至: {save_path}")

if __name__ == "__main__":
    main()