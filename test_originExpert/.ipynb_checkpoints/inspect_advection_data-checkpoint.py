"""
位置: test/inspect_advection_data.py
功能: 深度诊断 Advection 数据集，排查“真实值全为常数”的问题
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# 配置文件路径
DATA_PATH = "../data/1D_Advection_Sols_beta1.0.hdf5"

def inspect_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 文件不存在: {DATA_PATH}")
        return

    print(f"📂 打开文件: {DATA_PATH}")
    with h5py.File(DATA_PATH, "r") as f:
        print(f"   🔑 Root Keys: {list(f.keys())}")
        
        # 1. 检查 Tensor 形状和内容
        if "tensor" in f:
            data_full = f["tensor"]
            print(f"   📊 Tensor Dataset Shape: {data_full.shape}")
            
            # 尝试读取几个不同的 Seed，看看是否 Seed 0 是坏的
            for seed in [0, 1, 10]:
                if seed >= data_full.shape[0]: continue
                
                sample = np.array(data_full[seed], dtype=np.float32)
                print(f"\n   🔍 Inspecting Seed {seed}:")
                print(f"      Shape: {sample.shape}")
                print(f"      Range: [{sample.min():.4f}, {sample.max():.4f}]")
                print(f"      Std Dev: {sample.std():.4f} (如果接近0，说明是常数)")
                
                # 如果是常数，直接跳过绘图，因为没意义
                if sample.std() < 1e-6:
                    print("      ⚠️ 警告: 该样本看起来是常数/空数据！")
                    continue
                
                # 尝试判断维度
                # Advection 通常是 (Time, Spatial) 或 (Spatial, Time)
                # 我们可以通过看哪个维度更像 "1024" (Spatial) 来判断
                if sample.shape[0] == 1024:
                    print("      -> 推测维度: (Spatial, Time)")
                    plot_data = sample # (S, T)
                elif sample.shape[1] == 1024:
                    print("      -> 推测维度: (Time, Spatial)")
                    plot_data = sample.T # 转置为 (S, T) 方便绘图
                else:
                    print("      -> 无法确定维度，保持原样绘图")
                    plot_data = sample

                # 2. 绘图验证
                plt.figure(figsize=(10, 4))
                
                # 热力图 (S-T 视图)
                plt.subplot(1, 2, 1)
                plt.imshow(plot_data, aspect='auto', cmap='jet', origin='lower')
                plt.title(f"Heatmap Seed {seed} (Transposed)")
                plt.xlabel("Time-axis")
                plt.ylabel("Spatial-axis")
                plt.colorbar()
                
                # 波形切片 (检查是否随时间移动)
                plt.subplot(1, 2, 2)
                # 画第 0, 20, 40... 列 (假设列是时间)
                dim_t = plot_data.shape[1]
                steps = np.linspace(0, dim_t-1, 5).astype(int)
                for t in steps:
                    plt.plot(plot_data[:, t], label=f'idx={t}')
                plt.title("Spatial Profiles at different indices")
                plt.legend(fontsize='x-small')
                
                plt.tight_layout()
                save_name = f"OUTPUT/debug_advection_seed{seed}.png"
                os.makedirs("OUTPUT", exist_ok=True)
                plt.savefig(save_name)
                print(f"      🖼️  Saved debug plot to {save_name}")
                
                # 只画一个正常的样本就退出，避免刷屏
                break
        else:
            print("❌ 没找到 'tensor' 键，无法分析。")

if __name__ == "__main__":
    inspect_data()