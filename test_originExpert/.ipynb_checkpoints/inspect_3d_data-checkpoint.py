"""
位置: test_codes/inspect_3d_data.py
功能: 打印 3D CFD HDF5 文件的内部结构 (Keys 和 Shape)
"""
import h5py
import os

# 配置文件路径 (确保这个路径和你 test_expert_unet_3d_cfd.py 里的一致)
DATA_PATH = "../data/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"

def inspect_hdf5():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 文件不存在: {DATA_PATH}")
        return

    print(f">>> Inspecting: {DATA_PATH}")
    
    try:
        with h5py.File(DATA_PATH, "r") as f:
            print(f"✅ 文件成功打开！")
            print(f"📂 根目录下的 Keys: {list(f.keys())}")
            
            # 尝试打印每个 Key 的形状
            for key in f.keys():
                # 排除一些非数据的 key（如果存在）
                if isinstance(f[key], h5py.Dataset):
                    print(f"   🔹 Dataset ['{key}']: Shape = {f[key].shape}, Type = {f[key].dtype}")
                elif isinstance(f[key], h5py.Group):
                    print(f"   📂 Group ['{key}']")
                    
    except Exception as e:
        print(f"❌ 读取发生错误: {e}")

if __name__ == "__main__":
    inspect_hdf5()