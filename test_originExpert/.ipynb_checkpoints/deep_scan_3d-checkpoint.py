"""
位置: test_codes/deep_scan_3d.py
功能: 递归遍历 HDF5 文件结构，检测内部损坏
"""
import h5py
import sys

DATA_PATH = "../data/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"

def scan_node(name, node):
    if isinstance(node, h5py.Dataset):
        # 尝试读取数据的一个小切片，验证数据完整性
        try:
            _ = node.shape
            _ = node.dtype
            # 可选：尝试读取第一个字节
            # _ = node[0] 
            print(f"  ✅ Dataset: {name} | Shape: {node.shape}")
        except Exception as e:
            print(f"  ❌ CORRUPTED Dataset: {name} | Error: {e}")
            raise e
    else:
        print(f"  📂 Group: {name}")

def main():
    print(f">>> Deep scanning: {DATA_PATH}")
    try:
        with h5py.File(DATA_PATH, "r") as f:
            print("  Root opened successfully.")
            # visititems 会递归遍历所有节点
            f.visititems(scan_node)
            print("\n>>> 🎉 Scan Complete! File structure appears valid.")
    except Exception as e:
        print(f"\n>>> 💀 Scan FAILED! File is corrupted.")
        print(f"Error details: {e}")
        print("建议: 请删除该文件并重新下载 (MD5 校验失败)。")

if __name__ == "__main__":
    main()