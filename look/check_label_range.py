# 文件名: check_label_range.py

import torch
from tqdm import tqdm

# ⚠️ 【请确保这里的导入路径正确】
# 我们需要从你之前的代码里，把数据加载相关的函数和类复制过来
# ==============================================================================
import os
import numpy as np
import pandas as pd
import cv2
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from natsort import natsorted

def normalize_f_data(data, min_val=2.0, max_val=8.0):
    data = data.astype(np.float32)
    if (max_val - min_val) == 0: return data - min_val
    return (data - min_val) / (max_val - min_val)

class MatrixDataset(Dataset):
    # (此部分与你之前的代码完全一致，直接复制即可)
    def __init__(self, re_paths, im_paths, f_paths, label_re_paths, label_im_paths,
                 re_mean=None, re_std=None, im_mean=None, im_std=None, calculate_stats_mode=False):
        self.re_paths, self.im_paths, self.f_paths = re_paths, im_paths, f_paths
        self.label_re_paths = label_re_paths if label_re_paths else []
        self.label_im_paths = label_im_paths if label_im_paths else []
        self.target_size = (512, 512)
        self.re_mean, self.re_std, self.im_mean, self.im_std = re_mean, re_std, im_mean, im_std
        self.calculate_stats_mode = calculate_stats_mode

    def __len__(self):
        return len(self.re_paths)

    def __getitem__(self, idx):
        re_data = pd.read_excel(self.re_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        im_data = pd.read_excel(self.im_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        f_data_orig = pd.read_excel(self.f_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        
        re_data_resized = cv2.resize(re_data, self.target_size, interpolation=cv2.INTER_LINEAR)
        im_data_resized = cv2.resize(im_data, self.target_size, interpolation=cv2.INTER_LINEAR)
        f_data_resized = cv2.resize(normalize_f_data(f_data_orig), self.target_size, interpolation=cv2.INTER_LINEAR)

        if not self.calculate_stats_mode:
            if self.re_mean is not None and self.re_std is not None and self.re_std > 1e-7:
                re_data_resized = (re_data_resized - self.re_mean) / self.re_std
            if self.im_mean is not None and self.im_std is not None and self.im_std > 1e-7:
                im_data_resized = (im_data_resized - self.im_mean) / self.im_std
        
        input_data_np = np.stack([re_data_resized, im_data_resized, f_data_resized], axis=0)
        input_tensor = torch.from_numpy(input_data_np).float()

        if self.calculate_stats_mode:
            return input_tensor

        label_re_orig = pd.read_csv(self.label_re_paths[idx], header=None).values.astype(np.float32)
        label_im_orig = pd.read_csv(self.label_im_paths[idx], header=None).values.astype(np.float32)
        label_re_resized = cv2.resize(label_re_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_im_resized = cv2.resize(label_im_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_data_np = np.stack([label_re_resized, label_im_resized], axis=0)
        label_tensor = torch.from_numpy(label_data_np).float()
        return input_tensor, label_tensor

def get_test_dataloader(data_folder, test_split_ratio=0.2, random_state=42, batch_size=1):
    # (此函数与你提供的完全一致，直接复制即可)
    print("🚀 开始构建测试数据加载器...")
    E_folder_path_check = os.path.join(data_folder, 'E')
    if os.path.exists(E_folder_path_check):
      f_folder_path, re_folder_path, im_folder_path = [os.path.join(E_folder_path_check, d) for d in ['F', 'Re', 'Im']]
    else:
      f_folder_path, re_folder_path, im_folder_path = [os.path.join(data_folder, d) for d in ['F', 'Re', 'Im']]
    label_folder_path_check = os.path.join(data_folder, 'label')
    if os.path.exists(label_folder_path_check):
      label_re_folder_path, label_im_folder_path = [os.path.join(label_folder_path_check, d) for d in ['label_Re', 'label_Im']]
    else:
      label_re_folder_path, label_im_folder_path = [os.path.join(data_folder, d) for d in ['label_Re', 'label_Im']]
    re_all_filenames, im_all_filenames, f_all_filenames = [natsorted(os.listdir(p)) for p in [re_folder_path, im_folder_path, f_folder_path]]
    label_re_all_filenames, label_im_all_filenames = [natsorted(os.listdir(p)) for p in [label_re_folder_path, label_im_folder_path]]
    min_count = min(len(f) for f in [re_all_filenames, im_all_filenames, f_all_filenames, label_re_all_filenames, label_im_all_filenames])
    re_f, im_f, f_f, lr_f, li_f = [lst[:min_count] for lst in [re_all_filenames, im_all_filenames, f_all_filenames, label_re_all_filenames, label_im_all_filenames]]
    train_files, test_files = train_test_split(list(zip(re_f, im_f, f_f, lr_f, li_f)), test_size=test_split_ratio, random_state=random_state)
    re_train_files, im_train_files, f_train_files, _, _ = zip(*train_files)
    re_test_files, im_test_files, f_test_files, label_re_test_files, label_im_test_files = zip(*test_files)
    stat_dataset = MatrixDataset(re_paths=[os.path.join(re_folder_path, f) for f in re_train_files], im_paths=[os.path.join(im_folder_path, f) for f in im_train_files], f_paths=[os.path.join(f_folder_path, f) for f in f_train_files], label_re_paths=[], label_im_paths=[], calculate_stats_mode=True)
    stat_loader = DataLoader(stat_dataset, batch_size=8, shuffle=False, num_workers=0)
    re_sum, im_sum, re_sum_sq, im_sum_sq, total_pixels = 0.0, 0.0, 0.0, 0.0, 0
    for inputs in stat_loader:
        re_ch, im_ch = inputs[:, 0, :, :], inputs[:, 1, :, :]
        re_sum += torch.sum(re_ch).item(); im_sum += torch.sum(im_ch).item()
        re_sum_sq += torch.sum(torch.square(re_ch)).item(); im_sum_sq += torch.sum(torch.square(im_ch)).item()
        total_pixels += re_ch.nelement()
    re_m_train, im_m_train = re_sum / total_pixels, im_sum / total_pixels
    re_s_train, im_s_train = math.sqrt(max(0, (re_sum_sq/total_pixels) - (re_m_train**2))), math.sqrt(max(0, (im_sum_sq/total_pixels) - (im_m_train**2)))
    test_dataset = MatrixDataset(re_paths=[os.path.join(re_folder_path, f) for f in re_test_files], im_paths=[os.path.join(im_folder_path, f) for f in im_test_files], f_paths=[os.path.join(f_folder_path, f) for f in f_test_files], label_re_paths=[os.path.join(label_re_folder_path, f) for f in label_re_test_files], label_im_paths=[os.path.join(label_im_folder_path, f) for f in label_im_test_files], re_mean=re_m_train, re_std=re_s_train, im_mean=im_m_train, im_std=im_s_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"✅ 测试数据加载器创建成功，共 {len(test_dataset)} 个样本。")
    return test_loader
# ==============================================================================

def check_label_range(data_loader):
    """遍历数据加载器，检查所有标签张量的全局最大值和最小值。"""
    
    print("\n🔍 开始检查标签(label)数据范围...")
    
    # 初始化全局最大值和最小值
    global_min = float('inf')
    global_max = float('-inf')

    # 使用tqdm来显示进度条
    for i, (input_tensor, label_tensor) in enumerate(tqdm(data_loader, desc="正在检查...")):
        
        # 计算当前批次的最大值和最小值
        current_min = torch.min(label_tensor)
        current_max = torch.max(label_tensor)

        # 更新全局最大值和最小值
        if current_min < global_min:
            global_min = current_min.item()
        
        if current_max > global_max:
            global_max = current_max.item()

    print("\n" + "="*50)
    print("✅ 数据集范围检查完毕！")
    print(f"  - 全局最小值 (Global Min): {global_min:.4f}")
    print(f"  - 全局最大值 (Global Max): {global_max:.4f}")
    
    # PSNR的data_range应该是最大值和最小值的差
    data_range = global_max - global_min
    print(f"  - 建议的PSNR data_range: {data_range:.4f}")
    print("="*50)
    
    return data_range

if __name__ == "__main__":
    # ⚠️【请在此处修改】你的数据根目录
    DATA_ROOT_DIR = "E:/EMTdata"

    # 1. 获取测试数据加载器
    # 注意：这里的batch_size可以设大一点，能跑得更快
    test_loader = get_test_dataloader(
        data_folder=DATA_ROOT_DIR,
        batch_size=16, 
        random_state=42 # 保持和评估时一致的划分
    )

    # 2. 运行检查函数
    if test_loader:
        check_label_range(test_loader)