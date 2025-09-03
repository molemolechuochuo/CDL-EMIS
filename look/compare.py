import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
# ⚠️ 1. 【请在此处修改】导入你定义的模型类
# ==============================================================================
from COMPARE.unet import UNet
from COMPARE.FCN import FCN_ResNet50_Regression
from COMPARE.segnet import SegNet
from COMPARE.PSPNET import PSPNet
from COMPARE.deeplab import DeepLabV3ForRegression
from COMPARE.CDLmix import ResnetWithComplexNet
from COMPARE.CDL_ASPP import DeepLabV3WithComplexNet
# ==============================================================================


import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from natsort import natsorted
from tqdm import tqdm
import math

# 确保您已安装必要的库:
# pip install pandas openpyxl scikit-learn natsort opencv-python

def normalize_f_data(data, min_val=2.0, max_val=8.0):
    data = data.astype(np.float32)
    if (max_val - min_val) == 0: return data - min_val
    return (data - min_val) / (max_val - min_val)

class MatrixDataset(Dataset):
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
    print("🚀 开始构建测试数据加载器...")
    print("🔍 正在解析文件路径...")
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
    print("路径解析完毕。")

    print("🧾 正在列出并对齐所有数据文件...")
    re_all_filenames = natsorted([f for f in os.listdir(re_folder_path) if f.lower().endswith('.xlsx')])
    im_all_filenames = natsorted([f for f in os.listdir(im_folder_path) if f.lower().endswith('.xlsx')])
    f_all_filenames = natsorted([f for f in os.listdir(f_folder_path) if f.lower().endswith('.xlsx')])
    label_re_all_filenames = natsorted([f for f in os.listdir(label_re_folder_path) if f.lower().endswith('.csv')])
    label_im_all_filenames = natsorted([f for f in os.listdir(label_im_folder_path) if f.lower().endswith('.csv')])

    min_count = min(len(f) for f in [re_all_filenames, im_all_filenames, f_all_filenames, label_re_all_filenames, label_im_all_filenames])
    re_f, im_f, f_f, lr_f, li_f = [lst[:min_count] for lst in [re_all_filenames, im_all_filenames, f_all_filenames, label_re_all_filenames, label_im_all_filenames]]
    print(f"文件对齐完毕，共找到 {min_count} 组匹配数据。")
    
    print(f"🔪 正在以 random_state={random_state} 划分数据集...")
    train_files, test_files = train_test_split(list(zip(re_f, im_f, f_f, lr_f, li_f)), test_size=test_split_ratio, random_state=random_state)
    re_train_files, im_train_files, f_train_files, _, _ = zip(*train_files)
    re_test_files, im_test_files, f_test_files, label_re_test_files, label_im_test_files = zip(*test_files)
    print(f"划分完毕: 训练集 {len(train_files)} 个, 测试集 {len(test_files)} 个。")

    print("📊 正在计算【训练集】的均值和标准差用于归一化...")
    stat_dataset = MatrixDataset(
        re_paths=[os.path.join(re_folder_path, f) for f in re_train_files],
        im_paths=[os.path.join(im_folder_path, f) for f in im_train_files],
        f_paths=[os.path.join(f_folder_path, f) for f in f_train_files],
        label_re_paths=[], label_im_paths=[], calculate_stats_mode=True
    )
    stat_loader = DataLoader(stat_dataset, batch_size=8, shuffle=False, num_workers=0)

    re_sum, im_sum, re_sum_sq, im_sum_sq, total_pixels = 0.0, 0.0, 0.0, 0.0, 0
    for inputs in tqdm(stat_loader, desc="计算统计量"):
        re_ch, im_ch = inputs[:, 0, :, :], inputs[:, 1, :, :]
        re_sum += torch.sum(re_ch).item(); im_sum += torch.sum(im_ch).item()
        re_sum_sq += torch.sum(torch.square(re_ch)).item(); im_sum_sq += torch.sum(torch.square(im_ch)).item()
        total_pixels += re_ch.nelement()
    
    re_m_train = re_sum / total_pixels; im_m_train = im_sum / total_pixels
    re_s_train = math.sqrt(max(0, (re_sum_sq/total_pixels) - (re_m_train**2)))
    im_s_train = math.sqrt(max(0, (im_sum_sq/total_pixels) - (im_m_train**2)))
    print(f"统计结果: Re(均值={re_m_train:.4f}, Std={re_s_train:.4f}), Im(均值={im_m_train:.4f}, Std={im_s_train:.4f})")

    print("📦 正在创建最终的测试集加载器...")
    test_dataset = MatrixDataset(
        re_paths=[os.path.join(re_folder_path, f) for f in re_test_files],
        im_paths=[os.path.join(im_folder_path, f) for f in im_test_files],
        f_paths=[os.path.join(f_folder_path, f) for f in f_test_files],
        label_re_paths=[os.path.join(label_re_folder_path, f) for f in label_re_test_files],
        label_im_paths=[os.path.join(label_im_folder_path, f) for f in label_im_test_files],
        re_mean=re_m_train, re_std=re_s_train, im_mean=im_m_train, im_std=im_s_train
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"✅ 测试数据加载器创建成功，共 {len(test_dataset)} 个样本。")
    return test_loader

# --- ✨【全新重写的、更稳健的可视化函数】✨ ---
def visualize_grid_comparison(base_dir, model_map, test_loader, device, num_samples=4):
    print("\n🚀 Running Qualitative Analysis for Grid Visualization...")
    
    dataset = test_loader.dataset
    indices_to_visualize = np.arange(min(num_samples, len(dataset)))
    print(f"Selected image indices for visualization: {indices_to_visualize}")

    # --- 1. 模型加载和预测 ---
    predictions = {int(idx): {} for idx in indices_to_visualize}
    # ✨ 新增一个列表，只记录成功加载的模型名字
    successful_model_names = [] 
    
    for model_name_key, model_class in model_map.items():
        folder_name = model_name_key + "_Output"
        # 特殊处理你自己的模型名，确保能找到文件夹
        if model_name_key == "CDL (Ours)":
            folder_name = "ComplexNetDeepLab+aspp_Reg_Output"
        elif model_name_key == "U-Net":
            folder_name = "UNet_XLSX_Regression_Normalized_Output"
        elif model_name_key == "SegNet_XLSX":
            folder_name = "SegNet_XLSX_Output"
            
        folder_path = os.path.join(base_dir, folder_name)
        
        try:
            if callable(model_class) and not isinstance(model_class, type):
                model = model_class()
            else:
                model = model_class()

            pth_path_list = glob.glob(os.path.join(folder_path, '**', '*_best_model.pth'), recursive=True)
            if not pth_path_list:
                print(f"⚠️ Skipping {model_name_key}: No '*_best_model.pth' found in '{folder_path}'")
                continue # 找不到文件就跳过

            pth_path = pth_path_list[0]
            checkpoint = torch.load(pth_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            print(f"✅ Loaded model: {model_name_key}")
            
            # ✨ 模型加载成功后，才把它加入成功列表
            successful_model_names.append(model_name_key) 

            with torch.no_grad():
                for idx in indices_to_visualize:
                    input_tensor, _ = dataset[int(idx)]
                    output = model(input_tensor.unsqueeze(0).to(device))
                    predictions[int(idx)][model_name_key] = output.squeeze().cpu().numpy()
        except Exception as e:
            print(f"❌ Failed to process model {model_name_key}: {e}")

    # --- 2. 核心绘图逻辑 ---
    if not successful_model_names:
        print("\n❌ No models were successfully loaded. Cannot generate plot.")
        return

    print("\nGenerating grid comparison plot for successful models:", successful_model_names)
    
    num_rows = len(indices_to_visualize)
    num_cols = 1 + len(successful_model_names)  # ✨ 列数基于成功的模型数量

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4.5), squeeze=False)
    
    for row, sample_idx in enumerate(indices_to_visualize):
        _, ground_truth_tensor = dataset[int(sample_idx)]
        
        # ✨ 只处理成功加载的模型的预测结果
        gt_mag = np.sqrt(ground_truth_tensor[0, :, :]**2 + ground_truth_tensor[1, :, :]**2)
        pred_mags = {name: np.sqrt(predictions[sample_idx][name][0,:,:]**2 + predictions[sample_idx][name][1,:,:]**2) for name in successful_model_names}
        
        all_mags_in_row = [gt_mag] + list(pred_mags.values())
        vmin = min(m.min() for m in all_mags_in_row) if all_mags_in_row else 0
        vmax = max(m.max() for m in all_mags_in_row) if all_mags_in_row else 1
        
        # 第0列：Ground Truth
        ax = axes[row, 0]
        im = ax.imshow(gt_mag, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Ground Truth (Sample #{sample_idx})")
        ax.axis('off')

        # ✨ 只循环成功的模型
        for col, model_name in enumerate(successful_model_names):
            ax = axes[row, col + 1]
            ax.imshow(pred_mags[model_name], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(model_name)
            ax.axis('off')
            
        fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.8, pad=0.02)

    plt.tight_layout()
    plt.savefig("qualitative_comparison_grid.png", dpi=300)
    print(f"\n✅ Saved final grid comparison to 'qualitative_comparison_grid.png'")


if __name__ == "__main__":
    # --- 全局配置 ---
    DATA_ROOT_DIR = "E:/EMTdata"
    BASE_DIR = "E:/EMMESData/Remote/"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 选择要对比的模型 ---
    # ✨【重要】这里的键名要和你代码里的 `folder_name` 逻辑对应起来
    MODEL_CLASS_MAP = {
        'SegNet_XLSX': lambda: SegNet(input_channels=3, output_channels=2),
        'U-Net': lambda: UNet(n_channels_in=3, n_channels_out=2),
        'CDL (Ours)': DeepLabV3WithComplexNet,
    }
    
    test_loader = get_test_dataloader(data_folder=DATA_ROOT_DIR, batch_size=1, random_state=42)
    
    if test_loader:
        # 调用新的网格绘图函数
        visualize_grid_comparison(
            base_dir=BASE_DIR,
            model_map=MODEL_CLASS_MAP,
            test_loader=test_loader,
            device=DEVICE,
            num_samples=4 
        )