# evaluate.py (Final Version - Modified from your original script)

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# ==============================================================================
#  1. 从你的代码库导入核心模型定义
#     确保在项目根目录下运行，以便找到 COMPARE 文件夹
# ==============================================================================
try:
    from COMPARE.CDL_ASPP import DeepLabV3WithComplexNet, normalize_f_data
except ModuleNotFoundError:
    print("Error: Could not import model definition 'DeepLabV3WithComplexNet'.")
    print("Please make sure you are running this script from the project's root directory (e.g., 'CDL-EMIS/').")
    exit()

# ==============================================================================
#  2. 数据集定义 (从你的代码中复制)
# ==============================================================================
class MatrixDataset(Dataset):
    def __init__(self, re_fnames, im_fnames, f_fnames, label_re_fnames, label_im_fnames,
                 re_mean, re_std, im_mean, im_std):
        self.re_paths, self.im_paths, self.f_paths = re_fnames, im_fnames, f_fnames
        self.label_re_paths, self.label_im_paths = label_re_fnames, label_im_fnames
        self.target_size = (512, 512)
        self.re_mean, self.re_std = re_mean, re_std
        self.im_mean, self.im_std = im_mean, im_std

    def __len__(self):
        return len(self.re_paths)

    def __getitem__(self, idx):
        re_data = pd.read_excel(self.re_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        im_data = pd.read_excel(self.im_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        f_data_orig = pd.read_excel(self.f_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        f_data_normalized = normalize_f_data(f_data_orig)
        re_data_resized = cv2.resize(re_data, self.target_size, interpolation=cv2.INTER_LINEAR)
        im_data_resized = cv2.resize(im_data, self.target_size, interpolation=cv2.INTER_LINEAR)
        f_data_resized = cv2.resize(f_data_normalized, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # 保存一份未归一化的输入，用于绘图
        input_unnormalized_re = re_data_resized.copy()
        input_unnormalized_im = im_data_resized.copy()
        
        # 使用传入的均值和标准差进行归一化
        if self.re_std > 1e-7: re_data_resized = (re_data_resized - self.re_mean) / self.re_std
        if self.im_std > 1e-7: im_data_resized = (im_data_resized - self.im_mean) / self.im_std
        
        input_data_np = np.stack([re_data_resized, im_data_resized, f_data_resized], axis=0)
        input_tensor = torch.from_numpy(input_data_np).float()
        
        label_re_orig = pd.read_csv(self.label_re_paths[idx], header=None).values.astype(np.float32)
        label_im_orig = pd.read_csv(self.label_im_paths[idx], header=None).values.astype(np.float32)
        label_re_resized = cv2.resize(label_re_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_im_resized = cv2.resize(label_im_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_data_np = np.stack([label_re_resized, label_im_resized], axis=0)
        label_tensor = torch.from_numpy(label_data_np).float()
        
        return {
            'image': input_tensor, 
            'label': label_tensor,
            'input_unnormalized': np.stack([input_unnormalized_re, input_unnormalized_im], axis=0)
        }

# ==============================================================================
#  3. 绘图函数 (简化为你需要的功能)
# ==============================================================================
# 这是新的 plot_results 函数，请用它替换掉 evaluate.py 里的旧版本

def plot_results(input_data, label_data, prediction_data, output_path, sample_name):
    """
    可视化输入、真实标签和预测结果，并保存为图片。
    (新版本：采用与你原脚本一致的对称、共享色标方案)
    """
    input_re, input_im = input_data[0], input_data[1]
    label_re, label_im = label_data[0], label_data[1]
    pred_re, pred_im = prediction_data[0], prediction_data[1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Comparison for Sample: {sample_name}', fontsize=16)

    # --- 核心修改：计算对称且共享的颜色范围 ---
    
    # 1. 为 "Input" 计算独立的对称范围
    v_in_re = np.max(np.abs(input_re))
    v_in_im = np.max(np.abs(input_im))

    # 2. 为 "Ground Truth" 和 "Prediction" 计算共享的对称范围
    v_out_re = np.max([np.abs(label_re), np.abs(pred_re)])
    v_out_im = np.max([np.abs(label_im), np.abs(pred_im)])

    # --- 第一行: 实部 (Real Part) ---
    axes[0, 0].set_title('Input (Real Part)')
    im1 = axes[0, 0].imshow(input_re, cmap='coolwarm', vmin=-v_in_re, vmax=v_in_re)
    fig.colorbar(im1, ax=axes[0, 0])

    axes[0, 1].set_title('Ground Truth (Real Part)')
    im2 = axes[0, 1].imshow(label_re, cmap='coolwarm', vmin=-v_out_re, vmax=v_out_re)
    fig.colorbar(im2, ax=axes[0, 1])

    axes[0, 2].set_title('Prediction (Real Part)')
    im3 = axes[0, 2].imshow(pred_re, cmap='coolwarm', vmin=-v_out_re, vmax=v_out_re)
    fig.colorbar(im3, ax=axes[0, 2])

    # --- 第二行: 虚部 (Imaginary Part) ---
    axes[1, 0].set_title('Input (Imaginary Part)')
    im4 = axes[1, 0].imshow(input_im, cmap='coolwarm', vmin=-v_in_im, vmax=v_in_im)
    fig.colorbar(im4, ax=axes[1, 0])

    axes[1, 1].set_title('Ground Truth (Imaginary Part)')
    im5 = axes[1, 1].imshow(label_im, cmap='coolwarm', vmin=-v_out_im, vmax=v_out_im)
    fig.colorbar(im5, ax=axes[1, 1])

    axes[1, 2].set_title('Prediction (Imaginary Part)')
    im6 = axes[1, 2].imshow(pred_im, cmap='coolwarm', vmin=-v_out_im, vmax=v_out_im)
    fig.colorbar(im6, ax=axes[1, 2])
    
    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    print(f"✅ 结果对比图已保存到: {output_path} (已更新配色方案)")

# ==============================================================================
#  4. 主执行逻辑
# ==============================================================================
def main(args):
    # --- 关键：使用你提供的准确统计数据 ---
    TRAIN_STATS = {
        're_mean': 270.9892,
        're_std': 2192.1480,
        'im_mean': -156.8289,
        'im_std': 2003.2118
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    model = DeepLabV3WithComplexNet(output_channels=2).to(device)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'. Please check the path.")
        return
    print(f"Loading pre-trained weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device); model.load_state_dict(checkpoint['model_state_dict']); model.eval(); print("Model loaded successfully.")
    
    # --- 加载数据 ---
    try:
        subfolders = {'re': os.path.join(args.data_dir, 'E', 'Re'), 'im': os.path.join(args.data_dir, 'E', 'Im'),'f': os.path.join(args.data_dir, 'E', 'F'), 'label_re': os.path.join(args.data_dir, 'label', 'label_Re'), 'label_im': os.path.join(args.data_dir, 'label', 'label_Im')}
        filenames = {key: natsorted(os.listdir(path)) for key, path in subfolders.items()}
        num_samples = len(filenames['re'])
        if not (0 <= args.sample_index < num_samples):
            print(f"Error: Invalid sample index {args.sample_index}. Available range is 0 to {num_samples - 1}.")
            return
        
        dataset = MatrixDataset(
            re_fnames=[os.path.join(subfolders['re'], filenames['re'][args.sample_index])],
            im_fnames=[os.path.join(subfolders['im'], filenames['im'][args.sample_index])],
            f_fnames=[os.path.join(subfolders['f'], filenames['f'][args.sample_index])],
            label_re_fnames=[os.path.join(subfolders['label_re'], filenames['label_re'][args.sample_index])],
            label_im_fnames=[os.path.join(subfolders['label_im'], filenames['label_im'][args.sample_index])],
            **TRAIN_STATS # 使用正确的统计数据！
        )
        sample = dataset[0]; sample_name = os.path.splitext(filenames['re'][args.sample_index])[0]
    except Exception as e:
        print(f"Error: An error occurred while loading data. Details: {e}")
        return

    # --- 预测和绘图 ---
    input_tensor, label_tensor, input_unnormalized = sample['image'], sample['label'], sample['input_unnormalized']
    print(f"Predicting for sample '{sample_name}' (index: {args.sample_index})...")
    with torch.no_grad():
        prediction = model(input_tensor.unsqueeze(0).to(device))
    prediction_np = prediction.squeeze(0).cpu().numpy()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"prediction_{sample_name}.png")
    
    plot_results(input_unnormalized, label_tensor.numpy(), prediction_np, output_path, sample_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict and visualize results using a pre-trained CDL-Net model.")
    parser.add_argument('--model_path', type=str, required=True, help='Required: Path to the pre-trained .pth model file.')
    parser.add_argument('--data_dir', type=str, default='./sample_data', help="Path to the directory containing sample data. (default: './sample_data')")
    parser.add_argument('--sample_index', type=int, default=0, help="Index of the sample to predict from the data folder (0-based). (default: 0)")
    parser.add_argument('--output_dir', type=str, default='./results', help="Directory to save the resulting plot image. (default: './results')")
    args = parser.parse_args()
    main(args)