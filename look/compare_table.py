# 文件名: calculate_final_metrics.py

import os
import glob
import torch
import pandas as pd
import numpy as np
import cv2
import math
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from natsort import natsorted
from tqdm import tqdm

# 导入我之前给你的指标计算器
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ⚠️ 1. 【从你的代码中复制】导入你定义的所有模型类
# ==============================================================================
from COMPARE.unet import UNet
from COMPARE.FCN import FCN_ResNet50_Regression
from COMPARE.segnet import SegNet
from COMPARE.PSPNET import PSPNet
from COMPARE.deeplab import DeepLabV3ForRegression
from COMPARE.CDLmix import ResnetWithComplexNet
from COMPARE.CDL_ASPP import DeepLabV3WithComplexNet
# ==============================================================================

# ⚠️ 2. 【从你的代码中复制】你的数据加载函数 (无需任何修改)
# ==============================================================================
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
    # (此函数与你提供的完全一致，这里省略以节省空间，实际使用时请完整复制)
    # ... 完整复制你代码中的 get_test_dataloader 函数 ...
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
    re_all_filenames, im_all_filenames, f_all_filenames = [natsorted(os.listdir(p)) for p in [re_folder_path, im_folder_path, f_folder_path]]
    label_re_all_filenames, label_im_all_filenames = [natsorted(os.listdir(p)) for p in [label_re_folder_path, label_im_folder_path]]
    min_count = min(len(f) for f in [re_all_filenames, im_all_filenames, f_all_filenames, label_re_all_filenames, label_im_all_filenames])
    re_f, im_f, f_f, lr_f, li_f = [lst[:min_count] for lst in [re_all_filenames, im_all_filenames, f_all_filenames, label_re_all_filenames, label_im_all_filenames]]
    print(f"文件对齐完毕，共找到 {min_count} 组匹配数据。")
    print(f"🔪 正在以 random_state={random_state} 划分数据集...")
    train_files, test_files = train_test_split(list(zip(re_f, im_f, f_f, lr_f, li_f)), test_size=test_split_ratio, random_state=random_state)
    re_train_files, im_train_files, f_train_files, _, _ = zip(*train_files)
    re_test_files, im_test_files, f_test_files, label_re_test_files, label_im_test_files = zip(*test_files)
    print(f"划分完毕: 训练集 {len(train_files)} 个, 测试集 {len(test_files)} 个。")
    print("📊 正在计算【训练集】的均值和标准差用于归一化...")
    stat_dataset = MatrixDataset(re_paths=[os.path.join(re_folder_path, f) for f in re_train_files], im_paths=[os.path.join(im_folder_path, f) for f in im_train_files], f_paths=[os.path.join(f_folder_path, f) for f in f_train_files], label_re_paths=[], label_im_paths=[], calculate_stats_mode=True)
    stat_loader = DataLoader(stat_dataset, batch_size=8, shuffle=False, num_workers=0)
    re_sum, im_sum, re_sum_sq, im_sum_sq, total_pixels = 0.0, 0.0, 0.0, 0.0, 0
    for inputs in tqdm(stat_loader, desc="计算统计量"):
        re_ch, im_ch = inputs[:, 0, :, :], inputs[:, 1, :, :]
        re_sum += torch.sum(re_ch).item(); im_sum += torch.sum(im_ch).item()
        re_sum_sq += torch.sum(torch.square(re_ch)).item(); im_sum_sq += torch.sum(torch.square(im_ch)).item()
        total_pixels += re_ch.nelement()
    re_m_train, im_m_train = re_sum / total_pixels, im_sum / total_pixels
    re_s_train, im_s_train = math.sqrt(max(0, (re_sum_sq/total_pixels) - (re_m_train**2))), math.sqrt(max(0, (im_sum_sq/total_pixels) - (im_m_train**2)))
    print(f"统计结果: Re(均值={re_m_train:.4f}, Std={re_s_train:.4f}), Im(均值={im_m_train:.4f}, Std={im_s_train:.4f})")
    print("📦 正在创建最终的测试集加载器...")
    test_dataset = MatrixDataset(re_paths=[os.path.join(re_folder_path, f) for f in re_test_files], im_paths=[os.path.join(im_folder_path, f) for f in im_test_files], f_paths=[os.path.join(f_folder_path, f) for f in f_test_files], label_re_paths=[os.path.join(label_re_folder_path, f) for f in label_re_test_files], label_im_paths=[os.path.join(label_im_folder_path, f) for f in label_im_test_files], re_mean=re_m_train, re_std=re_s_train, im_mean=im_m_train, im_std=im_s_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"✅ 测试数据加载器创建成功，共 {len(test_dataset)} 个样本。")
    return test_loader
# ==============================================================================

# ⚠️ 3. 【这是我们的新核心函数】用于评估所有模型
# ==============================================================================
def calculate_metrics_for_all_models(base_dir, model_map, test_loader, device, data_range=59.0220):
    """
    加载每个模型的最佳权重，在整个测试集上计算PSNR和SSIM，并返回结果。
    """
    print("\n" + "="*80)
    print("🚀 Running Final Metric Calculation for All Models...")
    print("="*80)
    
    all_results = []

    # 遍历你在主函数里定义的每一个模型
    for model_name_key, model_class in model_map.items():
        folder_name = model_name_key + "_Output"
        folder_path = os.path.join(base_dir, folder_name)
        
        try:
            # --- 模型加载逻辑 (完全复用你的代码) ---
            # 1. 实例化模型，这里要特别处理lambda函数
            # callable()可以同时判断普通函数和lambda函数
            if callable(model_class) and not isinstance(model_class, type):
                model = model_class() # 调用lambda: UNet(...)
            else:
                model = model_class() # 直接实例化 MyModel()
                
            # 2. 查找最佳模型文件
            pth_path_list = glob.glob(os.path.join(folder_path, '*_best_model.pth'))
            if not pth_path_list:
                print(f"⚠️ Skipping {model_name_key}: No '*_best_model.pth' found in {folder_path}")
                continue
            
            # 3. 加载权重
            pth_path = pth_path_list[0]
            checkpoint = torch.load(pth_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            print(f"\n✅ Successfully loaded model: {model_name_key}")

            # --- 指标计算逻辑 (新添加的核心部分) ---
            # 1. 为当前模型初始化新的指标计算器
            psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(device)
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

            # 2. 在整个测试集上进行评估
            with torch.no_grad():
                for inputs, ground_truth in tqdm(test_loader, desc=f"Evaluating {model_name_key}"):
                    inputs = inputs.to(device)
                    ground_truth = ground_truth.to(device)
                    
                    predictions = model(inputs)
                    
                    # 累积计算指标
                    psnr_metric.update(predictions, ground_truth)
                    ssim_metric.update(predictions, ground_truth)

            # 3. 计算最终平均值
            final_psnr = psnr_metric.compute().item() # .item()获取纯数值
            final_ssim = ssim_metric.compute().item()

            print(f"  - Final PSNR: {final_psnr:.4f}")
            print(f"  - Final SSIM: {final_ssim:.4f}")

            # 4. 保存结果
            all_results.append({
                'Model': model_name_key,
                'PSNR': final_psnr,
                'SSIM': final_ssim
            })

        except Exception as e:
            print(f"❌ Failed to process model {model_name_key}: {e}")
            # raise e # 如果需要调试，可以取消这行注释来查看详细错误

    return all_results
# ==============================================================================


if __name__ == "__main__":
    # --- 全局配置 (从你的代码中复制) ---
    DATA_ROOT_DIR = "E:/EMTdata"
    BASE_DIR = "E:/EMMESData/Remote/"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 模型映射 (从你的代码中复制) ---
    MODEL_CLASS_MAP = {
        'ComplexNetDeepLab+aspp_Reg': DeepLabV3WithComplexNet,
        'ComplexNetDeepLab_Reg': ResnetWithComplexNet,
        'UNet_XLSX_Regression_Normalized': lambda: UNet(n_channels_in=3, n_channels_out=2),
        'DeepLabV3_Reg_UnifiedData': DeepLabV3ForRegression,
        'FCN_XLSX_Regression': FCN_ResNet50_Regression,
        'SegNet_XLSX': lambda: SegNet(input_channels=3, output_channels=2),
        'PSPNet_Regression_Unified': PSPNet
    }
    
    # 过滤掉未提供模型类的条目
    active_model_map = {k: v for k, v in MODEL_CLASS_MAP.items() if v is not None}
    
    if not active_model_map:
        print("\n❌ MODEL_CLASS_MAP is not configured. Please edit the script.")
    else:
        try:
            # --- 1. 获取测试数据加载器 ---
            test_loader = get_test_dataloader(
                data_folder=DATA_ROOT_DIR,
                batch_size=8, # 可以适当调大batch_size以加速评估
                random_state=42 
            )
            
            # --- 2. 运行评估函数 ---
            final_results = calculate_metrics_for_all_models(
                base_dir=BASE_DIR,
                model_map=active_model_map,
                test_loader=test_loader,
                device=DEVICE,
                data_range=59.0220 # 假设你的label也被归一化到了[0,1]之间，如果不是请修改
            )

            # --- 3. 整理并展示结果 ---
            if final_results:
                results_df = pd.DataFrame(final_results)
                results_df = results_df.sort_values(by='PSNR', ascending=False) # 按PSNR降序排序

                print("\n\n" + "="*80)
                print("🏆🏆🏆 FINAL PERFORMANCE METRICS 🏆🏆🏆")
                print("="*80)
                print(results_df.to_string(index=False))
                
                # 保存到CSV，方便你复制到论文里
                output_filename = "final_metrics_psnr_ssim.csv"
                results_df.to_csv(output_filename, index=False)
                print(f"\n✅ Results table saved to '{output_filename}'")

        except Exception as e:
            print(f"❌ An error occurred during the process: {e}")