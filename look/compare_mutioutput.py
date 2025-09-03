import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable # 确保这行导入在文件顶部

# --- 模型和数据加载部分 (从你的代码中复制，无需修改) ---
# ⚠️ 确保这些导入路径相对于你运行脚本的位置是正确的
from COMPARE.unet import UNet
from COMPARE.FCN import FCN_ResNet50_Regression
from COMPARE.segnet import SegNet
from COMPARE.PSPNET import PSPNet
from COMPARE.deeplab import DeepLabV3ForRegression
from COMPARE.CDLmix import ResnetWithComplexNet
from COMPARE.CDL_ASPP import DeepLabV3WithComplexNet
from COMPARE.two_channelCDL import DeepLabV3WithComplexNet2channeel
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

        input_mag_unnormalized = np.sqrt(re_data_resized**2 + im_data_resized**2)

        if not self.calculate_stats_mode:
            if self.re_mean is not None and self.re_std is not None and self.re_std > 1e-7:
                re_data_resized = (re_data_resized - self.re_mean) / self.re_std
            if self.im_mean is not None and self.im_std is not None and self.im_std > 1e-7:
                im_data_resized = (im_data_resized - self.im_mean) / self.im_std
        input_data_np = np.stack([re_data_resized, im_data_resized, f_data_resized], axis=0)
        input_tensor = torch.from_numpy(input_data_np).float()

        if self.calculate_stats_mode: return input_tensor
        
        label_re_orig = pd.read_csv(self.label_re_paths[idx], header=None).values.astype(np.float32)
        label_im_orig = pd.read_csv(self.label_im_paths[idx], header=None).values.astype(np.float32)
        label_re_resized = cv2.resize(label_re_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_im_resized = cv2.resize(label_im_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_data_np = np.stack([label_re_resized, label_im_resized], axis=0)
        label_tensor = torch.from_numpy(label_data_np).float()
        
        return input_tensor, label_tensor, input_mag_unnormalized

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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"✅ 测试数据加载器创建成功，共 {len(test_dataset)} 个样本。")
    return test_loader
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # 导入GridSpec

# ==============================================================================
# FINAL DEFINITIVE VERSION - Adding Dedicated Input Colorbars
# ==============================================================================
# def visualize_grid_comparison(base_dir, model_map, model_display_map, test_loader, device, num_samples=3):
#     print("\n🚀 Running Final Analysis with Dedicated Input Colorbars...")

#     # --- 1. 模型加载与预测 (不变) ---
#     dataset = test_loader.dataset
#     # ✅ 手动选取最具代表性的样本索引（你自己挑）
#     indices_to_visualize = [0, 5, 81]  # 举例，你可以自己选编号
#     predictions = {int(idx): {} for idx in indices_to_visualize}
#     successful_model_names = []
#     for model_name_key, model_class in model_map.items():
#         folder_path = os.path.join(base_dir, model_name_key + "_Output")
#         try:
#             model = model_class().to(device)
#             pth_path_list = glob.glob(os.path.join(folder_path, '**', '*_best_model.pth'), recursive=True)
#             if not pth_path_list: continue
#             checkpoint = torch.load(pth_path_list[-1], map_location=device)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             model.eval()
#             successful_model_names.append(model_name_key)
#             with torch.no_grad():
#                 for idx in indices_to_visualize:
#                     input_tensor, _, _ = dataset[idx]
#                     output = model(input_tensor.unsqueeze(0).to(device))
#                     predictions[idx][model_name_key] = output.squeeze().cpu().numpy()
#         except Exception as e: print(f"❌ Failed to process model {model_name_key}: {e}")
#     print(f"✅ Loaded {len(successful_model_names)} models successfully.")

#     if not successful_model_names: return

#     # --- 2. 核心绘图逻辑 (为Input添加颜色轴) ---
#     num_method_cols = 1 + 1 + len(successful_model_names) # Input, GT, + Models
#     num_rows = num_samples * 2
    
#     # ✨✨✨【核心修改】: 调整GridSpec，为Input和GT/Model的颜色轴都预留列 ✨✨✨
#     # 总列数 = 1(Input)+1(Input Cbar) + (GT+Models) + 1(GT/Models Cbar)
#     # width_ratios 控制每一列的相对宽度
#     width_ratios = [1.2, 0.15, 0.2] + [1] * (num_method_cols - 1) + [0.15]
#     num_total_cols = len(width_ratios)
#     fig = plt.figure(figsize=(num_total_cols * 1.5, num_rows *2))
#     gs = GridSpec(num_rows, num_total_cols, figure=fig, hspace=0.1, wspace=0.1, width_ratios=width_ratios)

#     for i, sample_idx in enumerate(indices_to_visualize):
#         row_offset = i * 2
        
#         # --- 数据准备 ---
#         input_tensor, ground_truth_tensor, _ = dataset[sample_idx]
#         input_re, input_im = input_tensor[0].numpy(), input_tensor[1].numpy()
#         gt_re, gt_im = ground_truth_tensor[0].numpy(), ground_truth_tensor[1].numpy()
#         preds_re = {key: p[0] for key, p in predictions.get(sample_idx, {}).items()}
#         preds_im = {key: p[1] for key, p in predictions.get(sample_idx, {}).items()}

#         #反归一化
#         # ✅ ✅ ✅ 新增：反归一化输入和标签（使用 mean/std from dataset）
#         input_re = input_re * dataset.re_std + dataset.re_mean
#         input_im = input_im * dataset.im_std + dataset.im_mean
#         gt_re = gt_re * dataset.re_std + dataset.re_mean
#         gt_im = gt_im * dataset.im_std + dataset.im_mean
#         for key in preds_re:
#             preds_re[key] = preds_re[key] * dataset.re_std + dataset.re_mean
#             preds_im[key] = preds_im[key] * dataset.im_std + dataset.im_mean

#         if not preds_re: continue

#         # --- 颜色范围计算 ---
#         v_in_re = max(abs(input_re.min()), abs(input_re.max()))
#         v_in_im = max(abs(input_im.min()), abs(input_im.max()))
        
#         comparable_re = [gt_re] + list(preds_re.values())
#         comparable_im = [gt_im] + list(preds_im.values())
#         v_gt_re = max(abs(d.min()) for d in comparable_re if d is not None)
#         v_gt_im = max(abs(d.max()) for d in comparable_im if d is not None)

#         # --- 行标签 ---
#         ax_row_label = fig.add_subplot(gs[row_offset:row_offset+2, 0])
#         full_path = test_loader.dataset.re_paths[sample_idx]
#         base_name = os.path.basename(full_path); clean_name, _ = os.path.splitext(base_name)
#         try: 
#             id_label = clean_name.split('_')[0]  # e.g. 'LR1'
#             freq_val = clean_name.split('_')[1]  # e.g. '7.5'
#             formatted_label = f"{id_label}\n$f = {freq_val}\\,\\mathrm{{GHz}}$"

#         except IndexError: formatted_label = clean_name
#         ax_row_label.set_ylabel(formatted_label, fontsize=14, rotation=0, va='center', ha='right', labelpad=10, weight='bold')
#         ax_row_label.set_facecolor('none'); [s.set_visible(False) for s in ax_row_label.spines.values()]; ax_row_label.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

#         # --- 填充所有子图 ---
#         # -- Input --
#         ax_in_re = fig.add_subplot(gs[row_offset, 0])
#         im_in_re = ax_in_re.imshow(input_re, cmap='coolwarm', vmin=-v_in_re, vmax=v_in_re)
#         if i == 0: ax_in_re.set_title("Input", fontsize=16, weight='bold')
#         ax_in_re.axis('off')
        
#         ax_in_im = fig.add_subplot(gs[row_offset+1, 0])
#         im_in_im = ax_in_im.imshow(input_im, cmap='coolwarm', vmin=-v_in_im, vmax=v_in_im)
#         ax_in_im.axis('off')

#         # ✨✨✨【核心修正】: 在预留的第1列中，为Input绘制颜色轴 ✨✨✨
#         cax_in_re = fig.add_subplot(gs[row_offset, 1])
#         cb_in_re=fig.colorbar(im_in_re, cax=cax_in_re)
#         cb_in_re.ax.tick_params(labelsize=8)
#         cb_in_re.ax.ticklabel_format(style='sci', scilimits=(0, 0)) #科学技术饭


#         cax_in_im = fig.add_subplot(gs[row_offset+1, 1])
#         cb_in_re=fig.colorbar(im_in_im, cax=cax_in_im)
#         cb_in_re.ax.ticklabel_format(style='sci', scilimits=(0, 0)) #科学技术饭


#                 # --- Spacer，占据 index=2 ---
#         ax_spacer_re = fig.add_subplot(gs[row_offset, 2])
#         ax_spacer_re.axis('off')

#         ax_spacer_im = fig.add_subplot(gs[row_offset + 1, 2])
#         ax_spacer_im.axis('off')

#         cb_in_re.ax.tick_params(labelsize=8)
        
#         # -- GT and Models --
#         gt_and_models_re = [gt_re] + [preds_re.get(key) for key in successful_model_names]
#         gt_and_models_im = [gt_im] + [preds_im.get(key) for key in successful_model_names]
        
#         for col, data in enumerate(gt_and_models_re):
#             ax_re = fig.add_subplot(gs[row_offset, col + 3])
#             im_gt_re = ax_re.imshow(data, cmap='coolwarm', vmin=-v_gt_re, vmax=v_gt_re)
#             if i == 0:
#                 title = "Ground Truth" if col == 0 else model_display_map.get(successful_model_names[col-1])
#                 ax_re.set_title(title, fontsize=16, weight='bold')
#             ax_re.axis('off')

#         for col, data in enumerate(gt_and_models_im):
#             ax_im = fig.add_subplot(gs[row_offset + 1, col + 3])
#             im_gt_im = ax_im.imshow(data, cmap='coolwarm', vmin=-v_gt_im, vmax=v_gt_im)
#             ax_im.axis('off')
            
#         # -- 为 GT和模型 添加共享颜色轴 --
#         cax_gt_re = fig.add_subplot(gs[row_offset, -1])
#         cb_gt_re = fig.colorbar(im_gt_re, cax=cax_gt_re, shrink=0.2)
#         cb_gt_re.ax.tick_params(labelsize=8)
#         cb_gt_re.ax.ticklabel_format(style='sci', scilimits=(0, 0))  # ✅ 科学计数法 ✅

#         cax_gt_im = fig.add_subplot(gs[row_offset + 1, -1])
#         cb_gt_im = fig.colorbar(im_gt_im, cax=cax_gt_im, shrink=0.2)
#         cb_gt_im.ax.tick_params(labelsize=8)
#         cb_gt_im.ax.ticklabel_format(style='sci', scilimits=(0, 0))  # ✅ 科学计数法 ✅


    
#     plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03, wspace=0.15, hspace=0.15)

#     plt.savefig("publication_ready_plot_final.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
#     print(f"\n✅ Saved final plot to 'publication_ready_plot_final.png'")


#跟上面那个版本不同，检测是否是三通道
# ==============================================================================
# FINAL DEFINITIVE VERSION - Adding Dedicated Input Colorbars
# ==============================================================================
def visualize_grid_comparison(base_dir, model_map, model_display_map, test_loader, device, num_samples=3):
    print("\n🚀 Running Final Analysis with Dedicated Input Colorbars...")

    # --- 1. 模型加载与预测 ---
    dataset = test_loader.dataset
    indices_to_visualize = [0, 5, 81] 
    predictions = {int(idx): {} for idx in indices_to_visualize}
    successful_model_names = []
    
    for model_name_key, model_class in model_map.items():
        folder_path = os.path.join(base_dir, model_name_key + "_Output")
        try:
            model = model_class().to(device)
            pth_path_list = glob.glob(os.path.join(folder_path, '**', '*_best_model.pth'), recursive=True)
            if not pth_path_list: 
                print(f"⚠️ Warning: No best_model.pth found for {model_name_key}, skipping.")
                continue
            
            checkpoint = torch.load(pth_path_list[-1], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            successful_model_names.append(model_name_key)
            
            with torch.no_grad():
                for idx in indices_to_visualize:
                    # 从数据集中获取原始的3通道输入
                    input_tensor, _, _ = dataset[idx]
                    
                    # ✨✨✨【核心修改点】✨✨✨
                    # 在这里根据模型名称，准备最终送入模型的张量
                    # 默认使用完整的3通道输入
                    input_for_model = input_tensor
                    
                    # 如果当前模型是那个特殊的2通道模型
                    if model_name_key == 'ComplexNetDeepLab_2Channel_Reg':
                        # 我们只取前两个通道 (Re, Im)
                        # input_tensor 形状是 [3, 512, 512]
                        # 切片后 input_for_model 形状变为 [2, 512, 512]
                        input_for_model = input_tensor[:2, :, :]
                        print(f"   -> Adapting input for '{model_name_key}': Using 2 channels.")

                    # 将准备好的、通道数正确的张量送入模型
                    output = model(input_for_model.unsqueeze(0).to(device))
                    predictions[idx][model_name_key] = output.squeeze().cpu().numpy()
                    
        except Exception as e:
            print(f"❌ Failed to process model {model_name_key}: {e}")
            
    print(f"✅ Loaded {len(successful_model_names)} models successfully.")

    if not successful_model_names: return

    # --- 2. 核心绘图逻辑 (这部分无需修改) ---
    num_method_cols = 1 + 1 + len(successful_model_names) # Input, GT, + Models
    num_rows = num_samples * 2
    
    width_ratios = [1.2, 0.15, 0.2] + [1] * (num_method_cols - 1) + [0.15]
    num_total_cols = len(width_ratios)
    fig = plt.figure(figsize=(num_total_cols * 1.5, num_rows * 2))
    gs = GridSpec(num_rows, num_total_cols, figure=fig, hspace=0.1, wspace=0.1, width_ratios=width_ratios)

    for i, sample_idx in enumerate(indices_to_visualize):
        row_offset = i * 2
        
        # --- 数据准备 ---
        input_tensor, ground_truth_tensor, _ = dataset[sample_idx]
        input_re, input_im = input_tensor[0].numpy(), input_tensor[1].numpy()
        gt_re, gt_im = ground_truth_tensor[0].numpy(), ground_truth_tensor[1].numpy()
        preds_re = {key: p[0] for key, p in predictions.get(sample_idx, {}).items()}
        preds_im = {key: p[1] for key, p in predictions.get(sample_idx, {}).items()}

        # --- 反归一化 ---
        input_re = input_re * dataset.re_std + dataset.re_mean
        input_im = input_im * dataset.im_std + dataset.im_mean
        gt_re = gt_re * dataset.re_std + dataset.re_mean
        gt_im = gt_im * dataset.im_std + dataset.im_mean
        for key in preds_re:
            preds_re[key] = preds_re[key] * dataset.re_std + dataset.re_mean
            preds_im[key] = preds_im[key] * dataset.im_std + dataset.im_mean

        if not preds_re: continue

        # --- 颜色范围计算 ---
        v_in_re = max(abs(input_re.min()), abs(input_re.max()))
        v_in_im = max(abs(input_im.min()), abs(input_im.max()))
        
        comparable_re = [gt_re] + list(preds_re.values())
        comparable_im = [gt_im] + list(preds_im.values())
        v_gt_re = max(abs(d.min()) for d in comparable_re if d is not None)
        v_gt_im = max(abs(d.max()) for d in comparable_im if d is not None)

        # --- 行标签 ---
        ax_row_label = fig.add_subplot(gs[row_offset:row_offset+2, 0])
        full_path = test_loader.dataset.re_paths[sample_idx]
        base_name = os.path.basename(full_path); clean_name, _ = os.path.splitext(base_name)
        try: 
            id_label = clean_name.split('_')[0]
            freq_val = clean_name.split('_')[1]
            formatted_label = f"{id_label}\n$f = {freq_val}\\,\\mathrm{{GHz}}$"
        except IndexError: 
            formatted_label = clean_name
        ax_row_label.set_ylabel(formatted_label, fontsize=14, rotation=0, va='center', ha='right', labelpad=10, weight='bold')
        ax_row_label.set_facecolor('none'); [s.set_visible(False) for s in ax_row_label.spines.values()]; ax_row_label.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # --- 填充所有子图 ---
        # -- Input --
        ax_in_re = fig.add_subplot(gs[row_offset, 0])
        im_in_re = ax_in_re.imshow(input_re, cmap='coolwarm', vmin=-v_in_re, vmax=v_in_re)
        if i == 0: ax_in_re.set_title("Input", fontsize=16, weight='bold')
        ax_in_re.axis('off')
        
        ax_in_im = fig.add_subplot(gs[row_offset+1, 0])
        im_in_im = ax_in_im.imshow(input_im, cmap='coolwarm', vmin=-v_in_im, vmax=v_in_im)
        ax_in_im.axis('off')

        # -- Input Colorbars --
        cax_in_re = fig.add_subplot(gs[row_offset, 1])
        cb_in_re = fig.colorbar(im_in_re, cax=cax_in_re)
        cb_in_re.ax.tick_params(labelsize=8)
        cb_in_re.ax.ticklabel_format(style='sci', scilimits=(0, 0))

        cax_in_im = fig.add_subplot(gs[row_offset+1, 1])
        cb_in_im = fig.colorbar(im_in_im, cax=cax_in_im)
        cb_in_im.ax.tick_params(labelsize=8)
        cb_in_im.ax.ticklabel_format(style='sci', scilimits=(0, 0))

        # -- Spacer --
        ax_spacer_re = fig.add_subplot(gs[row_offset, 2]); ax_spacer_re.axis('off')
        ax_spacer_im = fig.add_subplot(gs[row_offset + 1, 2]); ax_spacer_im.axis('off')
        
        # -- GT and Models --
        gt_and_models_re = [gt_re] + [preds_re.get(key) for key in successful_model_names]
        gt_and_models_im = [gt_im] + [preds_im.get(key) for key in successful_model_names]
        
        im_gt_re, im_gt_im = None, None # To store the last image artist for colorbar

        for col, data in enumerate(gt_and_models_re):
            ax_re = fig.add_subplot(gs[row_offset, col + 3])
            im_gt_re = ax_re.imshow(data, cmap='coolwarm', vmin=-v_gt_re, vmax=v_gt_re)
            if i == 0:
                title = "Ground Truth" if col == 0 else model_display_map.get(successful_model_names[col-1], "Unknown")
                ax_re.set_title(title, fontsize=16, weight='bold')
            ax_re.axis('off')

        for col, data in enumerate(gt_and_models_im):
            ax_im = fig.add_subplot(gs[row_offset + 1, col + 3])
            im_gt_im = ax_im.imshow(data, cmap='coolwarm', vmin=-v_gt_im, vmax=v_gt_im)
            ax_im.axis('off')
            
        # -- GT and Models Shared Colorbars --
        if im_gt_re:
            cax_gt_re = fig.add_subplot(gs[row_offset, -1])
            cb_gt_re = fig.colorbar(im_gt_re, cax=cax_gt_re)
            cb_gt_re.ax.tick_params(labelsize=8)
            cb_gt_re.ax.ticklabel_format(style='sci', scilimits=(0, 0))
        
        if im_gt_im:
            cax_gt_im = fig.add_subplot(gs[row_offset + 1, -1])
            cb_gt_im = fig.colorbar(im_gt_im, cax=cax_gt_im)
            cb_gt_im.ax.tick_params(labelsize=8)
            cb_gt_im.ax.ticklabel_format(style='sci', scilimits=(0, 0))
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

    plt.savefig("publication_ready_plot_final.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"\n✅ Saved final plot to 'publication_ready_plot_final.png'")

if __name__ == "__main__":
    # --- 全局配置 ---
    DATA_ROOT_DIR = "E:/EMTdata"  # 你的数据根目录
    BASE_DIR = "E:/EMMESData/Remote/" # 你的模型输出根目录
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 模型列表 ---
    # 确保键名是你希望在图上显示的标题
    MODEL_CLASS_MAP = {
        # 'SegNet_XLSX': lambda: SegNet(input_channels=3, output_channels=2),
        # 'PSPNet_Regression_Unified': PSPNet,
        # 'FCN_XLSX_Regression': FCN_ResNet50_Regression,
        # 'UNet_XLSX_Regression_Normalized': lambda: UNet(n_channels_in=3, n_channels_out=2),
        'DeepLabV3_Reg_UnifiedData': DeepLabV3ForRegression,
        'ComplexNetDeepLab_Reg': ResnetWithComplexNet,
        'ComplexNetDeepLab+aspp_Reg': DeepLabV3WithComplexNet,
        'ComplexNetDeepLab_2Channel_Reg': DeepLabV3WithComplexNet2channeel
    }
    
    MODEL_DISPLAY_NAMES = {
    # 'SegNet_XLSX': 'SegNet',
    # 'PSPNet_Regression_Unified': 'PSPNet',
    # 'FCN_XLSX_Regression': 'FCN',
    # 'UNet_XLSX_Regression_Normalized': 'UNet',
    # 'ComplexNetDeepLab+aspp_Reg': 'CDL(Ours)'
    'DeepLabV3_Reg_UnifiedData': 'DeepLabV3',
    'ComplexNetDeepLab_Reg': 'CDL w/o ASPP',
    'ComplexNetDeepLab_2Channel_Reg': 'CDL(2 channel)', # "Ours" 表示这是你的主要模型
    'ComplexNetDeepLab+aspp_Reg': 'CDL(Ours)'
    }

    # 假设 get_test_dataloader 函数已正确定义
    test_loader = get_test_dataloader(data_folder=DATA_ROOT_DIR, batch_size=1, random_state=42)
    
    if test_loader:
        visualize_grid_comparison(
            base_dir=BASE_DIR,
            model_map=MODEL_CLASS_MAP,
            model_display_map=MODEL_DISPLAY_NAMES,
            test_loader=test_loader,
            device=DEVICE,
            num_samples=3)