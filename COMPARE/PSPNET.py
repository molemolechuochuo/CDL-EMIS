import os
import numpy as np
import pandas as pd
import cv2 # 用于 MatrixDataset 中的 resize
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models # PSPNet 是自定义的，不需要此行加载预训练骨干
from sklearn.model_selection import train_test_split
from natsort import natsorted
from tqdm import tqdm
from datetime import datetime
import csv
import time # 保留
import math # 用于 math.sqrt
import matplotlib.pyplot as plt # 用于绘图

# --- 1. 文件路径设置 (与FCN参考脚本风格一致) ---
# data_folder = r'/root/autodl-fs' # 示例路径
data_folder = r'E:\EMTdata' # 您PSPNet脚本提供的路径

# 统一使用 _path 后缀作为文件夹变量名
E_folder_path_check = os.path.join(data_folder, 'E')
if not os.path.exists(E_folder_path_check):
    print(f"警告: 路径 {E_folder_path_check} 不存在，将尝试直接在 {data_folder} 下寻找 Re, Im, F 文件夹。")
    f_folder_path = os.path.join(data_folder, 'F')
    re_folder_path = os.path.join(data_folder, 'Re')
    im_folder_path = os.path.join(data_folder, 'Im')
else:
    f_folder_path = os.path.join(E_folder_path_check, 'F')
    re_folder_path = os.path.join(E_folder_path_check, 'Re')
    im_folder_path = os.path.join(E_folder_path_check, 'Im')

label_folder_path_check = os.path.join(data_folder, 'label')
if not os.path.exists(label_folder_path_check):
    print(f"警告: 路径 {label_folder_path_check} 不存在，将尝试直接在 {data_folder} 下寻找 label_Re, label_Im 文件夹。")
    label_re_folder_path = os.path.join(data_folder, 'label_Re')
    label_im_folder_path = os.path.join(data_folder, 'label_Im')
else:
    label_re_folder_path = os.path.join(label_folder_path_check, 'label_Re')
    label_im_folder_path = os.path.join(label_folder_path_check, 'label_Im')

print(f"输入 Re 文件夹: {re_folder_path}")
print(f"输入 Im 文件夹: {im_folder_path}")
print(f"输入 F 文件夹: {f_folder_path}")
print(f"标签 Re 文件夹: {label_re_folder_path}")
print(f"标签 Im 文件夹: {label_im_folder_path}")

path_tuples_validation = [ # Renamed for clarity
    ("输入Re", re_folder_path), ("输入Im", im_folder_path), ("输入F", f_folder_path),
    ("标签Re", label_re_folder_path), ("标签Im", label_im_folder_path)
]
all_paths_valid_check = True
for name, path_val in path_tuples_validation:
    if not os.path.isdir(path_val):
        print(f"错误: 文件夹 '{name}' 在路径 '{path_val}' 未找到。")
        all_paths_valid_check = False
if not all_paths_valid_check: exit("程序因路径错误而终止。")

re_files_list_all_names = natsorted([f for f in os.listdir(re_folder_path) if f.lower().endswith('.xlsx')])
im_files_list_all_names = natsorted([f for f in os.listdir(im_folder_path) if f.lower().endswith('.xlsx')])
f_files_list_all_names = natsorted([f for f in os.listdir(f_folder_path) if f.lower().endswith('.xlsx')])
label_re_files_list_all_names = natsorted([f for f in os.listdir(label_re_folder_path) if f.lower().endswith('.csv')])
label_im_files_list_all_names = natsorted([f for f in os.listdir(label_im_folder_path) if f.lower().endswith('.csv')])

min_len_all_files = min(len(re_files_list_all_names), len(im_files_list_all_names), len(f_files_list_all_names), len(label_re_files_list_all_names), len(label_im_files_list_all_names))
if min_len_all_files == 0: raise ValueError(f"一个或多个数据/标签文件夹为空或不包含预期文件。")

# 使用截断后的文件列表 (仅文件名)
re_files_names = re_files_list_all_names[:min_len_all_files]
im_files_names = im_files_list_all_names[:min_len_all_files]
f_files_names = f_files_list_all_names[:min_len_all_files]
label_re_files_names = label_re_files_list_all_names[:min_len_all_files]
label_im_files_names = label_im_files_list_all_names[:min_len_all_files]

(re_train_files, re_val_files, # Renamed test to val for clarity in training loop
 im_train_files, im_val_files,
 f_train_files, f_val_files,
 label_re_train_files, label_re_val_files,
 label_im_train_files, label_im_val_files) = train_test_split(
    re_files_names, im_files_names, f_files_names, label_re_files_names, label_im_files_names,
    test_size=0.2, random_state=42
)

# --- 2. 数据集定义 (采纳FCN参考脚本中的 MatrixDataset) ---
def normalize_f_data(data, min_val=2.0, max_val=8.0):
    data = data.astype(np.float32)
    if (max_val - min_val) == 0: return data - min_val
    return (data - min_val) / (max_val - min_val)

class MatrixDataset(Dataset): # 从FCN参考脚本复制而来
    def __init__(self, re_fnames, im_fnames, f_fnames, label_re_fnames, label_im_fnames,
                 re_mean=None, re_std=None, im_mean=None, im_std=None, calculate_stats_mode=False):
        self.re_paths = [os.path.join(re_folder_path, f) for f in re_fnames]
        self.im_paths = [os.path.join(im_folder_path, f) for f in im_fnames]
        self.f_paths = [os.path.join(f_folder_path, f) for f in f_fnames]
        self.label_re_paths = [os.path.join(label_re_folder_path, f) for f in label_re_fnames]
        self.label_im_paths = [os.path.join(label_im_folder_path, f) for f in label_im_fnames]
        self.target_size = (512, 512)

        self.re_mean, self.re_std = re_mean, re_std
        self.im_mean, self.im_std = im_mean, im_std
        self.calculate_stats_mode = calculate_stats_mode

    def __len__(self):
        return len(self.re_paths)

    def __getitem__(self, idx):
        try:
            re_data = pd.read_excel(self.re_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
            im_data = pd.read_excel(self.im_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
            f_data_orig = pd.read_excel(self.f_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        except Exception as e:
            print(f"读取输入文件时出错 (index {idx}): {e}, paths: R={self.re_paths[idx]}, I={self.im_paths[idx]}, F={self.f_paths[idx]}")
            raise
            
        f_data_normalized = normalize_f_data(f_data_orig)

        re_data_resized = cv2.resize(re_data, self.target_size, interpolation=cv2.INTER_LINEAR)
        im_data_resized = cv2.resize(im_data, self.target_size, interpolation=cv2.INTER_LINEAR)
        f_data_resized = cv2.resize(f_data_normalized, self.target_size, interpolation=cv2.INTER_LINEAR)

        if not self.calculate_stats_mode:
            if self.re_mean is not None and self.re_std is not None:
                re_data_resized = (re_data_resized - self.re_mean) / self.re_std
            if self.im_mean is not None and self.im_std is not None:
                im_data_resized = (im_data_resized - self.im_mean) / self.im_std
        
        input_data_np = np.stack([re_data_resized, im_data_resized, f_data_resized], axis=0)
        input_tensor = torch.from_numpy(input_data_np).float()

        if self.calculate_stats_mode:
            return {'image': input_tensor} 

        try:
            label_re_orig = pd.read_csv(self.label_re_paths[idx], header=None).values.astype(np.float32)
            label_im_orig = pd.read_csv(self.label_im_paths[idx], header=None).values.astype(np.float32)
        except Exception as e:
            print(f"读取标签文件时出错 (index {idx}): {e}, paths: LR={self.label_re_paths[idx]}, LI={self.label_im_paths[idx]}")
            raise
            
        label_re_resized = cv2.resize(label_re_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_im_resized = cv2.resize(label_im_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_data_np = np.stack([label_re_resized, label_im_resized], axis=0)
        label_tensor = torch.from_numpy(label_data_np).float()

        return {'image': input_tensor, 'label': label_tensor}

# --- 3. PSPNet 模型定义 (保持您提供的结构) ---
class PSPNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=2): # Default input_channels
        super(PSPNet, self).__init__()
        # Encoder (Feature extraction) - Simplified for example, replace with actual backbone e.g. ResNet
        self.initial_block = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), # Example deeper block
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Output size /4
        )
        # Example feature channels after a backbone, adjust if using a real backbone
        # Assuming `initial_block` outputs features with `feature_dim` channels.
        # For a ResNet50, this might be 2048. For the simple block above, it's 128.
        # Let's assume for this custom PSPNet, the output of `initial_block` is, say, 128 channels.
        # If you use a ResNet backbone, this dimension would be much larger (e.g., 2048 for ResNet50 layer4)
        feature_dim = 128 # Placeholder, adjust based on your actual encoder output
        
        # Pyramid Pooling Module
        self.pyramid_pooling = PPM(feature_dim, [1, 2, 3, 6]) # PPM class defined below
        
        # Decoder
        # The input channels to decoder is feature_dim + num_ppm_features (e.g., feature_dim * 2 if PPM halves channels and concatenates)
        # PPM class below concatenates original features + pooled features (each pooled is feature_dim / num_pools)
        # So, input to conv_upsample1 is feature_dim + (feature_dim/N_pools * N_pools) = feature_dim * 2
        # But the provided PSPNet concatenates the output of self.enc with PPM outputs.
        # Let's follow the user's original PPM concatenation logic more closely.
        # Their original PPM implied `enc_features_dim` + `4 * ppm_conv_output_dim`
        # If `enc_features_dim`=64 (from their original enc) and `ppm_conv_output_dim`=64, then total is 64 + 4*64 = 320
        
        # Replicating user's original structure for PPM part
        self.enc_for_ppm = nn.Sequential( # Simplified encoder as in user's script
             nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(2) # Output will be (H/2, W/2)
        )
        ppm_in_channels = 64 # Output channels of self.enc_for_ppm
        self.ppm_pools = nn.ModuleList([ # Renamed for clarity
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.AdaptiveAvgPool2d(output_size=(6, 6))
        ])
        self.ppm_convs = nn.ModuleList([ # Renamed for clarity
            nn.Conv2d(ppm_in_channels, ppm_in_channels // 4, kernel_size=1, bias=False), # Reduce channels
            nn.Conv2d(ppm_in_channels, ppm_in_channels // 4, kernel_size=1, bias=False),
            nn.Conv2d(ppm_in_channels, ppm_in_channels // 4, kernel_size=1, bias=False),
            nn.Conv2d(ppm_in_channels, ppm_in_channels // 4, kernel_size=1, bias=False)
        ])
        # After concat: ppm_in_channels (from enc_for_ppm) + 4 * (ppm_in_channels // 4) = 2 * ppm_in_channels
        decoder_in_channels = ppm_in_channels * 2

        self.decoder_conv = nn.Sequential( # Renamed for clarity
            nn.Conv2d(decoder_in_channels, ppm_in_channels, kernel_size=3, padding=1, bias=False), # Use ppm_in_channels
            nn.BatchNorm2d(ppm_in_channels),
            nn.ReLU(inplace=True),
            # The ConvTranspose2d needs to bring it back to original size from H/2, W/2
            nn.ConvTranspose2d(ppm_in_channels, output_channels, kernel_size=4, stride=2, padding=1) # Adjusted for upsampling
        )

    def forward(self, x):
        input_size = x.shape[-2:] # H, W
        
        enc_features = self.enc_for_ppm(x) # (B, 64, H/2, W/2)
        
        ppm_outputs = [enc_features]
        for pool, conv_layer in zip(self.ppm_pools, self.ppm_convs):
            pooled = pool(enc_features)
            conved = conv_layer(pooled)
            upsampled = F.interpolate(conved, size=enc_features.shape[-2:], mode='bilinear', align_corners=False)
            ppm_outputs.append(upsampled)
        
        x_ppm_cat = torch.cat(ppm_outputs, dim=1) # (B, 64*2, H/2, W/2)
        
        out = self.decoder_conv(x_ppm_cat) # Should output (B, output_channels, H, W)
        
        # Ensure final output size matches input size if necessary, or target label size (512,512)
        if out.shape[-2:] != input_size: # Or use target_size (512,512) directly
             out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out

# Helper PPM module (can be integrated or kept separate)
# This is a more standard PPM module if you decide to use a deeper backbone later
class PPM(nn.Module):
    def __init__(self, in_dim, pool_scales):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.stages = nn.ModuleList([self._make_stage(in_dim, size) for size in pool_scales])
        # Each stage outputs in_dim / len(pool_scales) channels
        self.conv_upsample = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, bias=False) # Example
        # Or more typically:
        # self.bottleneck = nn.Conv2d(in_dim * (1 + len(pool_scales)), out_dim, kernel_size=1, bias=False)


    def _make_stage(self, in_dim, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        # Each pooled feature map reduced to in_dim / N channels where N is number of pool_scales
        conv = nn.Conv2d(in_dim, in_dim // len(self.pool_scales), kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        # Concatenate original features with pooled features
        bottle = torch.cat(priors + [feats], 1)
        # return self.bottleneck(bottle)
        return bottle # Needs further processing, e.g. by self.conv_upsample or decoder

# --- 4. 训练逻辑 ---
def calculate_regression_metrics(outputs, labels):
    # Ensure labels and outputs have the same size, especially for interpolate after model
    if outputs.shape[-2:] != labels.shape[-2:]:
        outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    mse = F.mse_loss(outputs, labels).item()
    rmse = math.sqrt(mse)
    return mse, rmse

if __name__ == '__main__':
    model_name = "PSPNet_Regression_Unified"
    checkpoint_dir = f"./{model_name}_Output"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_model_path = os.path.join(checkpoint_dir, f"{model_name}_latest_model.pth")
    best_model_path = os.path.join(checkpoint_dir, f"{model_name}_best_model.pth")
    log_path = os.path.join(checkpoint_dir, f"{model_name}_training_log.csv")

    # --- 预计算训练集的均值和标准差 (采纳FCN参考脚本的风格) ---
    print("✨ 正在计算训练集Re和Im通道的均值和标准差...")
    stat_dataset = MatrixDataset(
        re_fnames=re_train_files, im_fnames=im_train_files, f_fnames=f_train_files,
        label_re_fnames=label_re_train_files, label_im_fnames=label_im_train_files,
        calculate_stats_mode=True
    )
    stat_loader = DataLoader(stat_dataset, batch_size=16, shuffle=False, num_workers=0)

    re_sum, im_sum, re_sum_sq, im_sum_sq = 0.0, 0.0, 0.0, 0.0
    total_pixels_re, total_pixels_im = 0, 0
    for batch_data in tqdm(stat_loader, desc="计算统计量"):
        re_ch, im_ch = batch_data['image'][:, 0, :, :], batch_data['image'][:, 1, :, :]
        re_sum += torch.sum(re_ch).item(); im_sum += torch.sum(im_ch).item()
        re_sum_sq += torch.sum(torch.square(re_ch)).item(); im_sum_sq += torch.sum(torch.square(im_ch)).item()
        total_pixels_re += re_ch.nelement(); total_pixels_im += im_ch.nelement()
    
    if not total_pixels_re or not total_pixels_im:
        print("警告: 统计数据计算时未处理任何像素。Re/Im归一化将使用(0,1)。")
        re_m_train, re_s_train, im_m_train, im_s_train = 0.0, 1.0, 0.0, 1.0
    else:
        re_m_train = re_sum / total_pixels_re; im_m_train = im_sum / total_pixels_im
        re_s_train = math.sqrt(max(0, (re_sum_sq/total_pixels_re) - (re_m_train**2))); re_s_train = max(re_s_train, 1e-7)
        im_s_train = math.sqrt(max(0, (im_sum_sq/total_pixels_im) - (im_m_train**2))); im_s_train = max(im_s_train, 1e-7)
    print(f"统计结果: Re(均值={re_m_train:.4f}, 标准差={re_s_train:.4f}), Im(均值={im_m_train:.4f}, 标准差={im_s_train:.4f})")
    print("--- 统计数据计算完毕 ---")

    # --- 创建正式的数据集和加载器 ---
    train_dataset = MatrixDataset(
        re_fnames=re_train_files, im_fnames=im_train_files, f_fnames=f_train_files,
        label_re_fnames=label_re_train_files, label_im_fnames=label_im_train_files,
        re_mean=re_m_train, re_std=re_s_train, im_mean=im_m_train, im_std=im_s_train
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True) # drop_last=True

    val_loader = None
    if re_val_files:
        val_dataset = MatrixDataset(
            re_fnames=re_val_files, im_fnames=im_val_files, f_fnames=f_val_files,
            label_re_fnames=label_re_val_files, label_im_fnames=label_im_val_files,
            re_mean=re_m_train, re_std=re_s_train, im_mean=im_m_train, im_std=im_s_train
        )
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    else:
        print("警告: 未找到验证文件，将跳过验证。早停策略将不可用。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PSPNet(input_channels=3, output_channels=2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 使用您PSPNet脚本的Adam和lr
    criterion = nn.MSELoss()

    start_epoch = 0; best_val_loss = float('inf')
    train_losses_history, val_losses_history = [], []
    train_mses_history, train_rmses_history = [], []
    val_mses_history, val_rmses_history = [], []

    early_stopping_patience = 50
    early_stopping_counter = 0

    if os.path.exists(latest_model_path):
        print(f"🟡 正在从最新的{model_name}检查点恢复: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            train_losses_history=checkpoint.get('train_losses_history', [])
            val_losses_history=checkpoint.get('val_losses_history', [])
            train_mses_history=checkpoint.get('train_mses_history', [])
            train_rmses_history=checkpoint.get('train_rmses_history', [])
            val_mses_history=checkpoint.get('val_mses_history', [])
            val_rmses_history=checkpoint.get('val_rmses_history', [])
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            print(f"✅ {model_name}模型加载完毕。将从 epoch {start_epoch} 开始。最佳验证Loss: {best_val_loss:.6f}, 早停计数: {early_stopping_counter}")
        except Exception as e:
            print(f"警告：加载{model_name}优化器或历史指标时出错 ({e})。")
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    else:
        print(f"⚪ 未找到{model_name}检查点。将从头开始训练。")
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_mse', 'train_rmse', 'val_loss', 'val_mse', 'val_rmse'])
    
    print(f'设备: {device}')
    num_epochs = 1000 # 您脚本中的设置

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss, epoch_train_mse, epoch_train_rmse = 0.0, 0.0, 0.0
        # 使用您PSPNet脚本中的tqdm设置 (leave=True, miniters=10, 具体step打印)
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{model_name} 训练]', ncols=130, leave=True)
        
        for step, batch_data in enumerate(progress_bar_train):
            inputs = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.shape[-2:] != labels.shape[-2:]: # 确保输出和标签尺寸一致
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            mse, rmse = calculate_regression_metrics(outputs.detach(), labels.detach())
            epoch_train_loss += loss.item()
            epoch_train_mse += mse
            epoch_train_rmse += rmse
            
            # 使用您PSPNet脚本中的set_postfix 和条件打印
            current_postfix = f"L={loss.item():.3f}|MSE={mse:.3f}"
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                 progress_bar_train.set_postfix_str(current_postfix, refresh=True) # refresh=True if needed
                 # timestamp = datetime.now().strftime('%H:%M:%S') # tqdm.write is better for multi-line
                 # tqdm.write(f"    [{timestamp}] Ep{epoch+1} Stp{step+1}/{len(train_loader)} Loss: {loss.item():.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
            else:
                 progress_bar_train.set_postfix_str(current_postfix, refresh=False)


        # 训练epoch总结
        len_tr_loader = len(train_loader) if len(train_loader) > 0 else 1
        avg_tr_loss = epoch_train_loss / len_tr_loader
        avg_tr_mse = epoch_train_mse / len_tr_loader
        avg_tr_rmse = epoch_train_rmse / len_tr_loader
        train_losses_history.append(avg_tr_loss)
        train_mses_history.append(avg_tr_mse)
        train_rmses_history.append(avg_tr_rmse)
        tqdm.write(f"Epoch {epoch+1} [{model_name} 训练总结]: Loss={avg_tr_loss:.4f}, MSE={avg_tr_mse:.4f}, RMSE={avg_tr_rmse:.4f}")
        progress_bar_train.close() # 在验证前关闭训练进度条

        # 验证
        avg_epoch_val_loss, avg_epoch_val_mse, avg_epoch_val_rmse = float('nan'), float('nan'), float('nan')
        progress_bar_val = None

        if val_loader:
            model.eval()
            epoch_val_loss, epoch_val_mse, epoch_val_rmse = 0.0,0.0,0.0
            with torch.no_grad():
                progress_bar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{model_name} 验证]', ncols=130, leave=False)
                for batch_data in progress_bar_val:
                    inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
                    outputs = model(inputs)
                    if outputs.shape[-2:] != labels.shape[-2:]:
                         outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                    loss = criterion(outputs, labels)
                    mse, rmse = calculate_regression_metrics(outputs, labels)
                    epoch_val_loss += loss.item(); epoch_val_mse += mse; epoch_val_rmse += rmse
                    progress_bar_val.set_postfix_str(f"L={loss.item():.3f}|MSE={mse:.3f}", refresh=True)
            if progress_bar_val: progress_bar_val.close()

            len_val_loader = len(val_loader) if len(val_loader) > 0 else 1
            avg_epoch_val_loss = epoch_val_loss / len_val_loader
            avg_epoch_val_mse = epoch_val_mse / len_val_loader
            avg_epoch_val_rmse = epoch_val_rmse / len_val_loader
            val_losses_history.append(avg_epoch_val_loss); val_mses_history.append(avg_epoch_val_mse); val_rmses_history.append(avg_epoch_val_rmse)
            tqdm.write(f"Epoch {epoch+1} [{model_name} 验证总结]: Loss={avg_epoch_val_loss:.4f}, MSE={avg_epoch_val_mse:.4f}, RMSE={avg_epoch_val_rmse:.4f}")

            if not math.isnan(avg_epoch_val_loss): # 确保验证损失有效
                if avg_epoch_val_loss < best_val_loss: # 之前FCN参考用了 min_delta，这里简化为直接比较
                    best_val_loss = avg_epoch_val_loss
                    early_stopping_counter = 0 # 重置计数器
                    torch.save({
                        'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                        'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
                        'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
                        'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history,
                        'early_stopping_counter': early_stopping_counter
                    }, best_model_path)
                    tqdm.write(f"✅ {model_name} 最佳模型已保存, 验证Loss: {best_val_loss:.6f}")
                else:
                    early_stopping_counter += 1
                    tqdm.write(f"🟡 {model_name} 早停计数器: {early_stopping_counter}/{early_stopping_patience} (当前验证Loss: {avg_epoch_val_loss:.6f}, 最佳: {best_val_loss:.6f})")
        else:
            val_losses_history.append(float('nan')); val_mses_history.append(float('nan')); val_rmses_history.append(float('nan'))
            early_stopping_counter = 0 # 如果没有验证集，也重置计数器或不更新

        # 保存最新模型
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
            'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
            'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
            'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history,
            'early_stopping_counter': early_stopping_counter
        }, latest_model_path)

        # CSV日志记录
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # 使用 epoch+1 记录轮次，与之前脚本一致
            writer.writerow([epoch + 1, avg_tr_loss, avg_tr_mse, avg_tr_rmse,
                             f"{avg_epoch_val_loss:.4f}" if not math.isnan(avg_epoch_val_loss) else "N/A",
                             f"{avg_epoch_val_mse:.4f}" if not math.isnan(avg_epoch_val_mse) else "N/A",
                             f"{avg_epoch_val_rmse:.4f}" if not math.isnan(avg_epoch_val_rmse) else "N/A"])
        
        # 早停检查
        if val_loader and early_stopping_counter >= early_stopping_patience:
            tqdm.write(f"🔴 {model_name} 早停触发！验证损失已连续 {early_stopping_patience} 个轮次没有改善。在 Epoch {epoch + 1} 停止训练。")
            break
            
    tqdm.write(f'🏁 {model_name} 训练完成。')

    # 绘图
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1); plt.plot(train_losses_history, label='训练损失'); plt.plot(val_losses_history, label='验证损失', linestyle='--'); plt.xlabel('轮次'); plt.ylabel('损失'); plt.legend(); plt.title(f'{model_name} 损失'); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot(train_mses_history, label='训练MSE'); plt.plot(val_mses_history, label='验证MSE', linestyle='--'); plt.xlabel('轮次'); plt.ylabel('MSE'); plt.legend(); plt.title(f'{model_name} MSE'); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(train_rmses_history, label='训练RMSE'); plt.plot(val_rmses_history, label='验证RMSE', linestyle='--'); plt.xlabel('轮次'); plt.ylabel('RMSE'); plt.legend(); plt.title(f'{model_name} RMSE'); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(checkpoint_dir, f"{model_name}_training_metrics_zh.png"))
    print(f"📈 {model_name} 训练指标图表已保存到 {os.path.join(checkpoint_dir, f'{model_name}_training_metrics_zh.png')}")
