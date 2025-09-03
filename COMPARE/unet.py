import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm
from datetime import datetime
import csv
import time
import math # 用于 math.sqrt

# --- 1. 文件路径设置 (保持不变) ---
# base_folder = r'/root/EMTdata'
base_folder = r'E:\EMTdata'

E_folder = os.path.join(base_folder, 'E')
if not os.path.exists(E_folder):
    print(f"警告: 路径 {E_folder} 不存在，将尝试直接在 {base_folder} 下寻找 Re, Im, F 文件夹。")
    f_folder_path = os.path.join(base_folder, 'F')
    re_folder_path = os.path.join(base_folder, 'Re')
    im_folder_path = os.path.join(base_folder, 'Im')
else:
    f_folder_path = os.path.join(E_folder, 'F')
    re_folder_path = os.path.join(E_folder, 'Re')
    im_folder_path = os.path.join(E_folder, 'Im')

label_folder = os.path.join(base_folder, 'label')
if not os.path.exists(label_folder):
    print(f"警告: 路径 {label_folder} 不存在，将尝试直接在 {base_folder} 下寻找 label_Re, label_Im 文件夹。")
    label_re_folder_path = os.path.join(base_folder, 'label_Re')
    label_im_folder_path = os.path.join(base_folder, 'label_Im')
else:
    label_re_folder_path = os.path.join(label_folder, 'label_Re')
    label_im_folder_path = os.path.join(label_folder, 'label_Im')

print(f"输入 Re 文件夹: {re_folder_path}")
print(f"输入 Im 文件夹: {im_folder_path}")
print(f"输入 F 文件夹: {f_folder_path}")
print(f"标签 Re 文件夹: {label_re_folder_path}")
print(f"标签 Im 文件夹: {label_im_folder_path}")


path_tuples = [
    ("输入Re", re_folder_path, '.xlsx'), ("输入Im", im_folder_path, '.xlsx'), ("输入F", f_folder_path, '.xlsx'),
    ("标签Re", label_re_folder_path, '.csv'), ("标签Im", label_im_folder_path, '.csv')
]
all_paths_valid = True
for name, path, ext in path_tuples:
    if not os.path.isdir(path):
        print(f"错误: 文件夹 '{name}' 在路径 '{path}' 未找到。")
        all_paths_valid = False
if not all_paths_valid: exit("程序因路径错误而终止。")

re_files_list = natsorted([f for f in os.listdir(re_folder_path) if f.lower().endswith('.xlsx')])
im_files_list = natsorted([f for f in os.listdir(im_folder_path) if f.lower().endswith('.xlsx')])
f_files_list = natsorted([f for f in os.listdir(f_folder_path) if f.lower().endswith('.xlsx')])
label_re_files_list = natsorted([f for f in os.listdir(label_re_folder_path) if f.lower().endswith('.csv')])
label_im_files_list = natsorted([f for f in os.listdir(label_im_folder_path) if f.lower().endswith('.csv')])

min_len = min(len(re_files_list), len(im_files_list), len(f_files_list), len(label_re_files_list), len(label_im_files_list))
if min_len == 0: raise ValueError(f"一个或多个数据/标签文件夹为空。")

re_files_list, im_files_list, f_files_list, label_re_files_list, label_im_files_list = \
    [lst[:min_len] for lst in [re_files_list, im_files_list, f_files_list, label_re_files_list, label_im_files_list]]

(re_train_files, re_test_files, im_train_files, im_test_files, f_train_files, f_test_files,
 label_re_train_files, label_re_test_files, label_im_train_files, label_im_test_files) = train_test_split(
    re_files_list, im_files_list, f_files_list, label_re_files_list, label_im_files_list,
    test_size=0.2, random_state=42
)

# --- 2. 数据集定义 (修改后) ---
def normalize_f_data(data, min_val=2.0, max_val=8.0):
    data = data.astype(np.float32)
    if (max_val - min_val) == 0: return data - min_val
    return (data - min_val) / (max_val - min_val)

class MatrixDataset(Dataset):
    def __init__(self, re_fnames, im_fnames, f_fnames, label_re_fnames, label_im_fnames,
                 re_mean=None, re_std=None, im_mean=None, im_std=None, calculate_stats_mode=False): # 新增统计参数和模式
        self.re_paths = [os.path.join(re_folder_path, f) for f in re_fnames]
        self.im_paths = [os.path.join(im_folder_path, f) for f in im_fnames]
        self.f_paths = [os.path.join(f_folder_path, f) for f in f_fnames]
        self.label_re_paths = [os.path.join(label_re_folder_path, f) for f in label_re_fnames]
        self.label_im_paths = [os.path.join(label_im_folder_path, f) for f in label_im_fnames]
        self.target_size = (512, 512)

        self.re_mean = re_mean
        self.re_std = re_std
        self.im_mean = im_mean
        self.im_std = im_std
        self.calculate_stats_mode = calculate_stats_mode # 用于统计计算时返回原始Re/Im数据

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

        if not self.calculate_stats_mode: # 正常模式下进行归一化
            if self.re_mean is not None and self.re_std is not None:
                re_data_resized = (re_data_resized - self.re_mean) / self.re_std
            if self.im_mean is not None and self.im_std is not None:
                im_data_resized = (im_data_resized - self.im_mean) / self.im_std
        
        input_data_np = np.stack([re_data_resized, im_data_resized, f_data_resized], axis=0)
        input_tensor = torch.from_numpy(input_data_np).float()

        if self.calculate_stats_mode: # 如果只是计算统计量，不需要标签
            return {'image': input_tensor} # 或者只返回需要统计的通道

        label_re_orig = pd.read_csv(self.label_re_paths[idx], header=None).values.astype(np.float32)
        label_im_orig = pd.read_csv(self.label_im_paths[idx], header=None).values.astype(np.float32)
        label_re_resized = cv2.resize(label_re_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_im_resized = cv2.resize(label_im_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_data_np = np.stack([label_re_resized, label_im_resized], axis=0)
        label_tensor = torch.from_numpy(label_data_np).float()

        return {'image': input_tensor, 'label': label_tensor}

# --- 3. PyTorch U-Net 模型定义 (保持不变) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__(); mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True))
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__(); self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]; diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1); return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(); self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=False):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128); self.down2 = Down(128, 256); self.down3 = Down(256, 512)
        self.dropout_enc = nn.Dropout(0.5)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.dropout_bottleneck = nn.Dropout(0.5)
        self.up1 = Up(1024, 512 // factor, bilinear); self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear); self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels_out)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4_before_dropout = self.down3(x3); x4 = self.dropout_enc(x4_before_dropout)
        x5_before_dropout = self.down4(x4); x5 = self.dropout_bottleneck(x5_before_dropout)
        out = self.up1(x5, x4); out = self.up2(out, x3); out = self.up3(out, x2)
        out = self.up4(out, x1); logits = self.outc(out)
        if logits.shape[2:] != x.shape[2:]: # 确保尺寸最终一致
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=True)
        return logits

# --- 4. 训练逻辑 (修改后) ---
def calculate_regression_metrics(outputs, labels):
    mse = F.mse_loss(outputs, labels).item()
    rmse = math.sqrt(mse) # 使用 math.sqrt
    return mse, rmse

if __name__ == '__main__':
    checkpoint_dir = r"./UNet_XLSX_Regression_Normalized_Output"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_model_path = os.path.join(checkpoint_dir, "unet_latest_model_norm.pth")
    best_model_path = os.path.join(checkpoint_dir, "unet_best_model_norm.pth")
    log_path = os.path.join(checkpoint_dir, "unet_training_log_norm.csv")

    # --- 预计算训练集的均值和标准差 ---
    print("✨ 正在计算训练集Re和Im通道的均值和标准差...")
    stat_dataset = MatrixDataset( # 不传入均值/标准差，或设置 calculate_stats_mode=True
        re_fnames=re_train_files, im_fnames=im_train_files, f_fnames=f_train_files,
        label_re_fnames=label_re_train_files, label_im_fnames=label_im_train_files, # 标签路径仍需提供，即使不用于统计输入
        calculate_stats_mode=True # 告知Dataset只为统计准备数据
    )
    stat_loader = DataLoader(stat_dataset, batch_size=16, shuffle=False, num_workers=0)

    re_sum, im_sum = 0.0, 0.0
    re_sum_sq, im_sum_sq = 0.0, 0.0
    total_pixels_re, total_pixels_im = 0, 0

    for batch_data in tqdm(stat_loader, desc="计算统计量"):
        re_channel_data = batch_data['image'][:, 0, :, :]
        im_channel_data = batch_data['image'][:, 1, :, :]
        re_sum += torch.sum(re_channel_data).item()
        im_sum += torch.sum(im_channel_data).item()
        re_sum_sq += torch.sum(torch.square(re_channel_data)).item()
        im_sum_sq += torch.sum(torch.square(im_channel_data)).item()
        total_pixels_re += re_channel_data.nelement()
        total_pixels_im += im_channel_data.nelement()
    
    if total_pixels_re == 0 or total_pixels_im == 0 :
        print("警告: 未能处理任何像素来计算Re或Im通道的统计数据。将使用默认值 (mean=0, std=1)。")
        re_mean_train, re_std_train = 0.0, 1.0
        im_mean_train, im_std_train = 0.0, 1.0
    else:
        re_mean_train = re_sum / total_pixels_re
        im_mean_train = im_sum / total_pixels_im
        re_std_train = math.sqrt(max(0, (re_sum_sq / total_pixels_re) - (re_mean_train**2)))
        im_std_train = math.sqrt(max(0, (im_sum_sq / total_pixels_im) - (im_mean_train**2)))
        re_std_train = max(re_std_train, 1e-7)
        im_std_train = max(im_std_train, 1e-7)

    print(f"计算得到的训练集统计数据:")
    print(f"  Re 通道: 均值={re_mean_train:.4f}, 标准差={re_std_train:.4f}")
    print(f"  Im 通道: 均值={im_mean_train:.4f}, 标准差={im_std_train:.4f}")
    print("--- 统计数据计算完毕 ---")

    train_dataset = MatrixDataset(
        re_fnames=re_train_files, im_fnames=im_train_files, f_fnames=f_train_files,
        label_re_fnames=label_re_train_files, label_im_fnames=label_im_train_files,
        re_mean=re_mean_train, re_std=re_std_train,
        im_mean=im_mean_train, im_std=im_std_train
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    if re_test_files:
        test_dataset = MatrixDataset(
            re_fnames=re_test_files, im_fnames=im_test_files, f_fnames=f_test_files,
            label_re_fnames=label_re_test_files, label_im_fnames=label_im_test_files,
            re_mean=re_mean_train, re_std=re_std_train,
            im_mean=im_mean_train, im_std=im_std_train
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    else:
        test_loader = None; print("警告: 未找到测试文件，跳过验证。早停策略将不可用。") # MODIFIED: Added "早停策略将不可用。"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels_in=3, n_channels_out=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    best_val_loss = float('inf')
    train_losses_history, val_losses_history = [], []
    train_mses_history, train_rmses_history = [], []
    val_mses_history, val_rmses_history = [], []

    if os.path.exists(latest_model_path):
        print(f"🟡 正在从最新的U-Net检查点恢复: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            train_losses_history = checkpoint.get('train_losses_history', [])
            val_losses_history = checkpoint.get('val_losses_history', [])
            train_mses_history = checkpoint.get('train_mses_history', [])
            train_rmses_history = checkpoint.get('train_rmses_history', [])
            val_mses_history = checkpoint.get('val_mses_history', [])
            val_rmses_history = checkpoint.get('val_rmses_history', [])
            # early_stopping_counter = checkpoint.get('early_stopping_counter', 0) # 可选：如果也保存了计数器
            print(f"✅ U-Net模型加载完毕。将从 epoch {start_epoch} 开始。最佳验证Loss: {best_val_loss:.6f}")
        except Exception as e:
            print(f"警告：加载U-Net优化器或历史指标时出错 ({e})。")
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    else:
        print("⚪ 未找到U-Net检查点。将从头开始训练。")
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_mse', 'train_rmse', 'val_loss', 'val_mse', 'val_rmse'])

    print(f'U-Net训练初始化完成。设备: {device}')
    num_epochs = 1000

    # --- 早停参数初始化 (MODIFIED) ---
    early_stopping_patience = 50  # 按要求设置为50
    early_stopping_counter = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss, epoch_train_mse, epoch_train_rmse = 0.0, 0.0, 0.0
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [U-Net训练]', ncols=120, miniters=10, mininterval=0, leave=False)
        
        for batch_idx, batch_data in enumerate(progress_bar_train):
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            mse, rmse = calculate_regression_metrics(outputs, labels)
            epoch_train_loss += loss.item(); epoch_train_mse += mse; epoch_train_rmse += rmse
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                progress_bar_train.set_postfix(loss=loss.item(), mse=mse, rmse=rmse)

        len_train_loader = len(train_loader) if len(train_loader) > 0 else 1
        avg_epoch_train_loss = epoch_train_loss / len_train_loader
        avg_epoch_train_mse = epoch_train_mse / len_train_loader
        avg_epoch_train_rmse = epoch_train_rmse / len_train_loader
        train_losses_history.append(avg_epoch_train_loss); train_mses_history.append(avg_epoch_train_mse); train_rmses_history.append(avg_epoch_train_rmse)
        tqdm.write(f"Epoch {epoch+1} U-Net训练总结: Loss={avg_epoch_train_loss:.4f}, MSE={avg_epoch_train_mse:.4f}, RMSE={avg_epoch_train_rmse:.4f}")

        avg_epoch_val_loss = float('nan') 
        avg_epoch_val_mse = float('nan')
        avg_epoch_val_rmse = float('nan')
        
        if test_loader:
            model.eval(); epoch_val_loss, epoch_val_mse, epoch_val_rmse = 0.0,0.0,0.0
            progress_bar_val = None # 初始化 progress_bar_val
            with torch.no_grad():
                progress_bar_val = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [U-Net验证]', ncols=120, leave=False)
                for batch_data in progress_bar_val:
                    inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
                    outputs = model(inputs); loss = criterion(outputs, labels)
                    mse, rmse = calculate_regression_metrics(outputs, labels)
                    epoch_val_loss += loss.item(); epoch_val_mse += mse; epoch_val_rmse += rmse
                    progress_bar_val.set_postfix(loss=loss.item(), mse=mse, rmse=rmse)
            
            len_test_loader = len(test_loader) if len(test_loader) > 0 else 1
            avg_epoch_val_loss = epoch_val_loss / len_test_loader
            avg_epoch_val_mse = epoch_val_mse / len_test_loader
            avg_epoch_val_rmse = epoch_val_rmse / len_test_loader
            
            # --- 关键: 记录实际的验证指标 (这部分在您的代码中已经是正确的) ---
            val_losses_history.append(avg_epoch_val_loss); val_mses_history.append(avg_epoch_val_mse); val_rmses_history.append(avg_epoch_val_rmse)
            tqdm.write(f"Epoch {epoch+1} U-Net验证总结: Loss={avg_epoch_val_loss:.4f}, MSE={avg_epoch_val_mse:.4f}, RMSE={avg_epoch_val_rmse:.4f}")
            
            # --- 早停逻辑判断 (MODIFIED) ---
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                             'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
                             'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
                             'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history
                             # 可选: 'early_stopping_counter': early_stopping_counter
                           }, best_model_path)
                tqdm.write(f"✅ U-Net最佳模型已保存, 验证Loss: {best_val_loss:.6f}")
                early_stopping_counter = 0 # 重置早停计数器
            else:
                early_stopping_counter += 1 # 验证损失没有改善
                tqdm.write(f"🟡 早停计数器: {early_stopping_counter}/{early_stopping_patience} (当前验证Loss: {avg_epoch_val_loss:.6f}, 最佳: {best_val_loss:.6f})")
            # --- 已移除您之前版本中在此处错误添加的 val_losses_history.append(float('nan')) ---
            if progress_bar_val: progress_bar_val.close() # 关闭验证进度条
        else: # 处理没有 test_loader 的情况 (这部分在您的代码中已经是正确的)
            val_losses_history.append(float('nan')); val_mses_history.append(float('nan')); val_rmses_history.append(float('nan'))
        
        progress_bar_train.close() 
        # if test_loader and 'progress_bar_val' in locals() and progress_bar_val: progress_bar_val.close() # 已移到 if test_loader 内部

        torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                     'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
                     'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
                     'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history
                     # 可选: 'early_stopping_counter': early_stopping_counter
                   }, latest_model_path)

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_epoch_train_loss, avg_epoch_train_mse, avg_epoch_train_rmse,
                             f"{avg_epoch_val_loss:.4f}" if not math.isnan(avg_epoch_val_loss) else "N/A",
                             f"{avg_epoch_val_mse:.4f}" if not math.isnan(avg_epoch_val_mse) else "N/A",
                             f"{avg_epoch_val_rmse:.4f}" if not math.isnan(avg_epoch_val_rmse) else "N/A"])
        
        # --- 早停检查 (这部分在您的代码中已经是正确的) ---
        if test_loader and early_stopping_counter >= early_stopping_patience:
            tqdm.write(f"🔴 早停触发！验证损失已连续 {early_stopping_patience} 个轮次没有改善。在 Epoch {epoch + 1} 停止训练。")
            break 
            
    tqdm.write('🏁 U-Net训练完成。')

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1); plt.plot(train_losses_history, label='训练损失'); plt.plot(val_losses_history, label='验证损失', linestyle='--'); plt.xlabel('轮次'); plt.ylabel('损失'); plt.legend(); plt.title('U-Net损失'); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot(train_mses_history, label='训练MSE'); plt.plot(val_mses_history, label='验证MSE', linestyle='--'); plt.xlabel('轮次'); plt.ylabel('MSE'); plt.legend(); plt.title('U-Net MSE'); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(train_rmses_history, label='训练RMSE'); plt.plot(val_rmses_history, label='验证RMSE', linestyle='--'); plt.xlabel('轮次'); plt.ylabel('RMSE'); plt.legend(); plt.title('U-Net RMSE'); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(checkpoint_dir, "training_metrics_unet_norm_zh.png"))
    print(f"📈 U-Net训练指标图表已保存到 {os.path.join(checkpoint_dir, 'training_metrics_unet_norm_zh.png')}")