import os
import numpy as np
import pandas as pd
import cv2 # Used in MatrixDataset for resize
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models # For DeepLabV3
from sklearn.model_selection import train_test_split
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import math
import time

# --- 1. 文件路径设置 (与FCN参考脚本风格一致) ---
# base_folder = r'/root/EMTdata' # Linux 示例路径 (from FCN ref)
# base_folder = r'E:\CellSegnetTset\U-net-master\data_set4\矩阵' # Your DeepLabV3 data path
base_folder = r'E:\EMTdata' # 使用您DeepLabV3脚本原有的路径

# 统一使用 _path 后缀作为文件夹变量名，与FCN参考一致
E_folder_path_check = os.path.join(base_folder, 'E') # For checking E subfolder existence
if not os.path.exists(E_folder_path_check):
    print(f"警告: 路径 {E_folder_path_check} 不存在，将尝试直接在 {base_folder} 下寻找 Re, Im, F 文件夹。")
    f_folder_path = os.path.join(base_folder, 'F')
    re_folder_path = os.path.join(base_folder, 'Re')
    im_folder_path = os.path.join(base_folder, 'Im')
else:
    f_folder_path = os.path.join(E_folder_path_check, 'F')
    re_folder_path = os.path.join(E_folder_path_check, 'Re')
    im_folder_path = os.path.join(E_folder_path_check, 'Im')

label_folder_path_check = os.path.join(base_folder, 'label') # For checking label subfolder
if not os.path.exists(label_folder_path_check):
    print(f"警告: 路径 {label_folder_path_check} 不存在，将尝试直接在 {base_folder} 下寻找 label_Re, label_Im 文件夹。")
    label_re_folder_path = os.path.join(base_folder, 'label_Re')
    label_im_folder_path = os.path.join(base_folder, 'label_Im')
else:
    label_re_folder_path = os.path.join(label_folder_path_check, 'label_Re')
    label_im_folder_path = os.path.join(label_folder_path_check, 'label_Im')

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
for name, path, ext_expected in path_tuples: # ext_expected is for clarity, not used in isdir
    if not os.path.isdir(path):
        print(f"错误: 文件夹 '{name}' 在路径 '{path}' 未找到。")
        all_paths_valid = False
if not all_paths_valid: exit("程序因路径错误而终止。")

re_files_list_all = natsorted([f for f in os.listdir(re_folder_path) if f.lower().endswith('.xlsx')])
im_files_list_all = natsorted([f for f in os.listdir(im_folder_path) if f.lower().endswith('.xlsx')])
f_files_list_all = natsorted([f for f in os.listdir(f_folder_path) if f.lower().endswith('.xlsx')])
label_re_files_list_all = natsorted([f for f in os.listdir(label_re_folder_path) if f.lower().endswith('.csv')])
label_im_files_list_all = natsorted([f for f in os.listdir(label_im_folder_path) if f.lower().endswith('.csv')])

min_len_files_count = min(len(re_files_list_all), len(im_files_list_all), len(f_files_list_all), len(label_re_files_list_all), len(label_im_files_list_all))
if min_len_files_count == 0: raise ValueError(f"一个或多个数据/标签文件夹为空。")

# 使用截断后的文件列表 (仅文件名)
re_files_list = re_files_list_all[:min_len_files_count]
im_files_list = im_files_list_all[:min_len_files_count]
f_files_list = f_files_list_all[:min_len_files_count]
label_re_files_list = label_re_files_list_all[:min_len_files_count]
label_im_files_list = label_im_files_list_all[:min_len_files_count]

# 划分训练集和验证集文件名列表
(re_train_files, re_val_files,
 im_train_files, im_val_files,
 f_train_files, f_val_files,
 label_re_train_files, label_re_val_files,
 label_im_train_files, label_im_val_files) = train_test_split(
    re_files_list, im_files_list, f_files_list, label_re_files_list, label_im_files_list,
    test_size=0.2, random_state=42 # test_size=0.2 表示验证集占20%
)

# --- 2. 数据集定义 (直接使用FCN参考脚本中的 MatrixDataset) ---
def normalize_f_data(data, min_val=2.0, max_val=8.0): # 与FCN参考一致
    data = data.astype(np.float32)
    if (max_val - min_val) == 0: return data - min_val
    return (data - min_val) / (max_val - min_val)

class MatrixDataset(Dataset): # 从FCN参考脚本复制而来
    def __init__(self, re_fnames, im_fnames, f_fnames, label_re_fnames, label_im_fnames,
                 re_mean=None, re_std=None, im_mean=None, im_std=None, calculate_stats_mode=False):
        # 使用全局定义的 folder_path 变量来构建完整路径
        self.re_paths = [os.path.join(re_folder_path, f) for f in re_fnames]
        self.im_paths = [os.path.join(im_folder_path, f) for f in im_fnames]
        self.f_paths = [os.path.join(f_folder_path, f) for f in f_fnames]
        # 即使在统计模式下，也构建标签路径，getitem 中会根据模式决定是否加载
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

        if not self.calculate_stats_mode: # 正常模式或验证模式
            if self.re_mean is not None and self.re_std is not None:
                re_data_resized = (re_data_resized - self.re_mean) / self.re_std
            if self.im_mean is not None and self.im_std is not None:
                im_data_resized = (im_data_resized - self.im_mean) / self.im_std
        
        input_data_np = np.stack([re_data_resized, im_data_resized, f_data_resized], axis=0) # C, H, W
        input_tensor = torch.from_numpy(input_data_np).float()

        if self.calculate_stats_mode:
            return {'image': input_tensor} # 统计模式下不返回标签

        try:
            label_re_orig = pd.read_csv(self.label_re_paths[idx], header=None).values.astype(np.float32)
            label_im_orig = pd.read_csv(self.label_im_paths[idx], header=None).values.astype(np.float32)
        except Exception as e:
            print(f"读取标签文件时出错 (index {idx}): {e}, paths: LR={self.label_re_paths[idx]}, LI={self.label_im_paths[idx]}")
            raise
            
        label_re_resized = cv2.resize(label_re_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_im_resized = cv2.resize(label_im_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_data_np = np.stack([label_re_resized, label_im_resized], axis=0) # C, H, W
        label_tensor = torch.from_numpy(label_data_np).float()

        return {'image': input_tensor, 'label': label_tensor}

# --- 3. 模型定义 (DeepLabV3ForRegression 保持不变) ---
class DeepLabV3ForRegression(nn.Module):
    def __init__(self, output_channels=2):
        super(DeepLabV3ForRegression, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Conv2d(256, output_channels, kernel_size=(1, 1), stride=(1, 1))
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
             self.model.aux_classifier[-1] = nn.Conv2d(256, output_channels, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        output = self.model(x)
        return output['out']

# --- 4. 训练逻辑 (与之前DeepLabV3脚本的标准化训练逻辑一致) ---
def calculate_regression_metrics(outputs, labels):
    if outputs.shape[-2:] != labels.shape[-2:]:
        outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    mse = F.mse_loss(outputs, labels).item()
    rmse = math.sqrt(mse)
    return mse, rmse

if __name__ == '__main__':
    model_name = "DeepLabV3_Reg_UnifiedData" # 更新模型名称以反映数据处理变化
    checkpoint_dir = f"./{model_name}_Output"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_model_path = os.path.join(checkpoint_dir, f"{model_name}_latest_model.pth")
    best_model_path = os.path.join(checkpoint_dir, f"{model_name}_best_model.pth")
    log_path = os.path.join(checkpoint_dir, f"{model_name}_training_log.csv")

    # --- 预计算训练集的均值和标准差 (采纳FCN参考脚本的风格) ---
    print("✨ 正在计算训练集Re和Im通道的均值和标准差...")
    stat_dataset = MatrixDataset( # 使用MatrixDataset
        re_fnames=re_train_files, im_fnames=im_train_files, f_fnames=f_train_files,
        label_re_fnames=label_re_train_files, label_im_fnames=label_im_train_files, # 即使统计模式也传递
        calculate_stats_mode=True
    )
    # FCN参考中使用batch_size=64进行统计，这里也采纳，可根据内存调整
    stat_loader = DataLoader(stat_dataset, batch_size=64, shuffle=False, num_workers=0)

    re_sum, im_sum = 0.0, 0.0
    re_sum_sq, im_sum_sq = 0.0, 0.0
    total_pixels_re, total_pixels_im = 0, 0

    for batch_data in tqdm(stat_loader, desc="计算统计量"):
        re_ch = batch_data['image'][:, 0, :, :]
        im_ch = batch_data['image'][:, 1, :, :]
        re_sum += torch.sum(re_ch).item()
        im_sum += torch.sum(im_ch).item()
        re_sum_sq += torch.sum(torch.square(re_ch)).item()
        im_sum_sq += torch.sum(torch.square(im_ch)).item()
        total_pixels_re += re_ch.nelement()
        total_pixels_im += im_ch.nelement()
    
    if not total_pixels_re or not total_pixels_im: # 与FCN参考脚本一致
        print("警告: 统计数据计算时未处理任何像素。Re/Im归一化将使用(0,1)。")
        re_m_train, re_s_train, im_m_train, im_s_train = 0.0, 1.0, 0.0, 1.0
    else:
        re_m_train = re_sum / total_pixels_re
        im_m_train = im_sum / total_pixels_im
        re_s_train = math.sqrt(max(0, (re_sum_sq/total_pixels_re) - (re_m_train**2)))
        im_s_train = math.sqrt(max(0, (im_sum_sq/total_pixels_im) - (im_m_train**2)))
        re_s_train = max(re_s_train, 1e-7) # 防止标准差为0
        im_s_train = max(im_s_train, 1e-7)
    print(f"统计结果: Re(均值={re_m_train:.4f}, 标准差={re_s_train:.4f}), Im(均值={im_m_train:.4f}, 标准差={im_s_train:.4f})")
    print("--- 统计数据计算完毕 ---")

    # --- 正式创建数据集和数据加载器 ---
    # MatrixDataset 内部处理 resize 和 to_tensor，所以外部 transform 为 None
    train_dataset = MatrixDataset(
        re_fnames=re_train_files, im_fnames=im_train_files, f_fnames=f_train_files,
        label_re_fnames=label_re_train_files, label_im_fnames=label_im_train_files,
        re_mean=re_m_train, re_std=re_s_train,
        im_mean=im_m_train, im_std=im_s_train,
        calculate_stats_mode=False # 正常模式
    )
    # DeepLabV3原脚本使用batch_size=8，这里保持
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    val_loader = None
    if re_val_files: # 确保使用正确的验证集文件名列表
        val_dataset = MatrixDataset(
            re_fnames=re_val_files, im_fnames=im_val_files, f_fnames=f_val_files,
            label_re_fnames=label_re_val_files, label_im_fnames=label_im_val_files,
            re_mean=re_m_train, re_std=re_s_train, # 验证集也用训练集的统计数据
            im_mean=im_m_train, im_std=im_s_train,
            calculate_stats_mode=False
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    else:
        print("警告: 未找到验证文件，将跳过验证。早停策略将不可用。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLabV3ForRegression(output_channels=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # 保持标准化风格的Adam和学习率

    start_epoch = 0
    best_val_loss = float('inf')
    train_losses_history, val_losses_history = [], []
    train_mses_history, train_rmses_history = [], []
    val_mses_history, val_rmses_history = [], []

    early_stopping_patience = 50
    early_stopping_counter = 0 # FCN参考用的是 epochs_no_improve，这里统一为 early_stopping_counter

    if os.path.exists(latest_model_path):
        print(f"🟡 正在从最新的{model_name}检查点恢复: {latest_model_path}")
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
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0) # 恢复计数器
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
    num_epochs = 1000 # 与您DeepLabV3脚本原设定一致

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss, epoch_train_mse, epoch_train_rmse = 0.0, 0.0, 0.0
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{model_name} 训练]', ncols=120, leave=False)

        for batch_data in progress_bar_train:
            inputs = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            mse, rmse = calculate_regression_metrics(outputs.detach(), labels.detach())
            epoch_train_loss += loss.item()
            epoch_train_mse += mse
            epoch_train_rmse += rmse
            progress_bar_train.set_postfix(loss=loss.item(), mse=mse, rmse=rmse)
        progress_bar_train.close()

        avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_epoch_train_mse = epoch_train_mse / len(train_loader) if len(train_loader) > 0 else 0
        avg_epoch_train_rmse = epoch_train_rmse / len(train_loader) if len(train_loader) > 0 else 0
        train_losses_history.append(avg_epoch_train_loss)
        train_mses_history.append(avg_epoch_train_mse)
        train_rmses_history.append(avg_epoch_train_rmse)
        tqdm.write(f"Epoch {epoch+1} [{model_name} 训练总结]: Loss={avg_epoch_train_loss:.4f}, MSE={avg_epoch_train_mse:.4f}, RMSE={avg_epoch_train_rmse:.4f}")

        avg_epoch_val_loss, avg_epoch_val_mse, avg_epoch_val_rmse = float('nan'), float('nan'), float('nan')
        progress_bar_val = None

        if val_loader:
            model.eval()
            epoch_val_loss, epoch_val_mse, epoch_val_rmse = 0.0, 0.0, 0.0
            with torch.no_grad():
                progress_bar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{model_name} 验证]', ncols=120, leave=False)
                for batch_data in progress_bar_val:
                    inputs = batch_data['image'].to(device)
                    labels = batch_data['label'].to(device)
                    outputs = model(inputs)
                    if outputs.shape[-2:] != labels.shape[-2:]:
                         outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                    loss = criterion(outputs, labels)
                    mse, rmse = calculate_regression_metrics(outputs, labels)
                    epoch_val_loss += loss.item(); epoch_val_mse += mse; epoch_val_rmse += rmse
                    progress_bar_val.set_postfix(loss=loss.item(), mse=mse, rmse=rmse)
            if progress_bar_val: progress_bar_val.close()

            avg_epoch_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0
            avg_epoch_val_mse = epoch_val_mse / len(val_loader) if len(val_loader) > 0 else 0
            avg_epoch_val_rmse = epoch_val_rmse / len(val_loader) if len(val_loader) > 0 else 0
            val_losses_history.append(avg_epoch_val_loss); val_mses_history.append(avg_epoch_val_mse); val_rmses_history.append(avg_epoch_val_rmse)
            tqdm.write(f"Epoch {epoch+1} [{model_name} 验证总结]: Loss={avg_epoch_val_loss:.4f}, MSE={avg_epoch_val_mse:.4f}, RMSE={avg_epoch_val_rmse:.4f}")

            # 早停逻辑 (与FCN参考脚本的 min_delta 不同，这里用我们之前统一的计数器逻辑)
            if avg_epoch_val_loss < best_val_loss: # 可以加入 min_delta 判断: best_val_loss - avg_epoch_val_loss > min_delta
                best_val_loss = avg_epoch_val_loss
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                    'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
                    'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
                    'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history,
                    'early_stopping_counter': 0 # 重置计数器
                }, best_model_path)
                tqdm.write(f"✅ {model_name} 最佳模型已保存, 验证Loss: {best_val_loss:.6f}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                tqdm.write(f"🟡 {model_name} 早停计数器: {early_stopping_counter}/{early_stopping_patience} (当前验证Loss: {avg_epoch_val_loss:.6f}, 最佳: {best_val_loss:.6f})")
        else:
            val_losses_history.append(float('nan')); val_mses_history.append(float('nan')); val_rmses_history.append(float('nan'))
            # 如果没有验证集，早停计数器不增加，下面的早停检查也不会触发

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
            'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
            'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
            'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history,
            'early_stopping_counter': early_stopping_counter # 保存当前计数器
        }, latest_model_path)

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_epoch_train_loss, avg_epoch_train_mse, avg_epoch_train_rmse,
                             f"{avg_epoch_val_loss:.4f}" if not math.isnan(avg_epoch_val_loss) else "N/A",
                             f"{avg_epoch_val_mse:.4f}" if not math.isnan(avg_epoch_val_mse) else "N/A",
                             f"{avg_epoch_val_rmse:.4f}" if not math.isnan(avg_epoch_val_rmse) else "N/A"])
        
        if val_loader and early_stopping_counter >= early_stopping_patience:
            tqdm.write(f"🔴 {model_name} 早停触发！验证损失已连续 {early_stopping_patience} 个轮次没有改善。在 Epoch {epoch + 1} 停止训练。")
            break
            
    tqdm.write(f'🏁 {model_name} 训练完成。')

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1); plt.plot(train_losses_history, label='训练损失'); plt.plot(val_losses_history, label='验证损失', linestyle='--'); plt.xlabel('轮次'); plt.ylabel('损失'); plt.legend(); plt.title(f'{model_name} 损失'); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot(train_mses_history, label='训练MSE'); plt.plot(val_mses_history, label='验证MSE', linestyle='--'); plt.xlabel('轮次'); plt.ylabel('MSE'); plt.legend(); plt.title(f'{model_name} MSE'); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(train_rmses_history, label='训练RMSE'); plt.plot(val_rmses_history, label='验证RMSE', linestyle='--'); plt.xlabel('轮次'); plt.ylabel('RMSE'); plt.legend(); plt.title(f'{model_name} RMSE'); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(checkpoint_dir, f"{model_name}_training_metrics_zh.png"))
    print(f"📈 {model_name} 训练指标图表已保存到 {os.path.join(checkpoint_dir, f'{model_name}_training_metrics_zh.png')}")