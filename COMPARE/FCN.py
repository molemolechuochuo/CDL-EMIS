import os
import numpy as np
import pandas as pd
import cv2 # 用于 MatrixDataset 中的 resize
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models # 用于 FCN
from sklearn.model_selection import train_test_split
from natsort import natsorted
from tqdm import tqdm
from datetime import datetime
import csv
import time # 保留
import math # 用于 math.sqrt
import matplotlib.pyplot as plt # 用于绘图

# --- 1. 文件路径设置 ---
# base_folder = r'/root/EMTdata' # Linux 示例路径
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

# --- 2. 数据集定义 (与之前脚本的 MatrixDataset 一致) ---
def normalize_f_data(data, min_val=2.0, max_val=8.0):
    data = data.astype(np.float32)
    if (max_val - min_val) == 0: return data - min_val
    return (data - min_val) / (max_val - min_val)

class MatrixDataset(Dataset):
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
        re_data = pd.read_excel(self.re_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        im_data = pd.read_excel(self.im_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        f_data_orig = pd.read_excel(self.f_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
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

        label_re_orig = pd.read_csv(self.label_re_paths[idx], header=None).values.astype(np.float32)
        label_im_orig = pd.read_csv(self.label_im_paths[idx], header=None).values.astype(np.float32)
        label_re_resized = cv2.resize(label_re_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_im_resized = cv2.resize(label_im_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_data_np = np.stack([label_re_resized, label_im_resized], axis=0)
        label_tensor = torch.from_numpy(label_data_np).float()

        return {'image': input_tensor, 'label': label_tensor}

# --- 3. FCN 模型定义 (用于回归) ---
class FCN_ResNet50_Regression(nn.Module):
    def __init__(self, n_output_channels=2):
        super(FCN_ResNet50_Regression, self).__init__()
        # 加载预训练的 FCN ResNet50 模型
        self.model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT, progress=True)
        
        # FCN ResNet50 的分类器 self.model.classifier 是一个 FCNHead，其最后一个卷积层是 self.model.classifier[-1] 或 self.model.classifier[4]
        # 这个卷积层的 in_channels 是 512 (来自ResNet50的中间层FCNHead处理后)
        # 我们需要将其 out_channels 修改为我们的回归任务所需的通道数 (2)
        # 同时，对于回归任务，通常不需要最后的激活函数
        self.model.classifier[-1] = nn.Conv2d(512, n_output_channels, kernel_size=1, stride=1)
        
        # 如果模型有辅助分类器 (aux_classifier)，也可能需要修改或移除
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            # FCN ResNet50 的 aux_classifier 的最后一个卷积层 in_channels 是 256
            # self.model.aux_classifier[-1] = nn.Conv2d(256, n_output_channels, kernel_size=1, stride=1)
            # 通常在微调或回归时，我们会移除或不使用辅助分类器损失
            self.model.aux_classifier = None


    def forward(self, x):
        output = self.model(x)
        # FCN ResNet50 返回一个 OrderedDict，主要输出在 'out' 键
        # 如果 aux_classifier 被移除，可能只返回 'out'
        return output['out']

# --- 4. 训练逻辑 ---
def calculate_regression_metrics(outputs, labels):
    mse = F.mse_loss(outputs, labels).item()
    rmse = math.sqrt(mse)
    return mse, rmse

if __name__ == '__main__':
    checkpoint_dir = r"./FCN_XLSX_Regression_Output" # 为FCN设置新的输出目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_model_path = os.path.join(checkpoint_dir, "fcn_latest_model.pth")
    best_model_path = os.path.join(checkpoint_dir, "fcn_best_model.pth")
    log_path = os.path.join(checkpoint_dir, "fcn_training_log.csv")

    # --- 预计算训练集的均值和标准差 ---
    print("✨ 正在计算训练集Re和Im通道的均值和标准差...")
    stat_dataset = MatrixDataset(
        re_fnames=re_train_files, im_fnames=im_train_files, f_fnames=f_train_files,
        label_re_fnames=label_re_train_files, label_im_fnames=label_im_train_files,
        calculate_stats_mode=True
    )
    stat_loader = DataLoader(stat_dataset, batch_size=64, shuffle=False, num_workers=0) # num_workers=0 在Windows上通常更安全

    re_sum, im_sum, re_sum_sq, im_sum_sq = 0.0, 0.0, 0.0, 0.0
    total_pixels_re, total_pixels_im = 0, 0
    for batch_data in tqdm(stat_loader, desc="计算统计量"):
        re_ch, im_ch = batch_data['image'][:, 0, :, :], batch_data['image'][:, 1, :, :]
        re_sum += torch.sum(re_ch).item(); im_sum += torch.sum(im_ch).item()
        re_sum_sq += torch.sum(torch.square(re_ch)).item(); im_sum_sq += torch.sum(torch.square(im_ch)).item()
        total_pixels_re += re_ch.nelement(); total_pixels_im += im_ch.nelement()
    
    if not total_pixels_re or not total_pixels_im:
        print("警告: 统计数据计算时未处理任何像素。Re/Im归一化将使用(0,1)。")
        re_m, re_s, im_m, im_s = 0.0, 1.0, 0.0, 1.0
    else:
        re_m = re_sum / total_pixels_re; im_m = im_sum / total_pixels_im
        re_s = math.sqrt(max(0, (re_sum_sq/total_pixels_re) - (re_m**2))); re_s = max(re_s, 1e-7)
        im_s = math.sqrt(max(0, (im_sum_sq/total_pixels_im) - (im_m**2))); im_s = max(im_s, 1e-7)
    print(f"统计结果: Re(均值={re_m:.4f}, 标准差={re_s:.4f}), Im(均值={im_m:.4f}, 标准差={im_s:.4f})")
    print("--- 统计数据计算完毕 ---")

    # --- 创建正式的数据集和加载器 ---
    train_dataset = MatrixDataset(
        re_fnames=re_train_files, im_fnames=im_train_files, f_fnames=f_train_files,
        label_re_fnames=label_re_train_files, label_im_fnames=label_im_train_files,
        re_mean=re_m, re_std=re_s, im_mean=im_m, im_std=im_s
    )
    # 您脚本中 batch_size=4, num_workers=4
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True) # 暂时用 num_workers=0
    
    if re_test_files:
        test_dataset = MatrixDataset(
            re_fnames=re_test_files, im_fnames=im_test_files, f_fnames=f_test_files,
            label_re_fnames=label_re_test_files, label_im_fnames=label_im_test_files,
            re_mean=re_m, re_std=re_s, im_mean=im_m, im_std=im_s
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    else:
        test_loader = None; print("警告: 未找到测试文件，跳过验证。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCN_ResNet50_Regression(n_output_channels=2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 使用您脚本中的 Adam 和 lr
    criterion = nn.MSELoss()

    # 早停参数
    early_stopping_patience = 50  # 如果验证损失连续 patience 轮没有改善则停止
    min_delta = 1e-3             # 被认为是“改善”的最小损失下降值
    epochs_no_improve = 0        # 记录验证损失没有改善的轮次数

    start_epoch = 0; best_val_loss = float('inf')
    train_losses_history, val_losses_history = [], []
    train_mses_history, train_rmses_history = [], []
    val_mses_history, val_rmses_history = [], []

    if os.path.exists(latest_model_path):
        print(f"🟡 正在从最新的FCN检查点恢复: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0) # 恢复早停计数器
            train_losses_history=checkpoint.get('train_losses_history',[]); val_losses_history=checkpoint.get('val_losses_history',[])
            train_mses_history=checkpoint.get('train_mses_history',[]); train_rmses_history=checkpoint.get('train_rmses_history',[])
            val_mses_history=checkpoint.get('val_mses_history',[]); val_rmses_history=checkpoint.get('val_rmses_history',[])
            print(f"✅ FCN模型加载完毕。将从 epoch {start_epoch} 开始。最佳验证Loss: {best_val_loss:.6f}, 未改善轮次: {epochs_no_improve}")
        except Exception as e:
            print(f"警告：加载FCN优化器或历史指标时出错 ({e})。")
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    else:
        print(f"⚪ 未找到FCN检查点 '{os.path.basename(latest_model_path)}'。将从头开始训练。")
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f: csv.writer(f).writerow(['epoch', 'train_loss', 'train_mse', 'train_rmse', 'val_loss', 'val_mse', 'val_rmse'])
    
    print(f'FCN训练初始化完成。设备: {device}')
    num_epochs = 1000 # 您脚本中的设置

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss, epoch_train_mse, epoch_train_rmse = 0.0, 0.0, 0.0
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [FCN训练]', ncols=130, leave=False)
        
        for step, batch_data in enumerate(progress_bar_train):
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            mse, rmse = calculate_regression_metrics(outputs, labels)
            epoch_train_loss += loss.item(); epoch_train_mse += mse; epoch_train_rmse += rmse
            
            progress_bar_train.set_postfix_str(f"L={loss.item():.3f}|MSE={mse:.3f}", refresh=True)
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                timestamp = datetime.now().strftime('%H:%M:%S')
                tqdm.write(f"    [{timestamp}] Ep{epoch+1} Stp{step+1}/{len(train_loader)} Loss: {loss.item():.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        len_loader = len(train_loader) if len(train_loader) > 0 else 1
        avg_tr_loss, avg_tr_mse, avg_tr_rmse = epoch_train_loss/len_loader, epoch_train_mse/len_loader, epoch_train_rmse/len_loader
        train_losses_history.append(avg_tr_loss); train_mses_history.append(avg_tr_mse); train_rmses_history.append(avg_tr_rmse)
        tqdm.write(f"Epoch {epoch+1} FCN训练总结: Loss={avg_tr_loss:.4f}, MSE={avg_tr_mse:.4f}, RMSE={avg_tr_rmse:.4f}")

        avg_epoch_val_loss, avg_epoch_val_mse, avg_epoch_val_rmse = float('nan'), float('nan'), float('nan')
        if test_loader:
            model.eval(); epoch_val_loss, epoch_val_mse, epoch_val_rmse = 0.0,0.0,0.0
            with torch.no_grad():
                progress_bar_val = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [FCN验证]', ncols=130, leave=False)
                for batch_data in progress_bar_val:
                    inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
                    outputs = model(inputs); loss = criterion(outputs, labels)
                    mse, rmse = calculate_regression_metrics(outputs, labels)
                    epoch_val_loss += loss.item(); epoch_val_mse += mse; epoch_val_rmse += rmse
                    progress_bar_val.set_postfix_str(f"L={loss.item():.3f}|MSE={mse:.3f}", refresh=True)
            len_loader = len(test_loader) if len(test_loader) > 0 else 1
            avg_epoch_val_loss = epoch_val_loss / len_loader
            avg_epoch_val_mse = epoch_val_mse / len_loader
            avg_epoch_val_rmse = epoch_val_rmse / len_loader
            val_losses_history.append(avg_epoch_val_loss); val_mses_history.append(avg_epoch_val_mse); val_rmses_history.append(avg_epoch_val_rmse)
            tqdm.write(f"Epoch {epoch+1} FCN验证总结: Loss={avg_epoch_val_loss:.4f}, MSE={avg_epoch_val_mse:.4f}, RMSE={avg_epoch_val_rmse:.4f}")

            if not math.isnan(avg_epoch_val_loss):
                if avg_epoch_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_epoch_val_loss
                    epochs_no_improve = 0
                    torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                                 'best_val_loss': best_val_loss, 'epochs_no_improve': epochs_no_improve, # 保存早停计数器
                                 'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
                                 'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
                                 'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history}, best_model_path)
                    tqdm.write(f"✅ FCN最佳模型已保存 (验证Loss改善): {best_val_loss:.6f}")
                else:
                    epochs_no_improve += 1
                    tqdm.write(f"Epoch {epoch+1}: 验证Loss未改善 ({avg_epoch_val_loss:.6f} vs best {best_val_loss:.6f}). 未改善轮次: {epochs_no_improve}/{early_stopping_patience}")

        else: # 没有验证集的情况
            val_losses_history.append(float('nan')); val_mses_history.append(float('nan')); val_rmses_history.append(float('nan'))
            # 如果没有验证集，早停也无法工作，可以考虑在这里给出提示或不启用早停
            epochs_no_improve = 0 # 重置或不更新

        if hasattr(progress_bar_train, 'close'): progress_bar_train.close()
        if test_loader and 'progress_bar_val' in locals() and hasattr(progress_bar_val, 'close'): progress_bar_val.close()
        
        torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                     'best_val_loss': best_val_loss, 'epochs_no_improve': epochs_no_improve, # 保存早停计数器
                     'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
                     'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
                     'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history}, latest_model_path)

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_tr_loss, avg_tr_mse, avg_tr_rmse,
                             f"{avg_epoch_val_loss:.4f}" if not math.isnan(avg_epoch_val_loss) else "N/A",
                             f"{avg_epoch_val_mse:.4f}" if not math.isnan(avg_epoch_val_mse) else "N/A",
                             f"{avg_epoch_val_rmse:.4f}" if not math.isnan(avg_epoch_val_rmse) else "N/A"])
        
        if test_loader and epochs_no_improve >= early_stopping_patience:
            tqdm.write(f"🛑 早停触发: 验证Loss连续 {early_stopping_patience} 轮未改善。在Epoch {epoch + 1} 停止训练。")
            break # 跳出epoch训练循环
            
    tqdm.write('🏁 FCN训练完成。')

    plt.figure(figsize=(18, 6)); plt.subplot(1, 3, 1)
    plt.plot(train_losses_history, label='训练损失'); plt.plot(val_losses_history, label='验证损失', linestyle='--')
    plt.xlabel('轮次'); plt.ylabel('损失'); plt.legend(); plt.title('FCN损失'); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot(train_mses_history, label='训练MSE'); plt.plot(val_mses_history, label='验证MSE', linestyle='--')
    plt.xlabel('轮次'); plt.ylabel('MSE'); plt.legend(); plt.title('FCN MSE'); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(train_rmses_history, label='训练RMSE'); plt.plot(val_rmses_history, label='验证RMSE', linestyle='--')
    plt.xlabel('轮次'); plt.ylabel('RMSE'); plt.legend(); plt.title('FCN RMSE'); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(checkpoint_dir, "training_metrics_fcn_norm_zh.png"))
    print(f"📈 FCN训练指标图表已保存。")