import os
import numpy as np
import pandas as pd
import cv2 # 用于 MatrixDataset 中的 resize
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models # 用于 DeepLabV3 的 backbone
from sklearn.model_selection import train_test_split
from natsort import natsorted
from tqdm import tqdm
from datetime import datetime
import csv
import time # 可选，但有时用于调试
import math # 用于 math.sqrt
import matplotlib.pyplot as plt # 用于绘图
from complextorch import CVTensor # 您脚本中用到的复数张量库


# --- 1. 文件路径设置 (已移除频率F通道) ---
data_folder = r'E:\EMTdata' # 您脚本提供的路径

# 统一使用 _path 后缀作为文件夹变量名
E_folder_path_check = os.path.join(data_folder, 'E') # 用于检查E子文件夹是否存在
if not os.path.exists(E_folder_path_check):
    print(f"警告: 路径 {E_folder_path_check} 不存在，将尝试直接在 {data_folder} 下寻找 Re, Im 文件夹。")
    re_folder_path = os.path.join(data_folder, 'Re')
    im_folder_path = os.path.join(data_folder, 'Im')
else:
    re_folder_path = os.path.join(E_folder_path_check, 'Re')
    im_folder_path = os.path.join(E_folder_path_check, 'Im')

label_folder_path_check = os.path.join(data_folder, 'label') # 用于检查label子文件夹
if not os.path.exists(label_folder_path_check):
    print(f"警告: 路径 {label_folder_path_check} 不存在，将尝试直接在 {data_folder} 下寻找 label_Re, label_Im 文件夹。")
    label_re_folder_path = os.path.join(data_folder, 'label_Re')
    label_im_folder_path = os.path.join(data_folder, 'label_Im')
else:
    label_re_folder_path = os.path.join(label_folder_path_check, 'label_Re')
    label_im_folder_path = os.path.join(label_folder_path_check, 'label_Im')

print(f"输入 Re 文件夹: {re_folder_path}")
print(f"输入 Im 文件夹: {im_folder_path}")
# print(f"输入 F 文件夹: {f_folder_path}") # 已移除
print(f"标签 Re 文件夹: {label_re_folder_path}")
print(f"标签 Im 文件夹: {label_im_folder_path}")

path_tuples_for_validation = [ # 重命名以避免与Python内置path冲突
    ("输入Re", re_folder_path), ("输入Im", im_folder_path),
    ("标签Re", label_re_folder_path), ("标签Im", label_im_folder_path)
]
all_paths_exist = True
for name, path_to_check in path_tuples_for_validation:
    if not os.path.isdir(path_to_check):
        print(f"错误: 文件夹 '{name}' 在路径 '{path_to_check}' 未找到。")
        all_paths_exist = False
if not all_paths_exist: exit("程序因路径错误而终止。")

re_all_filenames = natsorted([f for f in os.listdir(re_folder_path) if f.lower().endswith('.xlsx')])
im_all_filenames = natsorted([f for f in os.listdir(im_folder_path) if f.lower().endswith('.xlsx')])
# f_all_filenames = natsorted([f for f in os.listdir(f_folder_path) if f.lower().endswith('.xlsx')]) # 已移除
label_re_all_filenames = natsorted([f for f in os.listdir(label_re_folder_path) if f.lower().endswith('.csv')])
label_im_all_filenames = natsorted([f for f in os.listdir(label_im_folder_path) if f.lower().endswith('.csv')])

# 修改: min_file_count 不再包含 f_all_filenames
min_file_count = min(len(re_all_filenames), len(im_all_filenames),
                     len(label_re_all_filenames), len(label_im_all_filenames))
if min_file_count == 0: raise ValueError(f"一个或多个数据/标签文件夹为空或不包含预期文件。")

# 使用截断后的文件列表 (仅文件名)
re_filenames = re_all_filenames[:min_file_count]
im_filenames = im_all_filenames[:min_file_count]
# f_filenames = f_all_filenames[:min_file_count] # 已移除
label_re_filenames = label_re_all_filenames[:min_file_count]
label_im_filenames = label_im_all_filenames[:min_file_count]

# 修改: train_test_split 不再处理 f 文件
(re_train_files, re_val_files,
 im_train_files, im_val_files,
 label_re_train_files, label_re_val_files,
 label_im_train_files, label_im_val_files) = train_test_split(
    re_filenames, im_filenames,
    label_re_filenames, label_im_filenames,
    test_size=0.2, random_state=42
)

# --- 2. 数据集定义 (修改为2通道) ---
class MatrixDataset(Dataset):
    # 修改: __init__ 不再接收 f_fnames
    def __init__(self, re_fnames, im_fnames, label_re_fnames, label_im_fnames,
                 re_mean=None, re_std=None, im_mean=None, im_std=None, calculate_stats_mode=False):
        self.re_paths = [os.path.join(re_folder_path, f) for f in re_fnames]
        self.im_paths = [os.path.join(im_folder_path, f) for f in im_fnames]
        # self.f_paths = [os.path.join(f_folder_path, f) for f in f_fnames] # 已移除

        self.label_re_paths = [os.path.join(label_re_folder_path, f) for f in label_re_fnames] if label_re_fnames else []
        self.label_im_paths = [os.path.join(label_im_folder_path, f) for f in label_im_fnames] if label_im_fnames else []
        
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
            # f_data_orig = pd.read_excel(self.f_paths[idx], header=0, engine='openpyxl').values.astype(np.float32) # 已移除
        except Exception as e:
            # 修改: 错误信息中移除 F 路径
            print(f"读取输入文件时出错 (index {idx}): {e}, paths: R={self.re_paths[idx]}, I={self.im_paths[idx]}")
            raise
            
        # f_data_normalized = normalize_f_data(f_data_orig) # 已移除

        re_data_resized = cv2.resize(re_data, self.target_size, interpolation=cv2.INTER_LINEAR)
        im_data_resized = cv2.resize(im_data, self.target_size, interpolation=cv2.INTER_LINEAR)
        # f_data_resized = cv2.resize(f_data_normalized, self.target_size, interpolation=cv2.INTER_LINEAR) # 已移除

        if not self.calculate_stats_mode:
            if self.re_mean is not None and self.re_std is not None and self.re_std > 1e-7:
                re_data_resized = (re_data_resized - self.re_mean) / self.re_std
            if self.im_mean is not None and self.im_std is not None and self.im_std > 1e-7:
                im_data_resized = (im_data_resized - self.im_mean) / self.im_std
        
        # 修改: 堆叠成 (C, H, W) 格式，现在 C=2
        input_data_np = np.stack([re_data_resized, im_data_resized], axis=0)
        input_tensor = torch.from_numpy(input_data_np).float()

        if self.calculate_stats_mode:
            return {'image': input_tensor}

        if not self.label_re_paths or not self.label_im_paths:
            raise ValueError("在非统计模式下，标签路径未被正确初始化或标签文件列表为空。")

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

# --- 3. 模型定义 (Complex* 部分不变, DeepLabV3WithComplexNet2channeel 已修改) ---
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2d, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: CVTensor) -> CVTensor:
        real_out = self.real_conv(x.real) - self.imag_conv(x.imag)
        imag_out = self.real_conv(x.imag) + self.imag_conv(x.real)
        return CVTensor(real_out, imag_out)

class ComplexReLU(nn.Module):
    def forward(self, x: CVTensor) -> CVTensor:
        return CVTensor(F.relu(x.real), F.relu(x.imag))

class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = ComplexReLU()
        self.conv2 = ComplexConv2d(128, 256, kernel_size=2, stride=2)
        self.relu2 = ComplexReLU()
        self.conv3 = ComplexConv2d(256, 512, kernel_size=2, stride=2)
        self.relu3 = ComplexReLU()
        self.conv4 = ComplexConv2d(512, 1024, kernel_size=2, stride=2)
        self.relu4 = ComplexReLU()

    def forward(self, x: CVTensor) -> CVTensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DeepLabV3WithComplexNet2channeel(nn.Module):
    def __init__(self, output_channels=2):
        super(DeepLabV3WithComplexNet2channeel, self).__init__()
        
        full_deeplab_model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        
        # 路径1：DeepLabV3的骨干网络 + ASPP
        self.encoder_backbone = full_deeplab_model.backbone
        
        # --- 关键修改 ---
        # 预训练的ResNet backbone的第一个卷积层(conv1)是为3通道(RGB)图像设计的。
        # 我们需要将其替换为一个新的卷积层，以接受我们的2通道(Re, Im)输入。
        # 这会使conv1的权重从头开始训练，但骨干网络其余部分的预训练权重得以保留。
        original_conv1 = self.encoder_backbone.conv1
        self.encoder_backbone.conv1 = nn.Conv2d(
            in_channels=2, # 修改: 输入通道为 2
            out_channels=original_conv1.out_channels, # 保持输出通道数不变 (64)
            kernel_size=original_conv1.kernel_size,   # (7, 7)
            stride=original_conv1.stride,             # (2, 2)
            padding=original_conv1.padding,           # (3, 3)
            bias=original_conv1.bias                  # False
        )
        print("模型修改: DeepLab backbone的第一个卷积层已从3通道输入修改为2通道输入。")
        
        self.aspp_module = full_deeplab_model.classifier[0]

        # 路径2：您的ComplexNet
        self.complex_net = ComplexNet()

        # 特征融合模块定义 (这部分结构不变)
        aspp_output_channels = 256
        complex_net_real_output_channels = 1024 * 2
        concatenated_channels_for_fusion = aspp_output_channels + complex_net_real_output_channels
        
        self.fusion_conv = ConvBNReLU(concatenated_channels_for_fusion, 512, kernel_size=1, padding=0) 

        # 解码器部分 (这部分结构不变)
        self.decoder_in_channels = 512 
        self.decoder = nn.Sequential(
            ConvBNReLU(self.decoder_in_channels, 512, kernel_size=3, padding=1),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=1)
        )

    # 修改: forward函数的输入现在是一个2通道张量
    def forward(self, x_input_real_2chan: torch.Tensor) -> torch.Tensor: # 输入是 (B, 2, H_in, W_in)
        target_spatial_size_for_fusion = (64, 64) 

        # --- 路径1: DeepLabV3 Backbone + ASPP ---
        # 修改: 直接将2通道输入送入修改后的backbone
        features_from_backbone = self.encoder_backbone(x_input_real_2chan)['out'] 
        x1_from_aspp = self.aspp_module(features_from_backbone) 
        x1_processed = F.interpolate(x1_from_aspp, size=target_spatial_size_for_fusion, mode='bilinear', align_corners=False)

        # --- 路径2: ComplexNet ---
        # 修改: 从2通道输入中分离实部和虚部
        x_real_ch = x_input_real_2chan[:, 0, :, :].unsqueeze(1) # (B, 1, 512, 512)
        x_imag_ch = x_input_real_2chan[:, 1, :, :].unsqueeze(1) # (B, 1, 512, 512)
        x_complex_input = CVTensor(x_real_ch, x_imag_ch)
        
        x2_complex_out = self.complex_net(x_complex_input)
        x2_real_concatenated = torch.cat((x2_complex_out.real, x2_complex_out.imag), dim=1)

        # --- 特征融合 (逻辑不变) ---
        concatenated_features = torch.cat((x1_processed, x2_real_concatenated), dim=1) 
        fused_features = self.fusion_conv(concatenated_features) 
        
        # --- 解码器 (逻辑不变) ---
        decoded_output = self.decoder(fused_features)
        
        # 确保最终输出尺寸与标签一致
        if decoded_output.shape[-2:] != x_input_real_2chan.shape[-2:]:
             decoded_output = F.interpolate(decoded_output, size=x_input_real_2chan.shape[-2:], mode='bilinear', align_corners=False)
             
        return decoded_output

# --- 4. 训练逻辑 (采纳标准化框架) ---
def calculate_regression_metrics(outputs, labels):
    if outputs.shape[-2:] != labels.shape[-2:]:
        outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    mse = F.mse_loss(outputs, labels).item()
    rmse = math.sqrt(mse)
    return mse, rmse

if __name__ == '__main__':
    model_name = "ComplexNetDeepLab_2Channel_Reg" # 修改模型名称以反映变化
    checkpoint_dir = f"./{model_name}_Output"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_model_path = os.path.join(checkpoint_dir, f"{model_name}_latest_model.pth")
    best_model_path = os.path.join(checkpoint_dir, f"{model_name}_best_model.pth")
    log_path = os.path.join(checkpoint_dir, f"{model_name}_training_log.csv")

    # --- 预计算训练集的均值和标准差 ---
    print("✨ 正在计算训练集Re和Im通道的均值和标准差...")
    # 修改: stat_dataset 不再需要 f_fnames
    stat_dataset = MatrixDataset(
        re_fnames=re_train_files, im_fnames=im_train_files,
        label_re_fnames=label_re_train_files, label_im_fnames=label_im_train_files,
        calculate_stats_mode=True
    )
    stat_loader = DataLoader(stat_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 统计计算逻辑不变，因为它原本就只处理前两个通道
    re_sum, im_sum, re_sum_sq, im_sum_sq = 0.0, 0.0, 0.0, 0.0
    total_pixels_re, total_pixels_im = 0, 0
    for batch_data in tqdm(stat_loader, desc="计算统计量"):
        # batch_data['image'] 现在是 (B, 2, H, W)
        re_ch, im_ch = batch_data['image'][:, 0, :, :], batch_data['image'][:, 1, :, :]
        re_sum += torch.sum(re_ch).item(); im_sum += torch.sum(im_ch).item()
        re_sum_sq += torch.sum(torch.square(re_ch)).item(); im_sum_sq += torch.sum(torch.square(im_ch)).item()
        total_pixels_re += re_ch.nelement(); total_pixels_im += im_ch.nelement()
    
    if not total_pixels_re or not total_pixels_im:
        print("警告: 统计数据计算时未处理任何像素。Re/Im归一化将使用默认值(mean=0, std=1)。")
        re_m_train, re_s_train, im_m_train, im_s_train = 0.0, 1.0, 0.0, 1.0
    else:
        re_m_train = re_sum / total_pixels_re; im_m_train = im_sum / total_pixels_im
        re_s_train = math.sqrt(max(0, (re_sum_sq/total_pixels_re) - (re_m_train**2))); re_s_train = max(re_s_train, 1e-7)
        im_s_train = math.sqrt(max(0, (im_sum_sq/total_pixels_im) - (im_m_train**2))); im_s_train = max(im_s_train, 1e-7)
    print(f"统计结果: Re(均值={re_m_train:.4f}, 标准差={re_s_train:.4f}), Im(均值={im_m_train:.4f}, 标准差={im_s_train:.4f})")
    print("--- 统计数据计算完毕 ---")

    # --- 创建正式的数据集和加载器 ---
    # 修改: train_dataset 不再需要 f_fnames
    train_dataset = MatrixDataset(
        re_fnames=re_train_files, im_fnames=im_train_files,
        label_re_fnames=label_re_train_files, label_im_fnames=label_im_train_files,
        re_mean=re_m_train, re_std=re_s_train, im_mean=im_m_train, im_std=im_s_train,
        calculate_stats_mode=False
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    val_loader = None
    if re_val_files:
        # 修改: val_dataset 不再需要 f_fnames
        val_dataset = MatrixDataset(
            re_fnames=re_val_files, im_fnames=im_val_files,
            label_re_fnames=label_re_val_files, label_im_fnames=label_im_val_files,
            re_mean=re_m_train, re_std=re_s_train,
            im_mean=im_m_train, im_std=im_s_train,
            calculate_stats_mode=False
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    else:
        print("警告: 未找到验证文件，将跳过验证。早停策略将不可用。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLabV3WithComplexNet2channeel(output_channels=2).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0; best_val_loss = float('inf')
    train_losses_history, val_losses_history = [], []
    train_mses_history, train_rmses_history = [], []
    val_mses_history, val_rmses_history = [], []

    early_stopping_patience = 50
    early_stopping_counter = 0

    if os.path.exists(latest_model_path):
        print(f"🟡 正在从最新的{model_name}检查点恢复: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
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
            print(f"警告：加载{model_name}检查点时出错 ({e})。将尝试仅恢复模型权重和epoch。")
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = float('inf'); early_stopping_counter = 0
            train_losses_history, val_losses_history = [], []
            train_mses_history, train_rmses_history = [], []
            val_mses_history, val_rmses_history = [], []
            print(f"部分恢复：模型权重已加载，将从 epoch {start_epoch} 重新开始优化器和历史记录。")
    else:
        print(f"⚪ 未找到{model_name}检查点。将从头开始训练。")
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_mse', 'train_rmse', 'val_loss', 'val_mse', 'val_rmse'])
    
    print(f'设备: {device}')
    num_epochs = 1000

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss, epoch_train_mse, epoch_train_rmse = 0.0, 0.0, 0.0
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{model_name} 训练]', ncols=120, leave=False)
        
        for batch_data in progress_bar_train:
            inputs = batch_data['image'].to(device) # (B, 2, 512, 512)
            labels = batch_data['label'].to(device) # (B, 2, 512, 512)
            
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
            epoch_val_loss, epoch_val_mse, epoch_val_rmse = 0.0,0.0,0.0
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

            if not math.isnan(avg_epoch_val_loss):
                if avg_epoch_val_loss < best_val_loss:
                    best_val_loss = avg_epoch_val_loss
                    early_stopping_counter = 0 
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

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
            'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
            'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
            'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history,
            'early_stopping_counter': early_stopping_counter
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