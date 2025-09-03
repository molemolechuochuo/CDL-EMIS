import os
import numpy as np
# from skimage.io import imread # 根据新的Dataset，可能不再需要
import pandas as pd # 用于读取xlsx和csv
import cv2          # 用于resize
# from torchvision import transforms # 不再需要
from torch.utils.data import DataLoader, Dataset
# from torchvision.models.segmentation import fcn_resnet50 # 我们使用自定义的SegNet
import torch
from datetime import datetime
import csv
import torch.nn as nn
from tqdm import tqdm
from natsort import natsorted # 用于自然排序
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score # 这些是分类指标
import time
from sklearn.model_selection import train_test_split

# --- 1. 设置文件路径 ---
# 设置基础路径
base_folder = r'E:\EMTdata'

# E文件夹下的路径 (输入)
E_folder = os.path.join(base_folder, 'E')
if not os.path.exists(E_folder): # 如果 E 子文件夹不存在，则假定 Re, Im, F 直接在 base_folder 下
    print(f"警告: 路径 {E_folder} 不存在，将尝试直接在 {base_folder} 下寻找 Re, Im, F 文件夹。")
    f_folder = os.path.join(base_folder, 'F')
    re_folder = os.path.join(base_folder, 'Re')
    im_folder = os.path.join(base_folder, 'Im')
else:
    f_folder = os.path.join(E_folder, 'F')
    re_folder = os.path.join(E_folder, 'Re')
    im_folder = os.path.join(E_folder, 'Im')


# label文件夹下的路径 (标签)
label_folder = os.path.join(base_folder, 'label')
if not os.path.exists(label_folder): # 如果 label 子文件夹不存在，则假定 label_Re, label_Im 直接在 base_folder 下
    print(f"警告: 路径 {label_folder} 不存在，将尝试直接在 {base_folder} 下寻找 label_Re, label_Im 文件夹。")
    label_re_folder = os.path.join(base_folder, 'label_Re')
    label_im_folder = os.path.join(base_folder, 'label_Im')
else:
    label_re_folder = os.path.join(label_folder, 'label_Re')
    label_im_folder = os.path.join(label_folder, 'label_Im')

print(f"输入 Re 文件夹: {re_folder}")
print(f"输入 Im 文件夹: {im_folder}")
print(f"输入 F 文件夹: {f_folder}")
print(f"标签 Re 文件夹: {label_re_folder}")
print(f"标签 Im 文件夹: {label_im_folder}")


# 检查文件夹是否存在
path_tuples = [
    ("输入Re", re_folder, '.xlsx'), ("输入Im", im_folder, '.xlsx'), ("输入F", f_folder, '.xlsx'),
    ("标签Re", label_re_folder, '.csv'), ("标签Im", label_im_folder, '.csv')
]
all_paths_valid = True
for name, path, ext in path_tuples:
    if not os.path.isdir(path):
        print(f"错误: 文件夹 '{name}' 在路径 '{path}' 未找到。请检查路径设置。")
        all_paths_valid = False
if not all_paths_valid:
    exit("程序因路径错误而终止。")


re_files = natsorted([f for f in os.listdir(re_folder) if f.lower().endswith('.xlsx')])
im_files = natsorted([f for f in os.listdir(im_folder) if f.lower().endswith('.xlsx')])
f_files = natsorted([f for f in os.listdir(f_folder) if f.lower().endswith('.xlsx')])
label_re_files = natsorted([f for f in os.listdir(label_re_folder) if f.lower().endswith('.csv')])
label_im_files = natsorted([f for f in os.listdir(label_im_folder) if f.lower().endswith('.csv')])

min_len = min(len(re_files), len(im_files), len(f_files), len(label_re_files), len(label_im_files))
if min_len == 0:
    raise ValueError(f"一个或多个数据/标签文件夹为空，或不包含预期的 .xlsx 或 .csv 文件。请检查文件：\n"
                     f"Re (.xlsx): {len(re_files)} 个文件\n"
                     f"Im (.xlsx): {len(im_files)} 个文件\n"
                     f"F (.xlsx): {len(f_files)} 个文件\n"
                     f"label_Re (.csv): {len(label_re_files)} 个文件\n"
                     f"label_Im (.csv): {len(label_im_files)} 个文件")

re_files = re_files[:min_len]
im_files = im_files[:min_len]
f_files = f_files[:min_len]
label_re_files = label_re_files[:min_len]
label_im_files = label_im_files[:min_len]

(re_train_files, re_test_files,
 im_train_files, im_test_files,
 f_train_files, f_test_files,
 label_re_train_files, label_re_test_files,
 label_im_train_files, label_im_test_files) = train_test_split(
    re_files, im_files, f_files, label_re_files, label_im_files,
    test_size=0.2, random_state=42
)

# --- 2. 数据集定义 ---
# 频率通道归一化函数 (根据您的参考脚本)
def normalize_f_data(data, min_val=2.0, max_val=8.0):
    """将频率数据归一化到[0,1]范围 (假设原始值在[min_val, max_val]之间)"""
    # 确保data是浮点数以进行计算
    data = data.astype(np.float32)
    # 防止除以零
    if (max_val - min_val) == 0:
        return data - min_val # 或者返回全零，取决于具体期望
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

class MatrixDataset(Dataset): # 采用您提供的 MatrixDataset
    def __init__(self, re_paths_list, im_paths_list, f_paths_list, label_re_paths_list, label_im_paths_list):
        self.re_paths = [os.path.join(re_folder, f) for f in re_paths_list] # 构建完整路径
        self.im_paths = [os.path.join(im_folder, f) for f in im_paths_list]
        self.f_paths = [os.path.join(f_folder, f) for f in f_paths_list]
        self.label_re_paths = [os.path.join(label_re_folder, f) for f in label_re_paths_list]
        self.label_im_paths = [os.path.join(label_im_folder, f) for f in label_im_paths_list]
        self.target_size = (512, 512)

    def __len__(self):
        return len(self.re_paths)

    def __getitem__(self, idx):
        # 读取数据
        try:
            re_data = pd.read_excel(self.re_paths[idx], header=0, engine='openpyxl').values
            im_data = pd.read_excel(self.im_paths[idx], header=0, engine='openpyxl').values
            f_data_orig = pd.read_excel(self.f_paths[idx], header=0, engine='openpyxl').values
        except Exception as e:
            print(f"读取输入文件 (xlsx) 时出错: index {idx}")
            print(f"Re path: {self.re_paths[idx]}")
            print(f"Im path: {self.im_paths[idx]}")
            print(f"F path: {self.f_paths[idx]}")
            print(f"错误: {e}")
            raise

        f_data_normalized = normalize_f_data(f_data_orig) # 归一化F通道

        # resize 到 (512, 512)
        # pandas读取的.values可能是object类型或整数，cv2.resize需要float32或uint8
        re_data_resized = cv2.resize(re_data.astype(np.float32), self.target_size, interpolation=cv2.INTER_LINEAR)
        im_data_resized = cv2.resize(im_data.astype(np.float32), self.target_size, interpolation=cv2.INTER_LINEAR)
        f_data_resized = cv2.resize(f_data_normalized.astype(np.float32), self.target_size, interpolation=cv2.INTER_LINEAR)

        # 堆叠输入数据 (C, H, W) -> (3, 512, 512)
        input_data_np = np.stack([re_data_resized, im_data_resized, f_data_resized], axis=0)
        input_tensor = torch.from_numpy(input_data_np).float()

        # 读取并 resize 标签
        try:
            label_re_orig = pd.read_csv(self.label_re_paths[idx], header=None).values
            label_im_orig = pd.read_csv(self.label_im_paths[idx], header=None).values
        except Exception as e:
            print(f"读取标签文件 (csv) 时出错: index {idx}")
            print(f"Label Re path: {self.label_re_paths[idx]}")
            print(f"Label Im path: {self.label_im_paths[idx]}")
            print(f"错误: {e}")
            raise

        label_re_resized = cv2.resize(label_re_orig.astype(np.float32), self.target_size, interpolation=cv2.INTER_LINEAR)
        label_im_resized = cv2.resize(label_im_orig.astype(np.float32), self.target_size, interpolation=cv2.INTER_LINEAR)

        # 堆叠标签数据 (C, H, W) -> (2, 512, 512)
        label_data_np = np.stack([label_re_resized, label_im_resized], axis=0)
        label_tensor = torch.from_numpy(label_data_np).float()

        return {'image': input_tensor, 'label': label_tensor}

# --- 3. SegNet 模型定义 ---
class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.dec4 = self.upconv_block(512, 256)
        self.dec3 = self.upconv_block(256, 128)
        self.dec2 = self.upconv_block(128, 64)
        self.final_dec_layer = nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x): # x 预期尺寸 (B, C_in, 512, 512)
        original_size = x.shape[2:]

        x1_conv = self.enc1(x)
        x1_pool, idx1 = F.max_pool2d(x1_conv, kernel_size=2, stride=2, return_indices=True)
        x2_conv = self.enc2(x1_pool)
        x2_pool, idx2 = F.max_pool2d(x2_conv, kernel_size=2, stride=2, return_indices=True)
        x3_conv = self.enc3(x2_pool)
        x3_pool, idx3 = F.max_pool2d(x3_conv, kernel_size=2, stride=2, return_indices=True)
        x4_conv = self.enc4(x3_pool)
        x4_pool, idx4 = F.max_pool2d(x4_conv, kernel_size=2, stride=2, return_indices=True)

        d4_unpool = F.max_unpool2d(x4_pool, idx4, kernel_size=2, stride=2, output_size=x4_conv.size())
        d4 = self.dec4(d4_unpool)
        d3_unpool = F.max_unpool2d(d4, idx3, kernel_size=2, stride=2, output_size=x3_conv.size())
        d3 = self.dec3(d3_unpool)
        d2_unpool = F.max_unpool2d(d3, idx2, kernel_size=2, stride=2, output_size=x2_conv.size())
        d2 = self.dec2(d2_unpool)
        out_unpool_final = F.max_unpool2d(d2, idx1, kernel_size=2, stride=2, output_size=x1_conv.size())
        out = self.final_dec_layer(out_unpool_final)

        if out.shape[2:] != original_size: # 确保输出尺寸与固定输入尺寸(512,512)一致
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=True)
        return out

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# --- 4. 训练逻辑 ---
# 指标计算函数 (修正版)
def calculate_regression_metrics(outputs, labels):
    mse = F.mse_loss(outputs, labels).item()
    # RMSE 是 MSE 的平方根
    rmse = torch.sqrt(torch.tensor(mse)).item() # 或者直接 math.sqrt(mse)
    return mse, rmse

if __name__ == '__main__':
    # 定义模型和日志的保存路径 (使用 "SegNet" 标识)
    checkpoint_dir = r"./SegNet_XLSX_Output_final" # 可以根据需要更改此输出目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_model_path = os.path.join(checkpoint_dir, "segnet_latest_model.pth")
    best_model_path = os.path.join(checkpoint_dir, "segnet_best_model.pth")
    log_path = os.path.join(checkpoint_dir, "segnet_training_log.csv")

    # 数据集实例化
    train_dataset = MatrixDataset(
        re_paths_list=re_train_files, im_paths_list=im_train_files, f_paths_list=f_train_files,
        label_re_paths_list=label_re_train_files, label_im_paths_list=label_im_train_files
    )
    # batch_size 和 num_workers 根据您的硬件进行调整
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    if re_test_files: # 仅当存在测试文件时创建测试加载器
        test_dataset = MatrixDataset(
            re_paths_list=re_test_files, im_paths_list=im_test_files, f_paths_list=f_test_files,
            label_re_paths_list=label_re_test_files, label_im_paths_list=label_im_test_files
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    else:
        test_loader = None
        print("警告: 未找到测试文件 (re_test_files 为空)，将跳过验证过程。")

    # 模型、损失函数、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegNet(input_channels=3, output_channels=2).to(device) # 3通道输入, 2通道输出
    criterion = nn.MSELoss() # 均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # 学习率

    # 训练状态变量
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses_history, val_losses_history = [], [] # 使用更明确的变量名
    train_mses_history, train_rmses_history = [], []
    val_mses_history, val_rmses_history = [], []

    # ssh断联了！加载检查点
    if os.path.exists(latest_model_path):
        print(f"🟡 正在从最新的检查点恢复: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1 # .get 提供默认值
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            train_losses_history = checkpoint.get('train_losses_history', [])
            val_losses_history = checkpoint.get('val_losses_history', [])
            train_mses_history = checkpoint.get('train_mses_history', [])
            train_rmses_history = checkpoint.get('train_rmses_history', [])
            val_mses_history = checkpoint.get('val_mses_history', [])
            val_rmses_history = checkpoint.get('val_rmses_history', [])
            print(f"✅ 模型加载完毕。将从 epoch {start_epoch} 开始继续训练。当前最佳验证Loss: {best_val_loss:.6f}")
        except Exception as e: # 更通用的异常捕获
            print(f"警告：加载优化器或历史指标时出错 ({e})。部分状态可能未恢复。")
            # 可以选择重置这些状态或尝试部分恢复
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf')) # 至少尝试恢复这个
    else:
        print("⚪ 未找到检查点。将从头开始训练。")
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_mse', 'train_rmse', 'val_loss', 'val_mse', 'val_rmse'])

    print(f'训练初始化完成。设备: {device}')
    num_epochs = 100 # 训练轮次 (根据需要调整)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_mse = 0.0
        epoch_train_rmse = 0.0

        progress_bar_train = tqdm(train_loader,
                                  desc=f'Epoch {epoch+1}/{num_epochs} [训练]',
                                  ncols=120,     # 进度条宽度，可调整
                                  miniters=10,   # <-- 核心改动：每处理10个批次（迭代）才刷新一次显示
                                  mininterval=0, # <-- 确保更新主要由 miniters 控制 (0表示尽可能快，但会被miniters限制)
                                  leave=True)    # 循环结束后保留进度条状态
        for batch_idx,batch_data in enumerate(progress_bar_train):
            inputs = batch_data['image'].to(device)
            labels = batch_data['label'].to(device) # 使用 'label' 键

            optimizer.zero_grad()
            outputs = model(inputs) # SegNet直接输出张量
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            mse, rmse = calculate_regression_metrics(outputs, labels)
            epoch_train_loss += loss.item()
            epoch_train_mse += mse
            epoch_train_rmse += rmse

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader): # <--- 修改2：添加 if 条件
                progress_bar_train.set_postfix(loss=loss.item(), mse=mse, rmse=rmse)


        avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_epoch_train_mse = epoch_train_mse / len(train_loader) if len(train_loader) > 0 else 0
        avg_epoch_train_rmse = epoch_train_rmse / len(train_loader) if len(train_loader) > 0 else 0

        train_losses_history.append(avg_epoch_train_loss)
        train_mses_history.append(avg_epoch_train_mse)
        train_rmses_history.append(avg_epoch_train_rmse)
        print(f"Epoch {epoch+1} 训练总结: Loss: {avg_epoch_train_loss:.4f}, MSE: {avg_epoch_train_mse:.4f}, RMSE: {avg_epoch_train_rmse:.4f}")

        # 验证
        avg_epoch_val_loss = None
        if test_loader:
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_mse = 0.0
            epoch_val_rmse = 0.0
            with torch.no_grad():
                progress_bar_val = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [验证]', ncols=110)
                for batch_data in progress_bar_val:
                    inputs = batch_data['image'].to(device)
                    labels = batch_data['label'].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    mse, rmse = calculate_regression_metrics(outputs, labels)
                    epoch_val_loss += loss.item()
                    epoch_val_mse += mse
                    epoch_val_rmse += rmse
                    progress_bar_val.set_postfix(loss=loss.item(), mse=mse, rmse=rmse)

            avg_epoch_val_loss = epoch_val_loss / len(test_loader) if len(test_loader) > 0 else 0
            avg_epoch_val_mse = epoch_val_mse / len(test_loader) if len(test_loader) > 0 else 0
            avg_epoch_val_rmse = epoch_val_rmse / len(test_loader) if len(test_loader) > 0 else 0

            val_losses_history.append(avg_epoch_val_loss)
            val_mses_history.append(avg_epoch_val_mse)
            val_rmses_history.append(avg_epoch_val_rmse)
            print(f"Epoch {epoch+1} 验证总结: Loss: {avg_epoch_val_loss:.4f}, MSE: {avg_epoch_val_mse:.4f}, RMSE: {avg_epoch_val_rmse:.4f}")

            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                    'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
                    'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
                    'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history
                }, best_model_path)
                print(f"✅ 最佳模型已保存到 {best_model_path}，验证 Loss: {best_val_loss:.6f}")
        else: # 如果没有验证加载器
            val_losses_history.append(float('nan')) # 使用 NaN 作为占位符
            val_mses_history.append(float('nan'))
            val_rmses_history.append(float('nan'))


        # 保存最新模型
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
            'train_losses_history': train_losses_history, 'val_losses_history': val_losses_history,
            'train_mses_history': train_mses_history, 'train_rmses_history': train_rmses_history,
            'val_mses_history': val_mses_history, 'val_rmses_history': val_rmses_history
        }, latest_model_path)
        # print(f"☑️ 最新模型检查点已保存到 {latest_model_path}") # 减少打印频率

        # 记录日志
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            val_loss_log = f"{avg_epoch_val_loss:.4f}" if avg_epoch_val_loss is not None else "N/A"
            val_mse_log = f"{avg_epoch_val_mse:.4f}" if avg_epoch_val_loss is not None else "N/A" # 基于avg_epoch_val_loss判断是否有验证
            val_rmse_log = f"{avg_epoch_val_rmse:.4f}" if avg_epoch_val_loss is not None else "N/A"
            writer.writerow([epoch + 1, avg_epoch_train_loss, avg_epoch_train_mse, avg_epoch_train_rmse,
                             val_loss_log, val_mse_log, val_rmse_log])
    print('🏁 训练完成。')

    # 绘制图表
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses_history, label='训练损失')
    plt.plot(val_losses_history, label='验证损失', linestyle='--')
    plt.xlabel('轮次 (Epoch)'); plt.ylabel('损失 (Loss)'); plt.legend(); plt.title('损失变化')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_mses_history, label='训练 MSE')
    plt.plot(val_mses_history, label='验证 MSE', linestyle='--')
    plt.xlabel('轮次 (Epoch)'); plt.ylabel('MSE'); plt.legend(); plt.title('均方误差 (MSE)')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(train_rmses_history, label='训练 RMSE')
    plt.plot(val_rmses_history, label='验证 RMSE', linestyle='--')
    plt.xlabel('轮次 (Epoch)'); plt.ylabel('RMSE'); plt.legend(); plt.title('均方根误差 (RMSE)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "training_metrics_segnet_xlsx_zh.png"))
    print(f"📈 训练指标图表已保存到 {os.path.join(checkpoint_dir, 'training_metrics_segnet_xlsx_zh.png')}")