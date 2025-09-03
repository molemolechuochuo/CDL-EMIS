import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torchvision import models
from sklearn.model_selection import train_test_split
from natsort import natsorted
# import complextorch as cvtorch
# import complextorch.nn as cvnn
from complextorch import CVTensor
# import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
import csv

# 设置文件路径
data_folder = r'E:\CellSegnetTset\U-net-master\dataset6_18_8电场\矩阵'
re_folder = os.path.join(data_folder, 'Re')
im_folder = os.path.join(data_folder, 'Im')
F_folder = os.path.join(data_folder, 'F')  # 添加频率数据文件夹路径
label_re_folder = os.path.join(data_folder, 'label_Re')
label_im_folder = os.path.join(data_folder, 'label_Im')

re_files = natsorted(os.listdir(re_folder))
im_files = natsorted(os.listdir(im_folder))
F_files = natsorted(os.listdir(F_folder))  # 获取频率数据文件列表
label_re_files = natsorted(os.listdir(label_re_folder))
label_im_files = natsorted(os.listdir(label_im_folder))

# 按80/20的比例划分训练集和测试集
re_train_files, re_test_files, im_train_files, im_test_files, F_train_files, F_test_files, label_re_train_files, label_re_test_files, label_im_train_files, label_im_test_files = train_test_split(
    re_files, im_files, F_files, label_re_files, label_im_files, test_size=0.2, random_state=42
)

# 定义归一化函数
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

def normalize_f_data(data, min_val=2.0, max_val=8.0):
    return (data - min_val) / (max_val - min_val)

class CustomDataset(Dataset):
    def __init__(self, re_paths, im_paths, F_paths, label_re_paths, label_im_paths, transform=None):
        self.re_paths = re_paths
        self.im_paths = im_paths
        self.F_paths = F_paths
        self.label_re_paths = label_re_paths
        self.label_im_paths = label_im_paths
        self.transform = transform

    def __len__(self):
        return len(self.re_paths)

    def __getitem__(self, idx):
        re_path = self.re_paths[idx]
        im_path = self.im_paths[idx]
        F_path = self.F_paths[idx]
        label_re_path = self.label_re_paths[idx]
        label_im_path = self.label_im_paths[idx]

        # 读取数据
        re_data = pd.read_excel(re_path, header=0, engine='openpyxl').values
        im_data = pd.read_excel(im_path, header=0, engine='openpyxl').values
        F_data = pd.read_excel(F_path, header=0, engine='openpyxl').values  # 读取频率数据
        label_re_data = pd.read_csv(label_re_path, header=None).values
        label_im_data = pd.read_csv(label_im_path, header=None).values

        # 将数据转换为浮点数
        re_data = normalize(re_data.astype(np.float32))  # 普通归一化
        im_data = normalize(im_data.astype(np.float32))  # 普通归一化
        F_data = normalize_f_data(F_data.astype(np.float32))  # 归一化频率数据
        label_re_data = label_re_data.astype(np.float32)
        label_im_data = label_im_data.astype(np.float32)

        # 调整大小为 512x512，并添加频率维度
        re_data = torch.nn.functional.interpolate(torch.from_numpy(np.expand_dims(re_data, axis=0)).unsqueeze(0), size=(512,512), mode='bilinear', align_corners=True).squeeze(0)
        im_data = torch.nn.functional.interpolate(torch.from_numpy(np.expand_dims(im_data, axis=0)).unsqueeze(0), size=(512,512), mode='bilinear', align_corners=True).squeeze(0)
        F_data = torch.nn.functional.interpolate(torch.from_numpy(np.expand_dims(F_data, axis=0)).unsqueeze(0), size=(512,512), mode='bilinear', align_corners=True).squeeze(0)

        label_re_data = torch.nn.functional.interpolate(torch.from_numpy(np.expand_dims(label_re_data, axis=0)).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=True).squeeze(0)
        label_im_data = torch.nn.functional.interpolate(torch.from_numpy(np.expand_dims(label_im_data, axis=0)).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=True).squeeze(0)

        # 将实部和虚部结合为普通张量，并添加频率数据
        image = torch.cat((re_data, im_data, F_data), dim=0)  # 形状为 (3, 512, 512)
        label = torch.cat((label_re_data, label_im_data), dim=0)  # 形状为 (2, 512, 512) 

        return {'image': image, 'label': label}
    
def calculate_metrics(outputs, labels):
    mse = F.mse_loss(outputs, labels).item()
    rmse = torch.sqrt(F.mse_loss(outputs, labels)).item()
    return mse, rmse

# 定义复数卷积层和激活函数
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0):
        super(ComplexConv2d, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        real = self.real_conv(x.real) - self.imag_conv(x.imag)
        imag = self.real_conv(x.imag) + self.imag_conv(x.real)
        return CVTensor(real, imag)

class ComplexReLU(nn.Module):
    def forward(self, x):
        real = torch.relu(x.real)
        imag = torch.relu(x.imag)
        return CVTensor(real, imag)

# 定义复数网络
class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 128, kernel_size=3 ,stride=1, padding=1)
        self.relu1 = ComplexReLU()
        self.conv2 = ComplexConv2d(128, 256, kernel_size=2)
        self.relu2 = ComplexReLU()
        self.conv3 = ComplexConv2d(256, 512, kernel_size=2)
        self.relu3 = ComplexReLU()
        self.conv4 = ComplexConv2d(512, 1024, kernel_size=2)
        self.relu4 = ComplexReLU()

    def forward(self, x):
        # torch.Size([2, 1, 512, 512])  
        x = self.relu1(self.conv1(x)) #torch.Size([2, 16, 512, 512])
        x = self.relu2(self.conv2(x)) #torch.Size([2, 16, 512, 512])
        x = self.relu3(self.conv3(x)) #torch.Size([2, 64, 512, 512])
        x = self.relu4(self.conv4(x)) #torch.Size([2, 128, 512, 512])
        return x

# 定义 DeepLabV3 模型并结合复数网络
class DeepLabV3WithComplexNet(nn.Module):
    def __init__(self):
        super(DeepLabV3WithComplexNet, self).__init__()

        # DeepLabV3 的编码器部分
        self.encoder = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.encoder.classifier = nn.Identity()  # 移除 DeepLabV3 的分类头

        # 复数网络
        self.complex_net = ComplexNet()

        # Decoder 部分
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),  # 输入通道数改为 2048 + 1024
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 2, kernel_size=1)  # 最后输出2个通道
        )

    def forward(self, x):
        # 将输入数据分离成实部、虚部和频率部分
        # torch.Size([2, 3, 512, 512])
        x_real = x[:, 0, :, :]
        x_imag = x[:, 1, :, :]
        x_freq = x[:, 2, :, :]  # 添加一个维度以匹配卷积层的输入形状
        x_complex = CVTensor(x_real, x_imag)

        # 将所有通道组合并传递给 DeepLabV3 的编码器部分
        x_combined = torch.cat((x_real.unsqueeze(1), x_imag.unsqueeze(1), x_freq.unsqueeze(1)), dim=1)
        #torch.Size([2, 3, 512, 512]) 送进Encoder的
        # 组合的数据进入编码器
        x1 = self.encoder.backbone(x_combined)['out'] 
        # torch.Size([2, 2048, 64, 64]) #出Encoder的
        
        x_complex = CVTensor(x_real.squeeze(1), x_imag.squeeze(1))  # 恢复为 [batch_size, height, width]

        # 手动添加一个维度以匹配复数网络的输入要求
        x_complex.real = x_complex.real.unsqueeze(1)  # [batch_size, 1, height, width]
        x_complex.imag = x_complex.imag.unsqueeze(1)  # [batch_size, 1, height, width]
        # 转置操作，将形状从 [batch_size, height, width] 转换为 [1, batch_size, height, width]
        x_complex.real = x_complex.real.permute(0, 1, 2, 3)
        x_complex.imag = x_complex.imag.permute(0, 1, 2, 3)

        # 调整大小以匹配编码器输出
        x2 = self.complex_net(x_complex)
        # torch.Size([2, 512, 512, 512])
        x2 = torch.cat((x2.real, x2.imag), dim=1)  # [batch_size, 2 * channels, height, width]
        # torch.Size([2, 1024, 512, 512])
        # x2 = nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        # torch.Size([2, 1024, 64, 64])


        x = x1+x2
        # x = torch.cat((x1, x2), dim=1)
        # torch.Size([2, 3072, 64, 64])
        x = self.decoder(x)
        return x

if __name__ == '__main__':

    # 获取数据集
    train_dataset = CustomDataset(
        re_paths=[os.path.join(re_folder, f) for f in re_train_files],
        im_paths=[os.path.join(im_folder, f) for f in im_train_files],
        F_paths=[os.path.join(F_folder, f) for f in F_train_files],  # 传递频率数据路径
        label_re_paths=[os.path.join(label_re_folder, f) for f in label_re_train_files],
        label_im_paths=[os.path.join(label_im_folder, f) for f in label_im_train_files]
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = CustomDataset(
        re_paths=[os.path.join(re_folder, f) for f in re_test_files],
        im_paths=[os.path.join(im_folder, f) for f in im_test_files],
        F_paths=[os.path.join(F_folder, f) for f in F_test_files],  # 传递频率数据路径
        label_re_paths=[os.path.join(label_re_folder, f) for f in label_re_test_files],
        label_im_paths=[os.path.join(label_im_folder, f) for f in label_im_test_files]
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # 定义损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLabV3WithComplexNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
    save_path = "./best_model.pth"
    best_val_loss = float('inf') 

  # 训练过程
    num_epochs = 1000
    train_losses, val_losses = [], []
    train_mse, train_rmse, val_mse, val_rmse = [], [], [], []

    print('训练初始化完成')

    # ssh突然断联了！！！
    # 如果有保存的模型，就加载
    resume_path = r"D:\workspace\liyuzhe\best_model.pth"
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(resume_path):
        print("🟡 检测到已有模型 checkpoint，正在加载...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"✅ 模型加载完毕，将从 epoch {start_epoch} 开始继续训练，当前最优验证 Loss：{best_val_loss:.6f}")


    # 计算并打印 MSE 和 RMSE
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_rmse = 0.0
        train_steps = 0
        print(f"正在进行第{epoch}epoch训练")

        for step, batch in enumerate(tqdm(train_loader,desc=f'epoch{epoch}',ncols=130)):
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            mse, rmse = calculate_metrics(outputs, labels)
            running_loss += loss.item()
            running_mse += mse
            running_rmse += rmse
            train_steps += 1

            # 打印每一步的信息
            if (step + 1) % 10 == 0:  # 每10个批次打印一次
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] [Epoch {epoch+1}/{num_epochs}] Step {step+1}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")


        train_loss = running_loss / train_steps
        train_losses.append(train_loss)
        train_mse_val = running_mse / train_steps
        train_rmse_val = running_rmse / train_steps
        train_mse.append(train_mse_val)
        train_rmse.append(train_rmse_val)

        # 验证过程
        model.eval()
        val_loss = 0.0
        val_mse_epoch = 0.0
        val_rmse_epoch = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                mse, rmse = calculate_metrics(outputs, labels)
                val_mse_epoch += mse
                val_rmse_epoch += rmse
                val_steps += 1

        val_loss /= val_steps
        val_losses.append(val_loss)
        val_mse_val = val_mse_epoch / val_steps
        val_rmse_val = val_rmse_epoch / val_steps
        val_mse.append(val_mse_val)
        val_rmse.append(val_rmse_val)

            # 如果当前验证集上的 loss 更小，就保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, save_path)
            print(f"✅ 模型已保存，当前最优验证 Loss：{best_val_loss:.6f}")

        #保存日志
        log_path = "training_log.csv"
        # 如果文件不存在，先写表头
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_mse', 'train_rmse', 'val_loss', 'val_mse', 'val_rmse'])
        # 在每个 epoch 结束时保存
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_mse_val, train_rmse_val, val_loss, val_mse_val, val_rmse_val])