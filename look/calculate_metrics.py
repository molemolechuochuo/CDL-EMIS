import os
import torch
import numpy as np
import pandas as pd
import cv2
import math
from tqdm import tqdm
from natsort import natsorted
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import glob # 确保在文件顶部导入glob库

# --- 1. 相对路径导入模型定义 ---
# 假设此脚本与 COMPARE 和 REMOTE 文件夹在同一级目录下
print("正在从 COMPARE 目录导入模型定义...")
try:
    from COMPARE.unet import UNet
    from COMPARE.FCN import FCN_ResNet50_Regression
    from COMPARE.segnet import SegNet
    from COMPARE.PSPNET import PSPNet
    from COMPARE.deeplab import DeepLabV3ForRegression
    from COMPARE.CDLmix import ResnetWithComplexNet
    from COMPARE.CDL_ASPP import DeepLabV3WithComplexNet
    # ⚠️ 请确保 two_channelCDL.py 文件中定义的类名是 DeepLabV3WithComplexNet2channeel
    from COMPARE.two_channelCDL import DeepLabV3WithComplexNet2channeel
    print("✅ 所有模型导入成功！")
except ImportError as e:
    print(f"❌ 模型导入失败: {e}")
    print("请确保此脚本位于项目根目录，且 COMPARE 文件夹包含所有必要的模型 .py 文件。")
    exit()


# --- 2. 配置区域 ---
# ⚠️ 数据根目录，请根据您的实际情况修改
DATA_ROOT_DIR = "E:/EMTdata" 
# ⚠️ 模型权重文件所在的根目录 (使用相对路径)
BASE_MODELS_DIR = "." 

# 定义需要评估的模型
# 键名应与 REMOTE 文件夹下的子文件夹前缀完全一致
MODEL_CLASS_MAP = {
    'DeepLabV3_Reg_UnifiedData': DeepLabV3ForRegression,
    'ComplexNetDeepLab_Reg': ResnetWithComplexNet,
    'ComplexNetDeepLab+aspp_Reg': DeepLabV3WithComplexNet,
    # 键名 '6vComplexNetDeepLab_2Channel_Reg' 来自您的截图，请确保它与文件夹名匹配
    '6vComplexNetDeepLab_2Channel_Reg': DeepLabV3WithComplexNet2channeel 
    # 如果您还有其他模型，请在这里添加，例如:
    # 'UNet_XLSX_Regression_Normalized': lambda: UNet(n_channels_in=3, n_channels_out=2),
    # 'FCN_XLSX_Regression': FCN_ResNet50_Regression,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4 # 使用稍大的batch size可以加速计算

# --- 3. 灵活的数据集定义 (可处理2通道或3通道) ---
def normalize_f_data(data, min_val=2.0, max_val=8.0):
    data = data.astype(np.float32)
    return (data - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else data - min_val

class MatrixDataset(Dataset):
    def __init__(self, re_paths, im_paths, f_paths, label_re_paths, label_im_paths,
                 re_mean, re_std, im_mean, im_std, include_f_channel=True):
        self.re_paths, self.im_paths, self.f_paths = re_paths, im_paths, f_paths
        self.label_re_paths, self.label_im_paths = label_re_paths, label_im_paths
        self.re_mean, self.re_std = re_mean, re_std
        self.im_mean, self.im_std = im_mean, im_std
        self.include_f_channel = include_f_channel
        self.target_size = (512, 512)

    def __len__(self):
        return len(self.re_paths)

    def __getitem__(self, idx):
        re_data = pd.read_excel(self.re_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        im_data = pd.read_excel(self.im_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
        
        re_data_resized = cv2.resize(re_data, self.target_size, interpolation=cv2.INTER_LINEAR)
        im_data_resized = cv2.resize(im_data, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 标准化
        if self.re_std > 1e-7: re_data_resized = (re_data_resized - self.re_mean) / self.re_std
        if self.im_std > 1e-7: im_data_resized = (im_data_resized - self.im_mean) / self.im_std

        channels = [re_data_resized, im_data_resized]
        if self.include_f_channel:
            f_data_orig = pd.read_excel(self.f_paths[idx], header=0, engine='openpyxl').values.astype(np.float32)
            f_data_resized = cv2.resize(normalize_f_data(f_data_orig), self.target_size, interpolation=cv2.INTER_LINEAR)
            channels.append(f_data_resized)
            
        input_tensor = torch.from_numpy(np.stack(channels, axis=0)).float()

        label_re_orig = pd.read_csv(self.label_re_paths[idx], header=None).values.astype(np.float32)
        label_im_orig = pd.read_csv(self.label_im_paths[idx], header=None).values.astype(np.float32)
        label_re_resized = cv2.resize(label_re_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_im_resized = cv2.resize(label_im_orig, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_tensor = torch.from_numpy(np.stack([label_re_resized, label_im_resized], axis=0)).float()

        return input_tensor, label_tensor

# --- 4. 数据加载函数 ---
def get_test_dataloader(data_folder, batch_size, include_f_channel=True):
    print(f"正在构建{'3通道' if include_f_channel else '2通道'}测试数据加载器...")
    
    # 构建路径
    base_input_path = os.path.join(data_folder, 'E') if os.path.exists(os.path.join(data_folder, 'E')) else data_folder
    base_label_path = os.path.join(data_folder, 'label') if os.path.exists(os.path.join(data_folder, 'label')) else data_folder
    
    re_folder_path = os.path.join(base_input_path, 'Re')
    im_folder_path = os.path.join(base_input_path, 'Im')
    f_folder_path = os.path.join(base_input_path, 'F')
    label_re_folder_path = os.path.join(base_label_path, 'label_Re')
    label_im_folder_path = os.path.join(base_label_path, 'label_Im')
    
    # 获取文件名
    re_files = natsorted(os.listdir(re_folder_path))
    im_files = natsorted(os.listdir(im_folder_path))
    f_files = natsorted(os.listdir(f_folder_path)) if include_f_channel else []
    label_re_files = natsorted(os.listdir(label_re_folder_path))
    label_im_files = natsorted(os.listdir(label_im_folder_path))

    # 同步文件列表
    file_lists = [re_files, im_files, label_re_files, label_im_files]
    if include_f_channel:
        file_lists.insert(2, f_files)
    min_len = min(len(lst) for lst in file_lists)
    
    file_tuples = list(zip(*(lst[:min_len] for lst in file_lists)))
    
    # 划分训练/测试集以确定测试文件和标准化参数
    train_files, test_files = train_test_split(file_tuples, test_size=0.2, random_state=42)
    
    # 计算训练集的标准化参数
    print("计算标准化参数...")
    re_sum, im_sum, re_sum_sq, im_sum_sq, total_pixels = 0.0, 0.0, 0.0, 0.0, 0
    for files in tqdm(train_files, desc="Stat Calc"):
        re_data = pd.read_excel(os.path.join(re_folder_path, files[0]), header=0, engine='openpyxl').values
        im_data = pd.read_excel(os.path.join(im_folder_path, files[1]), header=0, engine='openpyxl').values
        re_sum += np.sum(re_data); im_sum += np.sum(im_data)
        re_sum_sq += np.sum(np.square(re_data)); im_sum_sq += np.sum(np.square(im_data))
        total_pixels += re_data.size
    
    re_mean = re_sum / total_pixels
    im_mean = im_sum / total_pixels
    re_std = math.sqrt(max(0, (re_sum_sq / total_pixels) - (re_mean ** 2)))
    im_std = math.sqrt(max(0, (im_sum_sq / total_pixels) - (im_mean ** 2)))
    print(f"统计完成: Re(μ={re_mean:.4f}, σ={re_std:.4f}), Im(μ={im_mean:.4f}, σ={im_std:.4f})")

    # 创建测试数据集
    if include_f_channel:
        test_dataset = MatrixDataset(
            re_paths=[os.path.join(re_folder_path, f[0]) for f in test_files],
            im_paths=[os.path.join(im_folder_path, f[1]) for f in test_files],
            f_paths=[os.path.join(f_folder_path, f[2]) for f in test_files],
            label_re_paths=[os.path.join(label_re_folder_path, f[3]) for f in test_files],
            label_im_paths=[os.path.join(label_im_folder_path, f[4]) for f in test_files],
            re_mean=re_mean, re_std=re_std, im_mean=im_mean, im_std=im_std,
            include_f_channel=True
        )
    else: # 2通道
        test_dataset = MatrixDataset(
            re_paths=[os.path.join(re_folder_path, f[0]) for f in test_files],
            im_paths=[os.path.join(im_folder_path, f[1]) for f in test_files],
            f_paths=[],
            label_re_paths=[os.path.join(label_re_folder_path, f[2]) for f in test_files],
            label_im_paths=[os.path.join(label_im_folder_path, f[3]) for f in test_files],
            re_mean=re_mean, re_std=re_std, im_mean=im_mean, im_std=im_std,
            include_f_channel=False
        )
        
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# --- 5. 核心指标计算函数 ---
def calculate_all_metrics(model, dataloader, device):
    model.to(device)
    model.eval()
    
    all_psnr, all_ssim, all_mse, all_rmse = [], [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="正在评估"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 将Tensor转为Numpy数组以便计算
            outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(outputs_np.shape[0]): # 遍历batch中的每个样本
                true_img, pred_img = labels_np[i], outputs_np[i]
                
                # 计算 MSE, RMSE
                mse = np.mean((true_img - pred_img) ** 2)
                all_mse.append(mse)
                all_rmse.append(np.sqrt(mse))

                # 计算 PSNR, SSIM
                # data_range 是关键参数，代表图像的动态范围
                # 由于标签不是归一化到[0,1]的，我们使用实际的范围
                data_range = true_img.max() - true_img.min()
                if data_range == 0: continue # 避免除以零

                # 对实部和虚部分别计算，然后求平均
                psnr_re = psnr(true_img[0], pred_img[0], data_range=data_range)
                psnr_im = psnr(true_img[1], pred_img[1], data_range=data_range)
                all_psnr.append((psnr_re + psnr_im) / 2)

                ssim_re = ssim(true_img[0], pred_img[0], data_range=data_range)
                ssim_im = ssim(true_img[1], pred_img[1], data_range=data_range)
                all_ssim.append((ssim_re + ssim_im) / 2)

    return {
        "MSE": np.mean(all_mse),
        "RMSE": np.mean(all_rmse),
        "PSNR": np.mean(all_psnr),
        "SSIM": np.mean(all_ssim)
    }

if __name__ == "__main__":
    print("🚀 开始执行模型性能评估脚本...")
    print(f"将使用设备: {DEVICE}")

    # 1. 准备数据加载器
    loader_3_channel = get_test_dataloader(DATA_ROOT_DIR, BATCH_SIZE, include_f_channel=True)
    loader_2_channel = get_test_dataloader(DATA_ROOT_DIR, BATCH_SIZE, include_f_channel=False)
    
    results = {}

    # 2. 遍历所有模型进行评估
    for model_key, model_class in MODEL_CLASS_MAP.items():
        print(f"\n{'='*25}\n正在处理模型: {model_key}\n{'='*25}")
        
        # 实例化模型
        if "UNet" in model_key:
             model = model_class()
        else:
             model = model_class()

        # 根据模型名选择对应的数据加载器
        if "2Channel" in model_key:
            current_loader = loader_2_channel
            print("INFO: 检测到 '2Channel' 关键字，使用2通道数据加载器。")
        else:
            current_loader = loader_3_channel
            print("INFO: 使用默认的3通道数据加载器。")

        # --- ✨✨✨ 最终修正点：使用 glob 动态查找文件 ✨✨✨ ---
        
        # 1. 定义模型的主输出文件夹名
        model_folder_name = f"{model_key}_Output"
        if model_key == '6vComplexNetDeepLab_2Channel_Reg':
             model_folder_name = "6vComplexNetDeepLab_2Channel_Reg_Output"
        
        # 2. 构建模型输出的根目录
        model_output_root_dir = os.path.join(BASE_MODELS_DIR, model_folder_name)

        # 3. 使用 glob 递归搜索 *_best_model.pth 文件
        # `**` 表示搜索所有子目录
        search_pattern = os.path.join(model_output_root_dir, '**', '*_best_model.pth')
        print(f"正在搜索: {search_pattern}")
        pth_path_list = glob.glob(search_pattern, recursive=True)

        # 4. 检查搜索结果
        if not pth_path_list:
            print(f"⚠️ 警告: 在目录 {model_output_root_dir} 及其所有子目录中均未找到匹配 '*_best_model.pth' 的文件。")
            print("请检查您的 MODEL_CLASS_MAP 键名是否与文件夹名完全对应。跳过此模型。")
            continue
        
        # 如果找到多个，默认使用第一个
        pth_path = pth_path_list[0] 
        # --- 修正结束 ---

        try:
            print(f"✅ 找到权重文件: {pth_path}")
            print(f"正在加载...")
            checkpoint = torch.load(pth_path, map_location=DEVICE)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("权重加载成功。")

            # 计算指标
            metrics = calculate_all_metrics(model, current_loader, DEVICE)
            results[model_key] = metrics

        except Exception as e:
            print(f"❌ 处理模型 {model_key} 时发生严重错误: {e}")
            import traceback
            traceback.print_exc()

    # 3. 打印最终的总结表格
    print("\n\n" + "="*60)
    print("📊 所有模型评估结果汇总")
    print("="*60)
    
    if not results:
        print("没有成功评估任何模型。请检查模型键名和文件夹路径是否完全对应。")
    else:
        header = f"{'Model':<40} | {'MSE':<10} | {'RMSE':<10} | {'PSNR':<10} | {'SSIM':<10}"
        print(header)
        print("-" * len(header))
        for model_name, metrics in results.items():
            print(f"{model_name:<40} | {metrics['MSE']:<10.4f} | {metrics['RMSE']:<10.4f} | {metrics['PSNR']:<10.4f} | {metrics['SSIM']:<10.4f}")
    
    print("="*60)
    print("🏁 脚本执行完毕。")
