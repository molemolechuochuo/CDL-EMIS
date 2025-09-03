import os
import glob
import pandas as pd

def generate_comparison_table(base_dir, key_metric='val_loss', higher_is_better=False):
    """
    遍历指定目录下的所有模型输出文件夹，读取CSV日志，
    根据指定的关键指标找到最佳epoch的数据，并生成一个性能对比表格。

    Args:
        base_dir (str): 包含所有模型输出文件夹的根目录。
                        例如："E:/EMMESData/Remote/"
        key_metric (str): 用于判断最佳epoch的列名。
                          根据你的截图，默认为 'val_loss'。
        higher_is_better (bool): 'key_metric'的值是否越高越好。
                                 对于 'val_loss' 或 'val_rmse'，应为 False。
                                 如果用 'val_psnr'，则为 True。
    
    Returns:
        pandas.DataFrame: 包含所有模型最佳性能的对比表格。
    """
    print("🚀 Running Quantitative Analysis...")
    print(f"🔍 Finding best epoch based on '{'highest' if higher_is_better else 'lowest'}' value of '{key_metric}'.")
    
    # 自动查找BASE_DIR下所有的子文件夹
    model_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    if not model_folders:
        print(f"❌ No model folders found in '{base_dir}'. Please check the path.")
        return None

    all_models_best_metrics = []

    # 遍历每个模型文件夹
    for folder in model_folders:
        folder_path = os.path.join(base_dir, folder)
        
        try:
            # 使用glob找到唯一的CSV日志文件，避免硬编码文件名
            csv_path_list = glob.glob(os.path.join(folder_path, '*_training_log.csv'))
            if not csv_path_list:
                print(f"⚠️ Warning: No '*_training_log.csv' file found in '{folder}'. Skipping.")
                continue
            
            csv_path = csv_path_list[0]
            log_df = pd.read_csv(csv_path)

            # --- 核心逻辑：找到最佳epoch ---
            if higher_is_better:
                # 找到key_metric值最大的那一行
                best_epoch_row = log_df.loc[log_df[key_metric].idxmax()]
            else:
                # 找到key_metric值最小的那一行
                best_epoch_row = log_df.loc[log_df[key_metric].idxmin()]
            
            model_name = folder.replace('_Output', '').replace('_final', '')
            
            # 从最佳行中提取所有你需要的指标
            metrics = {
                'Model': model_name,
                'Best Epoch': int(best_epoch_row['epoch']),
                'val_loss': best_epoch_row['val_loss'],
                'val_mse': best_epoch_row['val_mse'],
                'val_rmse': best_epoch_row['val_rmse'],
                # 你也可以提取训练集指标作为参考
                'train_loss': best_epoch_row['train_loss'],
                'train_rmse': best_epoch_row['train_rmse']

            }
            all_models_best_metrics.append(metrics)
            print(f"✅ Processed: {model_name} (Best epoch: {int(best_epoch_row['epoch'])})")

        except Exception as e:
            print(f"❌ Error processing folder '{folder}': {e}")

    if not all_models_best_metrics:
        print("No models were successfully processed.")
        return None

    # 创建最终的DataFrame并进行排序和打印
    results_df = pd.DataFrame(all_models_best_metrics)
    
    # 按关键指标排序，方便查看最优模型
    results_df = results_df.sort_values(by=key_metric, ascending=not higher_is_better)
    
    print("\n" + "="*80)
    print("📊 Quantitative Comparison Table (Best Epoch Results)")
    print("="*80)
    # 使用to_string()来保证所有列都能被打印出来
    print(results_df.to_string(index=False))
    
    # (可选) 将结果保存到CSV文件，方便在论文中引用
    output_filename = "model_comparison_best_epochs.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\n✅ Results saved to '{output_filename}'")
    
    return results_df

# ==============================================================================
# --- 请在这里修改你的项目路径 ---
# ==============================================================================
if __name__ == "__main__":
    # ⚠️【请在此处修改】你的数据根目录
    YOUR_PROJECT_BASE_DIR = "E:/EMMESData/Remote/" 
    
    # 运行分析函数
    generate_comparison_table(base_dir=YOUR_PROJECT_BASE_DIR, 
                              key_metric='val_loss', # 你可以用 'val_rmse' 等其他指标
                              higher_is_better=False)