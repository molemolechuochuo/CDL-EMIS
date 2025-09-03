# CDL-Net: Dual-Branch Complex-Valued Networks for Microwave Complex Permittivity Inversion

<img width="13170" height="7398" alt="CDL_fram" src="https://github.com/user-attachments/assets/29257f69-d9ef-49d4-85c8-b44476bf43df" />

## 摘要 (Abstract)

Microwave imaging (MWI) is a promising non-invasive technique for quantitatively reconstructing the complex permittivity of biological tissues. However, its clinical application is often limited by the non-linear and ill-posed nature of the electromagnetic inverse scattering problem. To address this challenge, we propose CDL-Net, a deep learning-based dual-branch network. The real-valued branch, leveraging a ResNet50 backbone and an Atrous Spatial Pyramid Pooling (ASPP) module, extracts multi-scale semantic features. Concurrently, the complex-valued branch processes phase information directly, preserving crucial physical properties consistent with wave propagation in lossy media. Our experiments, conducted on a realistic 3D human thigh phantom, demonstrate that CDL-Net achieves state-of-the-art performance, yielding a PSNR of 48.331 dB and an SSIM of 0.997.

## 项目结构 (Project Structure)

```
.
├── COMPARE/              # 包含所有模型训练和对比实验的Python脚本
│   ├── CDL_ASPP.py
│   ├── CDLmix.py
│   └── ...
├── CST/                  # CST电磁仿真工程文件
├── EMTData/              # 仿真的原始数据集
├── ComplexNetDeepLab_Reg_Output/ # 本项目(CDL-Net)的输出，包括模型权重
├── ...                   # 其他基线模型的输出文件夹
├── .gitattributes        # Git LFS 配置文件
├── requirements.txt      # 运行环境
└── README.md             # 本说明文件
```

## 环境配置 (Setup)

1.  **克隆本仓库**
    ```bash
    git clone [https://github.com/molemolechuchuo/Deeplab_complexnet.git](https://github.com/molemolechuchuo/Deeplab_complexnet.git)
    cd Deeplab_complexnet
    ```

2.  **创建虚拟环境并安装依赖** (推荐使用 Conda)
    ```bash
    conda create -n cdlnet python=3.9
    conda activate cdlnet
    pip install -r requirements.txt 
    ```

3.  **数据集**
    本项目使用的数据集由CST Studio Suite仿真生成，存放于EMTData文件夹。请确保数据路径与脚本中的配置一致。

## 模型训练 (Training)

所有的训练脚本都位于COMPARE/文件夹下。要训练我们的CDL-Net模型，请运行：

```bash
python COMPARE/CDL_ASPP.py 
```

训练过程中，模型检查点和日志会自动保存在ComplexNetDeepLab_Reg_Output/文件夹中。

## 预训练模型 (Pre-trained Models)

我们提供了在CST大腿模型数据集上训练好的最佳模型。您可以在本仓库的ComplexNetDeepLab+aspp_Reg_Output/ComplexNetDeepLab+aspp_Reg_best_model.pth下载。
## 引用 (Citation)

如果您觉得我们的工作对您的研究有所帮助，请考虑引用我们的论文：

```bibtex
@article{YourName_2025_CDLNet,
  title   = {[Dual-Branch Complex Networks for High-Resolution Complex Permittivity Inversion]},
  author  = {[绰绰,]},
  journal = {[啥啥啥,]},
  year    = {2025},
  volume  = {},
  number  = {},
  pages   = {}
}
```

## 致谢 (Acknowledgements)

感谢导师梁继民老师，郭开泰老师，蒋晓天学长和Dr.小盈🐈在本研究中的指导和帮助。
