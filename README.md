# CDL-Net: A Dual-Branch Complex-Valued Network for Wideband Microwave Inverse Imaging

<img width="13170" height="7398" alt="CDL_fram" src="https://github.com/user-attachments/assets/29257f69-d9ef-49d4-85c8-b44476bf43df" />

## Abstract

Microwave imaging (MWI) is a promising non-invasive technique for quantitatively reconstructing the complex permittivity of biological tissues. However, its clinical application is often limited by the non-linear and ill-posed nature of the electromagnetic inverse scattering problem. To address this challenge, we propose CDL-Net, a deep learning-based dual-branch network. The real-valued branch, leveraging a ResNet50 backbone and an Atrous Spatial Pyramid Pooling (ASPP) module, extracts multi-scale semantic features. Concurrently, the complex-valued branch processes phase information directly, preserving crucial physical properties consistent with wave propagation in lossy media. Our experiments, conducted on a realistic 3D human thigh phantom, demonstrate that CDL-Net achieves state-of-the-art performance, yielding a PSNR of 48.331 dB and an SSIM of 0.997.

## Project Structure

```
.
├── COMPARE/              # Contains all Python scripts for models, used for training and comparison
│   ├── CDL_ASPP.py       # The proposed CDL-Net model in this project
│   ├── unet.py           # U-Net baseline model
│   └── ...               # Other baseline models
├── sample_data/          # Contains a sample dataset for a quick demo
│   ├── E/                # Input data (electric field distribution)
│   └── label/            # Labels (ground truth complex permittivity distribution)
├── requirements.txt      # Required Python dependencies for the project
└── README.md             # This README file
```

---

## Setup

### 1. Clone this repository
```bash
git clone [https://github.com/molemolechuocho/CDL-EMIS.git](https://github.com/molemolechuocho/CDL-EMIS.git)
cd CDL-EMIS
```

### 2. Create a virtual environment and install dependencies
*(Conda is recommended)*
```bash
conda create -n cdlnet python=3.9
conda activate cdlnet
pip install -r requirements.txt 
```

---

## Dataset

A small dataset containing 5 test samples is provided in the `sample_data/` folder. This sample data is intended to be used with the provided `evaluate.py` script and our pre-trained model to demonstrate the model's functionality.

**The full dataset used for training is available upon reasonable request.** For access, please open an issue in this repository or contact the author at `yzli_3@stu.xidian.edu.cn`.

---

## Training

The scripts used to train the models in our paper are provided in the `COMPARE/` folder for methodological reference.

**Please note:** To run the training scripts, access to the full training dataset is required. As mentioned above, please contact the author to request access. The scripts are configured to read from a `EMTdata/` directory which you should create after receiving the dataset.

For instance, our proposed CDL-Net model was trained by running:
```bash
python COMPARE/CDL_ASPP.py
```
During the training process, model checkpoints and logs are automatically saved in a newly created directory,'ComplexNetDeepLab+aspp_Reg_Output\'.

## Pre-trained Models & Evaluation

We provide the weights for the best-performing model trained on the CST thigh phantom dataset.

### 1. Download the Model

* **Download Link:** [https://drive.google.com/file/d/1lVNff32iEdM0e5hqCfgtNNahy1iizu2b/view?usp=drive_link](https://drive.google.com/file/d/1lVNff32iEdM0e5hqCfgtNNahy1iizu2b/view?usp=drive_link)

* **Setup:** After downloading, please:
  1.  Create a folder named `pretrained_models` in the project's root directory.
  2.  Place the downloaded `.pth` model file inside this folder.

### 2. Run Evaluation

Once the model is in place, you can use `evaluate.py` to generate a comparison plot for a sample from the `sample_data` folder.

Run the following command in your terminal from the project's root directory:

```bash
python evaluate.py --model_path ./pretrained_models/cdl_best.pth
```

This command will:

1.  Load the pre-trained model.
2.  Perform inference on the first sample (index 0) from the `sample_data` directory.
3.  Create a `results/` folder and save the output image `prediction_....png` inside it.

To test other samples, you can specify the `--sample_index` argument:

```bash
# Example: Evaluate the third sample (index 2)
python evaluate.py --model_path ./pretrained_models/cdl_best.pth --sample_index 2
```

## Citation

If you find our work helpful for your research, please consider citing our paper:

```bibtex
@article{YourName_2025_CDLNet,
  title   = {CDL-Net: Dual-Branch Complex-Valued Networks for Microwave Complex Permittivity Inversion},
  author  = {[Author Name(s)]},
  journal = {[Journal Name]},
  year    = {2025},
  volume  = {},
  number  = {},
  pages   = {}
}
```

## Acknowledgements

We would like to thank our advisors, Prof. Jimin Liang and Prof. Kaitai Guo, as well as Xiaotian Jiang and Dr. Xiaoying 🐈 for their guidance and assistance in this research.
