# Probabilistic flow reconstruction with residual shifting model

This repository contains code and analysis for flow reconstruction with residual shifting model.
Please check out the report paper at `eng_106_report.pdf`.

## Dataset

The datasets for this project are available through Google Drive links provided below. Please download them and place in the home directory before running the code.

### Available Datasets

1. **Channel Flow**
   - Filename: `channel_flow.npy`
   - Download: [Google Drive Link](https://drive.google.com/file/d/1sgAjxpPCeB9JXfczI5sHTrZ1_Ocqq-rV/view?usp=drive_link)

2. **Cylinder Flow**
   - Filename: `cylinder_flow.npy`
   - Download: [Google Drive Link](https://drive.google.com/file/d/1rVKYG4CPhrpqXLQ5V5BgrGoRLh6yUGQ3/view?usp=sharing)

3. **Kolmogorov Flow**
   - Filename: `kolmogrov_flow.npy`
   - Download: [Google Drive Link](https://drive.google.com/file/d/1Q5DX2CPIaPqWLXL-axpoFdGSEJQ5OON2/view?usp=sharing)

## Training and testing the model

To train and test the model, simply run the provided shell script:

```bash
sh train.sh
```

This script will handle the data loading, model training, and evaluation processes automatically.

