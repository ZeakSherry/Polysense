# Polysense

Repository: [https://github.com/ZeakSherry/Polysense.git](https://github.com/ZeakSherry/Polysense.git)

## Installation

### Setup Environment
```bash
# Clone repository
git clone https://github.com/ZeakSherry/Polysense.git
cd Polysense

# Create conda environment
conda create -n sampoly python=3.10 -y
conda activate sampoly

# Install PyTorch (CUDA 11.7)
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117

# Install dependencies
pip install -r requirements.txt

# Install pycocotools
cd pycocotools
pip install .
cd ..
```


## Dataset Preparation

### SpaceNet Vegas Dataset
Download preprocessed dataset from [here](https://aistudio.baidu.com/datasetdetail/269168) and place in `dataset/spacenet` folder.

### WHU-mix Dataset
Download from [official source](http://gpcv.whu.edu.cn/data/whu-mix%20(vector)/whu_mix(vector).html), place in `dataset/whu_mix` folder, then run:
```bash
cd dataset
python preprocess.py
```

### Custom Dataset Structure
```
dataset/
└── dataset_name/
    ├── train/
    │   ├── images/
    │   └── ann.json
    ├── val/
    │   ├── images/
    │   └── ann.json
    └── test/
        ├── images/
        └── ann.json
```

## Training

### Prompt Mode
Single GPU:
```bash
python train.py --config configs/prompt_instance_spacenet.json --gpus 0
```

Multi-GPU:
```bash
python train.py --config configs/prompt_instance_spacenet.json --gpus 0 1 --distributed
```

### Auto Mode
Step 1 - Pretrain with full-image features:
```bash
python train.py --config configs/prompt_fullimg_spacenet.json
```

Step 2 - Train auto mode (update pretrain_chkpt path in config):
```bash
python train_auto.py --config configs/auto_spacenet.py
```

## Testing

### Prompt Mode
```bash
python test.py --task_name your_task_name
```

Or use pre-trained model:
```python
# In test.py:
args = load_args(parser, path='configs/prompt_instance_spacenet.json')
args.checkpoint = 'prompt_instance_spacenet.pth'
```

### Auto Mode
SpaceNet:
```bash
python test_auto.py --config configs/auto_spacenet.py --ckpt_path auto_spacenet.pth
```

WHU-mix Test2:
```bash
python test_auto.py --config configs/auto_whumix.py --ckpt_path auto_whumix.pth --gt_pth dataset/whu_mix/test2/ann.json
```

WHU-mix Test1 (change test2 to test1 in configs/data_whu_mix.py):
```bash
python test_auto.py --config configs/auto_whumix.py --ckpt_path auto_whumix.pth --gt_pth dataset/whu_mix/test1/ann.json --score_thr 0.3
```

## Inference

### Prompt Mode
Command-line inference:
```bash
python infer_poly_crop.py  # Set args.imgpth, bbox, prompt_point
```

Interactive GUI:
```bash
python interactive_prompt.py
```

### Auto Mode
```bash
python infer_auto.py  # Set args.img_dir and args.img_suffix
```

Visualize predictions:
```bash
python utils/show_pred.py  # Set img_dir, dt_pth, img_suffix
```

## Pre-trained Models



## Acknowledgement
Based on [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) and [RSPrompter](https://github.com/KyanChen/RSPrompter).
