# Prompt based continual learning with OOD Score based replay buffer

This repository contains the implementation of KSC2022 paper: **Prompt based continual learning with OOD Score based replay buffer**


## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 3090
- Python 3.8

## Usage
First, clone the repository locally:
```
git clone https://github.com/TY-LEE-KR/Capstone_Design_2.git
cd Capstone_Designe_2
```
Then, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
```
These packages can be installed easily by 
```
pip install -r requirements.txt
```

## Data preparation
If you already have CIFAR-100 datasets, pass your dataset path to  `--data-path`.


If the dataset isn't ready, change the download argument in `continual_dataloader.py` as follows
```
datasets.CIFAR100(download=True)
```

## Train
To train a model on CIFAR-100, set the `--data-path` (path to dataset) and `--output-dir` (result logging directory) and other options in `train.sh` properly and run in <a href="https://slurm.schedmd.com/documentation.html">Slurm</a> system.

## Training
To train a model on CIFAR-100 via command line:

```
python main.py --model vit_base_patch16_224 --batch-size 24 --data-path /local_datasets/ --output_dir ./output --epochs 5
```

Or you can simply use 'train.sh' if you use slurm system.

```
sbatch train.sh
```

Also available in Slurm by changing options on `train.sh`

## Evaluation
To evaluate a trained model:
```
python main.py --eval
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
