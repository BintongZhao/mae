#!/bin/bash

conda create --name mae python=3.8 -y
conda activate mae

pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install tensorboard==2.4.0
pip install protobuf==3.20.3
pip install numpy==1.19.5
pip install scipy

pip install matplotlib==3.3.4
pip install matplotlib-inline