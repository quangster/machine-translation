# Neural Machine Translation: English to Vietnamese

Deep Learning Project

![Python Version](https://img.shields.io/badge/python-3.10.15-blue)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.5.1-orange)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quangster/machine-translation/blob/main/colab.ipynb)

## Installation


```bash
# clone project
git clone https://github.com/quangster/machine-translation
cd machine-translation

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

## Download dataset
```bash
gdown --folder 1cPdLNnTlsj3N1FE9x6_K608bCAaYaVGM -O data
```

## Build vocabulary and dataset
```bash
"""
This function processes the given dataset and vocabulary files. 
Note that the vocab and dataset files have already been downloaded from Google Drive, so there is no need to re-run the command to download them. 
Please be aware that running this command might take a significant amount of time.
"""
python src/vocab.py
```