#!/bin/bash

# install.sh
conda create -y -n repe_env python=3.12
conda activate repe_env
pip install -e .

pip install datasets
pip install rich
pip install tqdm
pip install matplotlib
pip install torch
pip install transformers
pip install hf_xet
pip install "huggingface_hub[hf_transfer]"
pip install "huggingface_hub[cli]"

pip install git+https://github.com/bkj/rcode.git

export HF_HUB_ENABLE_HF_TRANSFER=1