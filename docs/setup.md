# Setup

We rely on the following external repositories:
* https://github.com/ngoyal2707/Megatron-LM/tree/fairseq_v2
* https://github.com/NVIDIA/apex
* https://github.com/facebookresearch/fairscale.git

## Install PyTorch
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
## Install Apex
```
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 265b451de8ba9bfcb67edc7360f3d8772d0a8bea
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```
Depending on the hardware you're running on, you may need to comment out lines 101-107 in setup.py here.
## Install Megatron
```
git clone --branch fairseq_v2 https://github.com/ngoyal2707/Megatron-LM.git
cd Megatron-LM
pip3 install six regex
pip3 install -e .
```
## Install fairscale
```
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
git checkout fixing_memory_issues_with_keeping_overlap_may24
pip3 install -e .
```
## Install metaseq
```
git clone https://github.com/facebookresearch/metaseq.git
cd metaseq
pip3 install -e .

# turn on pre-commit hooks
pre-commit install
```