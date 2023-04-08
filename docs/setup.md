# Setup

We rely on the following external repositories:
* https://github.com/facebookresearch/fairscale.git

## Install PyTorch
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
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
