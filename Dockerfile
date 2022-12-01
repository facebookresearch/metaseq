FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Required for GPU builds to work
ARG DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /build
WORKDIR /build

# Required to fix issues with NVIDIA GPG Keys 
RUN apt-key del 7fa2af80 && \
    apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends curl && \
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# Install Python 3.9
RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
    git \
    python3-pip python3.9-dev

# Install Pytorch
RUN python3.9 -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install metaseq dependencies
RUN python3.9 -m pip install flake8==3.9.2 black==22.3.0 transformers pyarrow \
    boto3 pandas protobuf==3.20.2 aim>=3.9.4 azure-storage-blob black==22.3.0 \
    click==8.0.4 cython dataclasses editdistance fire flask==2.1.1 hydra-core==1.1.0 \
    ipdb ipython Jinja2==3.1.1 markupsafe more_itertools mypy ninja numpy omegaconf==2.1.1 \
    portalocker>=2.5 pre-commit pytest pytest-regressions regex scikit-learn sacrebleu \
    tensorboard==2.8.0 timeout-decorator tokenizers tqdm typing_extensions bitarray \
    sacremoses sentencepiece pybind11 

# Install APEX
ENV TORCH_CUDA_ARCH_LIST="3.7 5.0 6.0 7.0 7.5 8.0 8.6"
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /build/apex
RUN python3.9 -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
    --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./

# Install Megatron-LM branch
WORKDIR /build
RUN git clone --branch fairseq_v3 https://github.com/ngoyal2707/Megatron-LM.git
WORKDIR /build/Megatron-LM
RUN python3.9 -m pip install six regex
RUN python3.9 -m pip install -e .

# Install Fairscale
WORKDIR /build
RUN git clone --branch ngoyal_bf16_changes https://github.com/facebookresearch/fairscale.git
WORKDIR /build/fairscale
RUN python3.9 -m pip install --no-build-isolation -e .

# Install metaseq
WORKDIR /build
RUN git clone https://github.com/facebookresearch/metaseq.git
WORKDIR /build/metaseq
RUN python3.9 -m pip install -e .
RUN python3.9 setup.py install

# Set up git keys for internal repos
RUN apt install -y git openssh-client curl
ARG GH_SSH_PRIVATE_KEY
RUN mkdir /root/.ssh --parents
RUN echo "${GH_SSH_PRIVATE_KEY}" > /root/.ssh/private_key
RUN chmod 600 /root/.ssh/private_key
RUN git config --global http.postBuffer 1048576000

# Install metaseq-internal
RUN eval "$(ssh-agent -s)" && \
    ssh-add /root/.ssh/private_key && \
    ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts && \
    git clone --branch capi git@github.com:fairinternal/metaseq-internal.git && \
    rm -r /root/.ssh/private_key
WORKDIR /build/metaseq/metaseq-internal
RUN python3.9 -m pip install -e .

# API dependencies
WORKDIR /build/metaseq/metaseq-internal/api-infra
RUN python3.9 -m pip install redis[hiredis] celery

# Command to start worker
WORKDIR /build/metaseq/metaseq-internal/api-infra
CMD ["python3.9", "metaseq_worker.py"]