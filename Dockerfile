FROM singularitybase.azurecr.io/base/job/pytorch/ptca-1.13.1-cuda11.7:20230112T151134502

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp/stage_dir
RUN mkdir -p ${STAGE_DIR} && \
    chmod 777 ${STAGE_DIR}

##############################################################################
# Installation/Basic Utilities
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        pdsh g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates \
        rsync iputils-ping net-tools sudo \
        libfuse-dev fuse \
        git git-lfs \
        # libnuma-dev is required by MLNX
        libnuma-dev \
        dos2unix psmisc graphviz llvm-10-dev ninja-build npm \
        libaio-dev \
        jq \
        lshw \
        dmidecode \
        util-linux \
        automake \
        autoconf \
        libtool \
        perftest \
        net-tools \
        openssh-client \
        openssh-server \
        pciutils \
        libaio-dev \
        libcap2 \
        default-jdk \
        lsb-release

RUN cp -s /usr/share/pyshared/lsb_release.py /opt/conda/envs/ptca/lib/python3.8/site-packages/lsb_release.py
RUN apt-get clean -y all

# Remove apt intermmediate files
RUN rm -rf /var/lib/apt/lists/*

##############################################################################
# Mellanox OFED
##############################################################################
ENV MLNX_OFED_VERSION=5.1-2.5.8.0
RUN cd ${STAGE_DIR} && \
        wget -q -O - http://content.mellanox.com/ofed/MLNX_OFED-${MLNX_OFED_VERSION}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64.tgz | tar xzf - && \
        cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64 && \
        ./mlnxofedinstall --user-space-only --without-fw-update --force --all -q --skip-unsupported-devices-check && \
        rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64*

##############################################################################
# Python (MLNX 5.1 requires python2 .......)
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3
RUN rm -f /usr/bin/python /usr/bin/python3 /usr/bin/pip && \
        ln -s /opt/conda/envs/ptca/bin/python3.8 /usr/bin/python3 && \
        ln -s /opt/conda/envs/ptca/bin/python3.8 /usr/bin/python && \
        ln -s /opt/conda/envs/ptca/bin/pip /usr/bin/pip && \
        # Print python and pip version
        python -V && pip -V

##############################################################################
# nv_peer_mem
##############################################################################
ENV NV_PEER_MEM_VERSION=1.1
ENV NV_PEER_MEM_TAG=1.1-0
RUN mkdir -p ${STAGE_DIR} && \
        git clone https://github.com/Mellanox/nv_peer_memory.git --branch ${NV_PEER_MEM_TAG} ${STAGE_DIR}/nv_peer_memory && \
        cd ${STAGE_DIR}/nv_peer_memory && \
        ./build_module.sh && \
        cd /tmp && \
        tar xzf /tmp/nvidia-peer-memory_${NV_PEER_MEM_VERSION}.orig.tar.gz && \
        cd nvidia-peer-memory-${NV_PEER_MEM_VERSION} && \
        apt-get update && \
        apt-get install -y dkms && \
        dpkg-buildpackage -us -uc && \
        dpkg -i /tmp/nvidia-peer-memory_${NV_PEER_MEM_TAG}_all.deb && \
        rm -rf /var/lib/apt/lists/* ${STAGE_DIR}/nv_peer_memory /tmp/nvidia-peer-memory_${NV_PEER_MEM_VERSION}.orig.tar.gz /tmp/nvidia-peer-memory-${NV_PEER_MEM_VERSION}

##############################################################################
# NCCL RDMA Sharp plugin
##############################################################################
RUN cd ${STAGE_DIR} && \
    mkdir -p /usr/local/nccl-rdma-sharp-plugins && \
    apt-get update && \
    apt-get install -y zlib1g-dev && \
    git clone https://github.com/Mellanox/nccl-rdma-sharp-plugins.git && \
    cd nccl-rdma-sharp-plugins && \
    git checkout v2.0.x-ar && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local/nccl-rdma-sharp-plugins --with-cuda=/usr/local/cuda && \
    make && \
    make install && \
    LD_LIBRARY_PATH=/usr/local/nccl-rdma-sharp-plugins/lib:${LD_LIBRARY_PATH} && \
    LD_PRELOAD=/usr/local/nccl-rdma-sharp-plugins/lib/libnccl-net.so:${LD_PRELOAD}

ENV LD_LIBRARY_PATH=/usr/local/nccl-rdma-sharp-plugins/lib:${LD_LIBRARY_PATH}

##############################################################################
# Create a non-root user. see https://aka.ms/vscode-remote/containers/non-root-user
##############################################################################
# we do this here to ensure that our user packages below are installed with the
# proper permissions

RUN sudo echo -e "[No password prompt]\nIdentity=unix-group:sudo\nAction=*\nResultActive=yes" \
> /etc/polkit-1/localauthority/50-local.d/45-allow-no-password.pkla
RUN chmod -R 777 /opt/conda/envs/ptca
RUN chmod -R 777 /tmp

ARG USERNAME=aiscuser

RUN echo $USERNAME ALL=\(ALL\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
        && chmod 0440 /etc/sudoers.d/$USERNAME

ENV SHELL /bin/bash
USER $USERNAME
WORKDIR /home/$USERNAME
ENV PATH="/home/${USERNAME}/.local/bin:/opt/conda/condabin:${PATH}"
RUN conda init bash
RUN sudo passwd -d `whoami`

##############################################################################
# User Packages
##############################################################################
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

RUN cd ${STAGE_DIR} && \
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    git checkout 265b451de8ba9bfcb67edc7360f3d8772d0a8bea && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./

# git checkout fa6c0860b62e4ed2ac13a513e7d950d72f576a44
RUN cd ${STAGE_DIR} && \
    git clone --branch fairseq_v3 https://github.com/ngoyal2707/Megatron-LM.git && \
    cd Megatron-LM && \
    git checkout fa6c0860b62e4ed2ac13a513e7d950d72f576a44 && \
    pip install six regex && \
    pip install .

# git checkout 91132c7e997c5affe97ce002e52cadd798220b06
RUN cd ${STAGE_DIR} && \
    git clone https://github.com/facebookresearch/fairscale.git && \
    cd fairscale && \
    git checkout fixing_memory_issues_with_keeping_overlap_may24 && \
    pip install .

RUN pip install \
        aim==3.16.2 \
        py-rouge==1.1 \
        rouge_score==0.1.2 \
        parlai==1.7.1 \
        evaluate==0.4.0

##############################################################################
# Switch back to root user so singularity can do runtime-setup
##############################################################################
USER root

ENV NLTK_DATA="/usr/share/nltk_data"
RUN python -c "import nltk; nltk.download('punkt', download_dir='${NLTK_DATA}')"
