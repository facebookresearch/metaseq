FROM metaseqopt.azurecr.io/metaseq-opt-singularity:latest

# switch to aiscuser as the user we want to use in the devcontainer.
# Note: this is the non-root user configured in the metaseq-opt-singularity
# image.
USER aiscuser

RUN pip install pre-commit==3.0.2
RUN pip install amlt==9.12.1 --extra-index-url https://msrpypi.azurewebsites.net/stable/leloojoo

# install zsh and oh-my-zsh
RUN sudo apt update && \
    sudo apt install -yq zsh && \
    sudo chsh $USER -s /usr/bin/zsh && \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# install azcopy, a tool to copy to/from blob storage
# for more info: https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-blobs-upload#upload-a-file
RUN wget https://azcopyvnext.azureedge.net/release20230123/azcopy_linux_amd64_10.17.0.tar.gz && \
    tar xvf azcopy_linux_amd64_10.17.0.tar.gz && \
    mkdir -p ~/.local/bin && \
    mv azcopy_linux_amd64_10.17.0/azcopy ~/.local/bin && \
    echo "export AZCOPY_CONCURRENCY_VALUE=AUTO" >> ~/.zshrc && \
    chmod +x ~/.local/bin/azcopy && \
    rm -rf azcopy_linux_amd64*

# setup rust-based tools
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="~/.cargo/bin:${PATH}"

RUN cargo install --locked \
    fd-find \
    ripgrep \
    http-server \
    watchexec-cli

# add user tools
RUN sudo apt install -yq \
        jq \
        jp \
        tree \
        tldr

RUN conda init zsh
