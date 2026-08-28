#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

trap 'error_exit "### ERROR ###"' ERR

echo "### Setting up Stable Diffusion WebUI Forge Neo ###"
log "Setting up Forge Neo"

# ============================================================
# CUDA 13.0 FORWARD COMPATIBILITY SETUP
# ============================================================
setup_cuda13() {
    if [[ -d "/usr/local/cuda-13.0/compat" && -f "/usr/local/cuda-13.0/bin/nvcc" ]]; then
        echo "CUDA 13.0 compat + toolkit already present."
        return 0
    fi
    echo "Installing CUDA 13.0 forward-compat layer + toolkit..."

    if ! apt-cache show cuda-compat-13-0 &>/dev/null; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb || true
        apt-get update
    fi

    if [[ ! -d "/usr/local/cuda-13.0/compat" ]]; then
        apt-get install -y cuda-compat-13-0 || return 1
    fi

    if [[ ! -f "/usr/local/cuda-13.0/bin/nvcc" ]]; then
        apt-get install -y cuda-toolkit-13-0 || \
            apt-get install -y cuda-nvcc-13-0 cuda-cudart-dev-13-0 \
                cuda-libraries-dev-13-0 cuda-nvml-dev-13-0
    fi

    [[ -d "/usr/local/cuda-13.0/compat" && -f "/usr/local/cuda-13.0/bin/nvcc" ]]
}

# --- CUDA variant detection ---
CUDA_VER="cpu"
TORCH_VER="2.5.1"
TORCHVISION_VER="0.20.1"
TORCHAUDIO_VER="2.5.1"

if command -v nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo "NVIDIA driver detected: $DRIVER_VER"

    if setup_cuda13; then
        export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}"
        export CUDA_HOME="/usr/local/cuda-13.0"
        export PATH="$CUDA_HOME/bin:$PATH"
        CUDA_VER="cu130"
        TORCH_VER="2.9.1"
        TORCHVISION_VER="0.24.1"
        TORCHAUDIO_VER="2.9.1"
        echo "CUDA 13.0 stack active (compat + nvcc) on driver $DRIVER_VER"
    else
        echo "CUDA 13.0 setup failed. Falling back to driver-native CUDA."
        MAJOR=$(echo $DRIVER_VER | cut -d'.' -f1)
        if [[ $MAJOR -ge 530 ]]; then
            CUDA_VER="cu124"
        elif [[ $MAJOR -ge 450 ]]; then
            CUDA_VER="cu121"
        else
            echo "Driver too old for CUDA builds. CPU-only."
            CUDA_VER="cpu"
        fi
    fi
else
    echo "No NVIDIA GPU detected. Installing CPU-only PyTorch."
fi
echo "Selected: CUDA=$CUDA_VER torch=$TORCH_VER"

install_torch() {
    if [[ "$CUDA_VER" == "cpu" ]]; then
        pip install torch==$TORCH_VER torchvision==$TORCHVISION_VER torchaudio==$TORCHAUDIO_VER
    else
        pip install torch==$TORCH_VER torchvision==$TORCHVISION_VER torchaudio==$TORCHAUDIO_VER \
            --extra-index-url https://download.pytorch.org/whl/$CUDA_VER
    fi
}

verify_cuda() {
    python -c "import torch; torch.cuda.current_device(); print('CUDA OK -', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)" 2>/dev/null
}

purge_xformers_if_unwanted() {
    if [[ "$CUDA_VER" == "cu130" || "$CUDA_VER" == "cpu" ]]; then
        echo "Ensuring xformers is not present (incompatible with torch 2.9.x/cu130)..."
        pip uninstall -y xformers || true
    fi
}

if [[ "$REINSTALL_FORGE_NEO" || ! -f "/tmp/forge_neo.prepared" ]]; then

    # --- Clone / update Forge Neo ---
    if [[ ! -d "$REPO_DIR" ]]; then
        echo "Cloning Forge Classic Neo..."
        git clone https://github.com/Haoming02/sd-webui-forge-classic.git \
            "$REPO_DIR" --branch neo
    else
        echo "Repo exists. Updating to latest neo..."
        cd "$REPO_DIR"
        git fetch origin
        git checkout neo
        git reset --hard origin/neo
        cd "$current_dir"
    fi

    # --- Symlinks ---
    symlinks=(
        "$REPO_DIR/outputs:$IMAGE_OUTPUTS_DIR/stable-diffusion-webui"
        "$MODEL_DIR:$WORKING_DIR/models"
        "$MODEL_DIR/sd:$LINK_MODEL_TO"
        "$MODEL_DIR/lora:$LINK_LORA_TO"
        "$MODEL_DIR/vae:$LINK_VAE_TO"
        "$MODEL_DIR/esrgan:$LINK_ESRGAN_TO"
        "$MODEL_DIR/text_encoder:$LINK_TEXT_ENCODER_TO"
        "$MODEL_DIR/embedding:$LINK_EMBEDDING_TO"
    )
    prepare_link "${symlinks[@]}"

    # --- Fresh venv ---
    rm -rf $VENV_DIR/forge_neo-env
    python3.11 -m venv $VENV_DIR/forge_neo-env
    source $VENV_DIR/forge_neo-env/bin/activate

    pip install pip==24.0
    pip install --upgrade wheel setuptools
    pip install numpy==1.26.4

    apt-get install -y libcairo2-dev libjpeg-dev libgif-dev

    pip uninstall -y torch torchvision torchaudio protobuf lxml || true

    # --- Torch install with fallback chain ---
    echo "Installing PyTorch $TORCH_VER for $CUDA_VER ..."
    if ! install_torch; then
        echo "torch $TORCH_VER+$CUDA_VER wheels unavailable. Trying cu124/2.5.1..."
        unset LD_LIBRARY_PATH
        unset CUDA_HOME
        CUDA_VER="cu124"; TORCH_VER="2.5.1"; TORCHVISION_VER="0.20.1"; TORCHAUDIO_VER="2.5.1"
        install_torch
    fi

    if [[ "$CUDA_VER" != "cpu" ]]; then
        echo "Verifying CUDA runtime..."
        if ! verify_cuda; then
            echo "$CUDA_VER runtime test failed. Falling back to cu124/2.5.1."
            pip uninstall -y torch torchvision torchaudio || true
            unset LD_LIBRARY_PATH
            unset CUDA_HOME
            CUDA_VER="cu124"; TORCH_VER="2.5.1"; TORCHVISION_VER="0.20.1"; TORCHAUDIO_VER="2.5.1"
            install_torch
            if ! verify_cuda; then
                echo "cu124 also failed. CPU-only it is."
                pip uninstall -y torch torchvision torchaudio || true
                CUDA_VER="cpu"
                install_torch
            fi
        fi
        [[ "$CUDA_VER" != "cpu" ]] && echo "CUDA verified working ($CUDA_VER)."
    fi

    # --- xformers: ONLY on cu124/cu121 path ---
    if [[ "$CUDA_VER" == "cu124" || "$CUDA_VER" == "cu121" ]]; then
        echo "Installing xformers for VAE fallback..."
        pip install xformers==0.0.28.post3 --no-deps || echo "xformers skipped (non-fatal)"
    fi

    purge_xformers_if_unwanted

    # --- Triton ---
    if [[ "$CUDA_VER" != "cpu" ]]; then
        pip install triton || echo "Using torch-bundled triton"
    fi

    # --- SageAttention from source ---
    if [[ "$CUDA_VER" != "cpu" ]]; then
        echo "Installing SageAttention from source..."
        echo "nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo 'not found')"

        SAGE_DIR="$REPO_DIR/repo/SageAttention"
        mkdir -p "$REPO_DIR/repo"
        if [[ ! -d "$SAGE_DIR" ]]; then
            git clone https://github.com/thu-ml/SageAttention.git "$SAGE_DIR"
        else
            cd "$SAGE_DIR" && git pull && cd "$current_dir"
        fi
        cd "$SAGE_DIR"
        pip install -e . --no-build-isolation
        cd "$current_dir"

        if python -c "from sageattention import sageattn; print('SageAttention IMPORT OK')" 2>/dev/null; then
            echo "SageAttention installed and importable."
        else
            echo "SageAttention build succeeded but import test failed."
        fi
    fi

    # --- Forge Neo's own dependency preparation ---
    export PYTHONPATH="$PYTHONPATH:$REPO_DIR"
    cd $REPO_DIR
    python $current_dir/preinstall.py
    cd $current_dir

    # --- Final pins ---
    if [[ "$CUDA_VER" != "cpu" ]]; then
        pip install --force-reinstall --no-deps torch==$TORCH_VER torchvision==$TORCHVISION_VER torchaudio==$TORCHAUDIO_VER \
            --extra-index-url https://download.pytorch.org/whl/$CUDA_VER
    fi
    pip install numpy==1.26.4
    pip install "pillow>=9.0.1,<11.0"
    pip install "markupsafe>=2.0,<3.0"

    touch /tmp/forge_neo.prepared
else
    source $VENV_DIR/forge_neo-env/bin/activate
    if [[ -d "/usr/local/cuda-13.0/compat" ]]; then
        export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}"
        export CUDA_HOME="/usr/local/cuda-13.0"
        export PATH="$CUDA_HOME/bin:$PATH"
        CUDA_VER="cu130"
    fi
    purge_xformers_if_unwanted
fi

log "Finished Preparing Environment for Forge Neo"

if [[ -z "$INSTALL_ONLY" ]]; then
    echo "### Starting Forge Neo ###"
    log "Starting Forge Neo"
    cd $REPO_DIR

    auth=""
    if [[ -n "${FORGE_NEO_GRADIO_AUTH}" ]]; then
        auth="--gradio-auth ${FORGE_NEO_GRADIO_AUTH}"
    fi

    EXTRA_FLAGS="--skip-torch-cuda-test --skip-python-version-check"
    if [[ "$CUDA_VER" == "cpu" ]]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --no-half"
    fi

    # --- PURGE xformers one last time before launch ---
    source $VENV_DIR/forge_neo-env/bin/activate 2>/dev/null || true
    pip uninstall -y xformers 2>/dev/null || true

    LAUNCH_ARGS="--port $FORGE_NEO_PORT --subpath sd-webui $auth --enable-insecure-extension-access $EXTRA_FLAGS ${EXTRA_FORGE_NEO_ARGS}"

    echo "Launch args: $LAUNCH_ARGS"

    mkdir -p "$LOG_DIR"
    PYTHONUNBUFFERED=1 service_loop "python launch.py $LAUNCH_ARGS" 2>&1 | tee $LOG_DIR/forge_neo.log &
    echo $! > /tmp/forge_neo.pid
fi

send_to_discord "Forge Neo Started"

if env | grep -q "PAPERSPACE"; then
    send_to_discord "Link: https://$PAPERSPACE_FQDN/sd-webui/"
fi

if [[ -n "${CF_TOKEN}" ]]; then
    if [[ "$RUN_SCRIPT" != *"forge_neo"* ]]; then
        export RUN_SCRIPT="$RUN_SCRIPT,forge_neo"
    fi
    bash $current_dir/../cloudflare_reload.sh
fi

echo "### Done ###"
