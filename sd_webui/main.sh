#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

# Set up a trap to call the error_exit function on ERR signal
trap 'error_exit "### ERROR ###"' ERR

echo "### Setting up Stable Diffusion WebUI ###"
log "Setting up Stable Diffusion WebUI"
if [[ "$REINSTALL_SD_WEBUI" || ! -f "/tmp/sd_webui.prepared" ]]; then

    TARGET_REPO_URL="https://github.com/AUTOMATIC1111/stable-diffusion-webui.git" \
    TARGET_REPO_DIR=$REPO_DIR \
    UPDATE_REPO=$SD_WEBUI_UPDATE_REPO \
    UPDATE_REPO_COMMIT=$SD_WEBUI_UPDATE_REPO_COMMIT \
    prepare_repo

    # --- Force A1111 dev branch (fixes GitHub auth / repo issues) ---
    cd "$REPO_DIR"
    git fetch origin
    git switch dev || git checkout -b dev origin/dev
    cd "$current_dir"

    # git clone extensions that have their own model folder
    if [[ ! -d "${REPO_DIR}/extensions/sd-webui-controlnet" ]]; then
        git clone https://github.com/Mikubill/sd-webui-controlnet.git "${REPO_DIR}/extensions/sd-webui-controlnet"
    fi
    if [[ ! -d "${REPO_DIR}/extensions/sd-webui-additional-networks" ]]; then
        git clone https://github.com/kohya-ss/sd-webui-additional-networks.git  "${REPO_DIR}/extensions/sd-webui-additional-networks"
    fi

    symlinks=(
        "$REPO_DIR/outputs:$IMAGE_OUTPUTS_DIR/stable-diffusion-webui"
        "$MODEL_DIR:$WORKING_DIR/models"
        "$MODEL_DIR/sd:$LINK_MODEL_TO"
        "$MODEL_DIR/lora:$LINK_LORA_TO"
        "$MODEL_DIR/vae:$LINK_VAE_TO"
        "$MODEL_DIR/hypernetwork:$LINK_HYPERNETWORK_TO"
        "$MODEL_DIR/controlnet:$LINK_CONTROLNET_TO"
        "$MODEL_DIR/embedding:$LINK_EMBEDDING_TO"
    )
    prepare_link  "${symlinks[@]}"

    # --- Create fresh virtual environment ---
    rm -rf $VENV_DIR/sd_webui-env
    python3.10 -m venv $VENV_DIR/sd_webui-env
    source $VENV_DIR/sd_webui-env/bin/activate

    # --- Bootstrap pip, wheel, setuptools ---
    pip install pip==24.0
    pip install --upgrade wheel setuptools

    # 🔒 Pin NumPy early to avoid ABI mismatch
    pip install numpy==1.26.4

    # fix install issue with pycairo, which is needed by sd-webui-controlnet
    apt-get install -y libcairo2-dev libjpeg-dev libgif-dev

    # remove any preinstalled versions
    pip uninstall -y torch torchvision torchaudio protobuf lxml || true

    # Detect CUDA version
    CUDA_VER=""
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        echo "NVIDIA driver detected: $DRIVER_VER"
        if [[ $(python3 -c "import torch; import re; print('cu121' if int(''.join(re.findall(r'[0-9]+', '$DRIVER_VER'))[:3]) >= 525 else 'cu118')") == "cu121" ]]; then
            CUDA_VER="cu121"
        else
            CUDA_VER="cu118"
        fi
    else
        CUDA_VER="cpu"
        echo "No NVIDIA GPU detected. Installing CPU-only PyTorch."
    fi

    echo "Installing torch/torchvision/torchaudio for $CUDA_VER ..."
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/$CUDA_VER

    export PYTHONPATH="$PYTHONPATH:$REPO_DIR"
    cd $REPO_DIR

    # 🔒 Prevent preinstall.py from upgrading NumPy
    pip install "numpy==1.26.4" --no-deps
    python $current_dir/preinstall.py
    cd $current_dir

    pip install xformers

    # 🔁 Reinstall scikit-image after everything to guarantee ABI match
    pip install --force-reinstall --no-cache-dir scikit-image

    touch /tmp/sd_webui.prepared
else
    source $VENV_DIR/sd_webui-env/bin/activate
fi

log "Finished Preparing Environment for Stable Diffusion WebUI"

if [[ -z "$SKIP_MODEL_DOWNLOAD" ]]; then
  echo "### Downloading Model for Stable Diffusion WebUI ###"
  log "Downloading Model for Stable Diffusion WebUI"
  bash $current_dir/../utils/sd_model_download/main.sh
  log "Finished Downloading Models for Stable Diffusion WebUI"
else
  log "Skipping Model Download for Stable Diffusion WebUI"
fi

if [[ -z "$INSTALL_ONLY" ]]; then
  echo "### Starting Stable Diffusion WebUI ###"
  log "Starting Stable Diffusion WebUI"
  cd $REPO_DIR
  auth=""
  if [[ -n "${SD_WEBUI_GRADIO_AUTH}" ]]; then
    auth="--gradio-auth ${SD_WEBUI_GRADIO_AUTH}"
  fi

  # ✅ Show logs in terminal and save to file
  PYTHONUNBUFFERED=1 service_loop "python webui.py --xformers --port $SD_WEBUI_PORT --subpath sd-webui $auth --controlnet-dir $MODEL_DIR/controlnet/ --enable-insecure-extension-access ${EXTRA_SD_WEBUI_ARGS}" 2>&1 | tee $LOG_DIR/sd_webui.log &
  echo $! > /tmp/sd_webui.pid
fi

send_to_discord "Stable Diffusion WebUI Started"

if env | grep -q "PAPERSPACE"; then
  send_to_discord "Link: https://$PAPERSPACE_FQDN/sd-webui/"
fi

if [[ -n "${CF_TOKEN}" ]]; then
  if [[ "$RUN_SCRIPT" != *"sd_webui"* ]]; then
    export RUN_SCRIPT="$RUN_SCRIPT,sd_webui"
  fi
  bash $current_dir/../cloudflare_reload.sh
fi

echo "### Done ###"
