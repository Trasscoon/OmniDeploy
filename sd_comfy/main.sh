#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

# Set up a trap to call the error_exit function on ERR signal
trap 'error_exit "### ERROR ###"' ERR

echo "### Setting up Stable Diffusion Comfy ###"
log "Setting up Stable Diffusion Comfy"

# ═══════════════════════════════════════════
# CUDA 13.0 FORWARD COMPAT LAYER
# Installs UMD 580 userspace on the locked 550 host
# kernel, enabling cu130 PyTorch + comfy_kitchen's
# native CUDA kernels on the RTX A6000.
# Technique verified working on Paperspace A6000.
# ═══════════════════════════════════════════
if [[ ! -d /usr/local/cuda-13.0/compat ]]; then
    log "Installing CUDA 13.0 forward-compatibility layer"
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y -qq cuda-compat-13-0
fi
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
ldconfig
# ═══════════════════════════════════════════

if [[ "$REINSTALL_SD_COMFY" || ! -f "/tmp/sd_comfy.prepared" ]]; then

    TARGET_REPO_URL="https://github.com/comfyanonymous/ComfyUI.git" \
    TARGET_REPO_DIR=$REPO_DIR \
    UPDATE_REPO=$SD_COMFY_UPDATE_REPO \
    UPDATE_REPO_COMMIT=$SD_COMFY_UPDATE_REPO_COMMIT \
    prepare_repo

    symlinks=(
      "$REPO_DIR/output:$IMAGE_OUTPUTS_DIR/stable-diffusion-comfy"
      "$MODEL_DIR:$WORKING_DIR/models"
      "$MODEL_DIR/sd:$LINK_MODEL_TO"
      "$MODEL_DIR/lora:$LINK_LORA_TO"
      "$MODEL_DIR/vae:$LINK_VAE_TO"
      "$MODEL_DIR/upscaler:$LINK_UPSCALER_TO"
      "$MODEL_DIR/controlnet:$LINK_CONTROLNET_TO"
      "$MODEL_DIR/embedding:$LINK_EMBEDDING_TO"
    )
    prepare_link "${symlinks[@]}"
    # Create temporary symlinks for additional model folders
    mkdir -p /tmp/stable-diffusion-models

    ln -sfn /storage/stable-diffusion-comfy/models/diffusion_models \
        /tmp/stable-diffusion-models/diffusion_models

    ln -sfn /storage/stable-diffusion-comfy/models/text_encoders \
        /tmp/stable-diffusion-models/text_encoders

    rm -rf $VENV_DIR/sd_comfy-env

    python3.10 -m venv "$VENV_DIR/sd_comfy-env"
    source $VENV_DIR/sd_comfy-env/bin/activate

    # Persist the compat libcuda path for any shell that sources this venv
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH' \
        >> $VENV_DIR/sd_comfy-env/bin/activate

    pip install pip==24.0
    pip install --upgrade wheel setuptools

    cd $REPO_DIR

    # ═══════════════════════════════════════════
    # FIXED INSTALL ORDER — cu130 stack
    # ═══════════════════════════════════════════

    # STEP 1: Install torch, torchvision, and torchaudio FIRST
    pip install \
        torch==2.13.0 \
        torchvision==0.28.0 \
        torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu130

    # STEP 2: Install xformers, telling it not to mess with torch
    pip install xformers==0.0.35 --no-deps --index-url https://download.pytorch.org/whl/cu130

    # STEP 3: Create a constraints file to protect torch from requirements.txt
    cat > /tmp/torch-constraints.txt << 'EOF'
torch==2.13.0
torchvision==0.28.0
torchaudio==2.11.0
xformers==0.0.35
EOF

    # STEP 4: Install ComfyUI's requirements, respecting our constraints
    pip install -r requirements.txt -c /tmp/torch-constraints.txt

    # STEP 5: Install comfy-aimdo LAST, so it builds against the correct torch
    pip install --no-cache-dir comfy-aimdo==0.4.13

    # STEP 6: SageAttention — int8 attention on SM86 tensor cores
    pip install sageattention

    # STEP 7: Verify everything is correct
    python -c "
import ctypes, torch
import comfy_aimdo.host_buffer
import comfy_aimdo.vram_buffer

lib = ctypes.CDLL('libcuda.so.1')
v = ctypes.c_int()
lib.cuDriverGetVersion(ctypes.byref(v))
assert v.value >= 13000, f'CUDA 13.0 compat layer NOT active (got {v.value})'
print(f'--- Verification Success: torch {torch.__version__}, comfy-aimdo OK, compat layer API {v.value} ---')
"
    # ═══════════════════════════════════════════
    # END OF FIX
    # ═══════════════════════════════════════════

    touch /tmp/sd_comfy.prepared
else
    source $VENV_DIR/sd_comfy-env/bin/activate
fi
log "Finished Preparing Environment for Stable Diffusion Comfy"

# THIS IS THE CORRECTED VERSION
if [[ -z "$INSTALL_ONLY" ]]; then
  echo "### Starting Stable Diffusion Comfy ###"
  log "Starting Stable Diffusion Comfy"

  # THE ONE-LINE FIX: Create the log directory if it doesn't exist.
  mkdir -p "$(dirname "$LOG_DIR/sd_comfy.log")"

  cd "$REPO_DIR"
  VENV_PYTHON="$VENV_DIR/sd_comfy-env/bin/python"
  PYTHONUNBUFFERED=1 service_loop \
"$VENV_PYTHON main.py --listen 0.0.0.0 --highvram --port $SD_COMFY_PORT --use-sage-attention --preview-method latent-fast ${EXTRA_SD_COMFY_ARGS}" \
> "$LOG_DIR/sd_comfy.log" 2>&1 &
  echo $! > /tmp/sd_comfy.pid
fi

send_to_discord "Stable Diffusion Comfy Started"

if env | grep -q "PAPERSPACE"; then
  send_to_discord "Link: https://$PAPERSPACE_FQDN/sd-comfy/"
fi

if [[ -n "${CF_TOKEN}" ]]; then
  if [[ "$RUN_SCRIPT" != *"sd_comfy"* ]]; then
    export RUN_SCRIPT="$RUN_SCRIPT,sd_comfy"
  fi
  bash $current_dir/../cloudflare_reload.sh
fi

echo "### Done ###"
