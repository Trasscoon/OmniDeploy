#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

# Set up a trap to call the error_exit function on ERR signal
trap 'error_exit "### ERROR ###"' ERR

echo "### Setting up Stable Diffusion Comfy ###"
log "Setting up Stable Diffusion Comfy"
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

    pip install pip==24.0
    pip install --upgrade wheel setuptools

    cd $REPO_DIR

    # ═══════════════════════════════════════════
    # FIXED INSTALL ORDER
    # ═══════════════════════════════════════════

    # STEP 1: Install torch, torchvision, and torchaudio FIRST
    pip install \
        torch==2.6.0 \
        torchvision==0.21.0 \
        torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

    # STEP 2: Install xformers, telling it not to mess with torch
    pip install xformers==0.0.29.post2 --no-deps

    # STEP 3: Create a constraints file to protect torch from requirements.txt
    cat > /tmp/torch-constraints.txt << 'EOF'
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
xformers==0.0.29.post2
EOF

    # STEP 4: Install ComfyUI's requirements, respecting our constraints
    pip install -r requirements.txt -c /tmp/torch-constraints.txt

    # STEP 5: Install comfy-aimdo LAST, so it builds against the correct torch
    pip install --no-cache-dir comfy-aimdo==0.4.13

    # STEP 6: Verify everything is correct
    python -c "
import torch
import comfy_aimdo.host_buffer
import comfy_aimdo.vram_buffer
print('--- Verification Success: torch and comfy-aimdo are correctly installed ---')
"
    # ═══════════════════════════════════════════
    # END OF FIX
    # ═══════════════════════════════════════════

    touch /tmp/sd_comfy.prepared
else
    source $VENV_DIR/sd_comfy-env/bin/activate
fi
log "Finished Preparing Environment for Stable Diffusion Comfy"

if [[ -z "$SKIP_MODEL_DOWNLOAD" ]]; then
  echo "### Downloading Model for Stable Diffusion Comfy ###"
  log "Downloading Model for Stable Diffusion Comfy"
  bash $current_dir/../utils/sd_model_download/main.sh
  log "Finished Downloading Models for Stable Diffusion Comfy"
else
  log "Skipping Model Download for Stable Diffusion Comfy"
fi

# THIS IS THE CORRECTED VERSION
if [[ -z "$INSTALL_ONLY" ]]; then
  echo "### Starting Stable Diffusion Comfy ###"
  log "Starting Stable Diffusion Comfy"

  # THE ONE-LINE FIX: Create the log directory if it doesn't exist.
  mkdir -p "$(dirname "$LOG_DIR/sd_comfy.log")"

  cd "$REPO_DIR"
  VENV_PYTHON="$VENV_DIR/sd_comfy-env/bin/python"
  PYTHONUNBUFFERED=1 service_loop \
"$VENV_PYTHON main.py --listen 0.0.0.0 --highvram --port $SD_COMFY_PORT ${EXTRA_SD_COMFY_ARGS}" \
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
