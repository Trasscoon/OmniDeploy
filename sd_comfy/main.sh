#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

trap 'error_exit "### ERROR ###"' ERR

echo "### Setting up Stable Diffusion Comfy ###"
log "Setting up Stable Diffusion Comfy"

# --- CUDA 13.0 forward compat layer (curl, not wget) ---
if [[ ! -d /usr/local/cuda-13.0/compat ]]; then
    log "Installing CUDA 13.0 forward-compatibility layer"
    curl -L -o /tmp/cuda-keyring_1.1-1_all.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y -qq cuda-compat-13-0
fi
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
ldconfig

# --- aria2 ---
command -v aria2c &> /dev/null || apt-get install -y -qq aria2

# --- Create persistent and ephemeral model folders ---
PERSISTENT_SUBDIRS=(checkpoints loras vae upscale_models controlnet embeddings diffusion_models text_encoders unet vae_approx model_patches geometry_estimation)
EPHEMERAL_SUBDIRS=(checkpoints loras vae upscale_models controlnet embeddings diffusion_models text_encoders unet vae_approx model_patches geometry_estimation)

mkdir -p "$MODEL_DIR"
for sub in "${PERSISTENT_SUBDIRS[@]}"; do
    mkdir -p "$MODEL_DIR/$sub"
done

# Remove any old symlink at /tmp/stable-diffusion-models
if [[ -L "$WORKING_DIR" ]]; then
    unlink "$WORKING_DIR"
fi
mkdir -p "$WORKING_DIR"
for sub in "${EPHEMERAL_SUBDIRS[@]}"; do
    mkdir -p "$WORKING_DIR/$sub"
done

# Remove old symlinks from ComfyUI repo's models folder (we rely on extra_model_paths.yaml)
rm -f "$REPO_DIR/models/checkpoints" "$REPO_DIR/models/vae" "$REPO_DIR/models/loras" \
       "$REPO_DIR/models/controlnet" "$REPO_DIR/models/embeddings" "$REPO_DIR/models/upscale_models" \
       "$REPO_DIR/models/diffusion_models" "$REPO_DIR/models/text_encoders" "$REPO_DIR/models/unet" \
       "$REPO_DIR/models/vae_approx" "$REPO_DIR/models/model_patches" "$REPO_DIR/models/geometry_estimation"

# --- Write extra_model_paths.yaml so ComfyUI scans BOTH /storage and /tmp ---
cat > "$REPO_DIR/extra_model_paths.yaml" << 'YAMLEOF'
# Persistent models on /storage
persistent:
    base_path: /storage/stable-diffusion-models
    checkpoints: checkpoints
    loras: loras
    vae: vae
    upscale_models: upscale_models
    controlnet: controlnet
    embeddings: embeddings
    diffusion_models: diffusion_models
    text_encoders: text_encoders
    unet: unet
    vae_approx: vae_approx
    model_patches: model_patches
    geometry_estimation: geometry_estimation

# Ephemeral models on /tmp
ephemeral:
    base_path: /tmp/stable-diffusion-models
    checkpoints: checkpoints
    loras: loras
    vae: vae
    upscale_models: upscale_models
    controlnet: controlnet
    embeddings: embeddings
    diffusion_models: diffusion_models
    text_encoders: text_encoders
    unet: unet
    vae_approx: vae_approx
    model_patches: model_patches
    geometry_estimation: geometry_estimation
YAMLEOF

# --- Download VAE files into /tmp (ephemeral, no storage cost) ---
dl() {
    local url="$1" dir="$2" file
    file=$(basename "$url")
    mkdir -p "$dir"
    if [[ ! -f "$dir/$file" ]]; then
        log "Downloading $file"
        aria2c -x 16 -s 16 -k 64M --console-log-level=warn --summary-interval=0 \
            -d "$dir" -o "$file" "$url" || wget -q --show-progress -P "$dir" "$url"
    else
        log "Exists, skipping: $file"
    fi
}

dl "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" "$WORKING_DIR/vae"
dl "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" "$WORKING_DIR/vae"
dl "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors" "$WORKING_DIR/vae"

if [[ "$REINSTALL_SD_COMFY" || ! -f "/tmp/sd_comfy.prepared" ]]; then

    TARGET_REPO_URL="https://github.com/comfyanonymous/ComfyUI.git" \
    TARGET_REPO_DIR=$REPO_DIR \
    UPDATE_REPO=$SD_COMFY_UPDATE_REPO \
    UPDATE_REPO_COMMIT=$SD_COMFY_UPDATE_REPO_COMMIT \
    prepare_repo

    # --- Outputs go to /storage (small) ---
    prepare_link "$REPO_DIR/output:$IMAGE_OUTPUTS_DIR/stable-diffusion-comfy"

    # --- Persistent custom nodes on /storage, symlinked into repo ---
    mkdir -p /storage/comfy_custom_nodes
    for node in "https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager" \
                "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3 ComfyUI-Spectrum-MiniMax-H3"; do
        url=$(echo $node | cut -d' ' -f1)
        name=$(echo $node | cut -d' ' -f2)
        [ ! -d "/storage/comfy_custom_nodes/$name" ] && \
            git clone "$url" "/storage/comfy_custom_nodes/$name"
        ln -sfn "/storage/comfy_custom_nodes/$name" "$REPO_DIR/custom_nodes/$name"
    done

    rm -rf $VENV_DIR/sd_comfy-env
    python3.10 -m venv $VENV_DIR/sd_comfy-env
    source $VENV_DIR/sd_comfy-env/bin/activate

    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH' \
        >> $VENV_DIR/sd_comfy-env/bin/activate

    pip install pip==24.0
    pip install --upgrade wheel setuptools

    cd $REPO_DIR

    pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu130
    pip install xformers==0.0.35 --no-deps --index-url https://download.pytorch.org/whl/cu130

    cat > /tmp/torch-constraints.txt << 'CONSTEOF'
torch==2.13.0
torchvision==0.28.0
torchaudio==2.11.0
xformers==0.0.35
CONSTEOF
    pip install -r requirements.txt -c /tmp/torch-constraints.txt

    pip install --no-cache-dir comfy-aimdo==0.4.13
    pip install GitPython opencv-python-headless imageio-ffmpeg uv

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if [[ "$GPU_NAME" == *"A100"* ]]; then
        SAGE_ARCH="8.0"; SAGE_DIR="/storage/sage_wheels/sm80"
    else
        SAGE_ARCH="8.6"; SAGE_DIR="/storage/sage_wheels/sm86"
    fi
    log "GPU: $GPU_NAME -> sm_${SAGE_ARCH//./}"

    if ls "$SAGE_DIR"/sageattention-*.whl 1>/dev/null 2>&1; then
        log "Cached SageAttention wheel found, installing"
        pip install "$SAGE_DIR"/sageattention-*.whl
    else
        log "Building SageAttention (one-time, ~5 min)"
        pip install nvidia-cuda-nvcc nvidia-cuda-cccl
        NV="/tmp/sd_comfy-env/lib/python3.10/site-packages/nvidia"
        rm -rf /tmp/cuda13kit && mkdir -p /tmp/cuda13kit
        ln -sfn "$NV/cu13/bin"     /tmp/cuda13kit/bin
        ln -sfn "$NV/cu13/include" /tmp/cuda13kit/include
        ln -sfn "$NV/cu13/lib"     /tmp/cuda13kit/lib64
        mkdir -p /tmp/cuda13kit/lib64
        ln -sfn "$NV/cu13/lib/libcudart.so.13" /tmp/cuda13kit/lib64/libcudart.so.13 || true
        ln -sfn libcudart.so.13 /tmp/cuda13kit/lib64/libcudart.so || true

        cd /tmp && rm -rf sageattention_build
        git clone --depth 1 https://github.com/thu-ml/SageAttention.git sageattention_build
        cd sageattention_build
        export CUDA_HOME=/tmp/cuda13kit
        export PATH="$CUDA_HOME/bin:$PATH"
        export LIBRARY_PATH="$CUDA_HOME/lib64:$LIBRARY_PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        export TORCH_CUDA_ARCH_LIST="$SAGE_ARCH"
        export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
        pip wheel . --no-build-isolation --no-deps -w /tmp/sage_wheel_build
        mkdir -p "$SAGE_DIR"
        cp /tmp/sage_wheel_build/sageattention*.whl "$SAGE_DIR/"
        pip install "$SAGE_DIR"/sageattention*.whl
        cd /tmp && rm -rf sageattention_build /tmp/sage_wheel_build
    fi

    python -c "
import ctypes, torch
lib = ctypes.CDLL('libcuda.so.1')
v = ctypes.c_int()
lib.cuDriverGetVersion(ctypes.byref(v))
assert v.value >= 13000, f'CUDA compat layer not active ({v.value})'
print(f'--- torch {torch.__version__}, compat API {v.value} ---')
"
    python -c "
import torch, sageattention
q = torch.randn(1, 8, 256, 64, device='cuda', dtype=torch.float16)
sageattention.sageattn(q, q, q)
torch.cuda.synchronize()
print('SageAttention OK on', torch.cuda.get_device_name(0))
"

    touch /tmp/sd_comfy.prepared
else
    source $VENV_DIR/sd_comfy-env/bin/activate
fi

log "Finished Preparing Environment for Stable Diffusion Comfy"

if [[ -z "$INSTALL_ONLY" ]]; then
  echo "### Starting Stable Diffusion Comfy ###"
  log "Starting Stable Diffusion Comfy"

  pkill -9 -f "service_loop" 2>/dev/null || true
  fuser -k ${SD_COMFY_PORT}/tcp 2>/dev/null || true
  pkill -9 -f "main.py" 2>/dev/null || true
  sleep 3

  mkdir -p "$(dirname "$LOG_DIR/sd_comfy.log")"
  cd "$REPO_DIR"
  VENV_PYTHON="$VENV_DIR/sd_comfy-env/bin/python"
  PYTHONUNBUFFERED=1 service_loop \
"$VENV_PYTHON main.py --listen 0.0.0.0 --highvram --port $SD_COMFY_PORT --comfy-api-base sd-comfy --disable-pinned-memory --disable-async-offload ${EXTRA_SD_COMFY_ARGS}" \
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
