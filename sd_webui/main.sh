#!/bin/bash
set -e

current_dir=$(dirname "$(realpath "$0")")
cd $current_dir
source .env

# Set up a trap to call the error_exit function on ERR signal
trap 'error_exit "### ERROR ###"' ERR

echo "### Setting up Stable Diffusion WebUI ###"
log "Setting up Stable Diffusion WebUI"

# --- CUDA detection runs ALWAYS (outside install block) so launch section can use it too ---
CUDA_VER="cpu"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo "NVIDIA driver detected: $DRIVER_VER"
    MAJOR=$(echo $DRIVER_VER | cut -d'.' -f1)
    echo "Driver major version: $MAJOR"
    if [[ $MAJOR -ge 530 ]]; then
        CUDA_VER="cu121"
    elif [[ $MAJOR -ge 520 ]]; then
        CUDA_VER="cu120"
    elif [[ $MAJOR -ge 450 ]]; then
        CUDA_VER="cu118"
    else
        echo "Driver $DRIVER_VER is too old for CUDA builds. Falling back to CPU-only PyTorch."
        CUDA_VER="cpu"
    fi
else
    echo "No NVIDIA GPU detected. Installing CPU-only PyTorch."
fi
echo "Selected CUDA variant: $CUDA_VER"

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

    rm -rf $VENV_DIR/sd_webui-env

    # --- Use python3.11 to match the gradient-base:pt211 container ---
    python3.11 -m venv $VENV_DIR/sd_webui-env
    source $VENV_DIR/sd_webui-env/bin/activate

    pip install pip==24.0
    pip install --upgrade wheel setuptools

    # 🔒 Pin NumPy early to prevent ABI mismatches
    pip install numpy==1.26.4

    # fix install issue with pycairo, which is needed by sd-webui-controlnet
    apt-get install -y libcairo2-dev libjpeg-dev libgif-dev

    # remove any preinstalled versions
    pip uninstall -y torch torchvision torchaudio protobuf lxml || true

    echo "Installing torch/torchvision/torchaudio for $CUDA_VER ..."
    if [[ "$CUDA_VER" == "cpu" ]]; then
        pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
    else
        pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
            --extra-index-url https://download.pytorch.org/whl/$CUDA_VER
    fi

    # --- Live CUDA verification: if torch.cuda actually fails at runtime, fall back to CPU ---
    if [[ "$CUDA_VER" != "cpu" ]]; then
        echo "Verifying CUDA is actually usable at runtime..."
        if ! python3.11 -c "import torch; torch.cuda.current_device(); print('CUDA OK')" 2>/dev/null; then
            echo "⚠️  CUDA runtime test failed with $CUDA_VER build. Falling back to CPU-only torch."
            pip uninstall -y torch torchvision torchaudio || true
            pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
            CUDA_VER="cpu"
        else
            echo "✅ CUDA verified working."
        fi
    fi

    # --- Patch A1111 errors.py for Python 3.11 traceback compatibility ---
    # Only applied if the unpatched pattern is present; skipped on dev branch builds
    # that already handle this natively.
    ERRORS_PY="$REPO_DIR/modules/errors.py"
    if [[ -f "$ERRORS_PY" ]] && grep -q "te = traceback.TracebackException.from_exception(e)" "$ERRORS_PY" && ! grep -q "isinstance(e, BaseException)" "$ERRORS_PY"; then
        echo "Patching $ERRORS_PY ..."
        python3.11 - "$ERRORS_PY" <<'PYEOF'
import sys, py_compile, tempfile, os

path = sys.argv[1]
with open(path, 'r') as f:
    src = f.read()

old = 'te = traceback.TracebackException.from_exception(e)'

# Detect the exact indentation of the target line
indent = ''
for line in src.split('\n'):
    if old in line:
        indent = line[:len(line) - len(line.lstrip())]
        break
else:
    print("Pattern not found - skipping patch.")
    sys.exit(0)

# Build replacement preserving detected indentation
new = (
    indent + 'if isinstance(e, BaseException):\n' +
    indent + '    te = traceback.TracebackException.from_exception(e)\n' +
    indent + 'else:\n' +
    indent + '    te = traceback.TracebackException(type(Exception), Exception(str(e)), None)'
)

patched = src.replace(indent + old, new)

# Validate syntax before writing to disk
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
    tmp.write(patched)
    tmp_path = tmp.name
try:
    py_compile.compile(tmp_path, doraise=True)
except py_compile.PyCompileError as e:
    print(f"Patch would produce invalid Python ({e}) - skipping.")
    os.unlink(tmp_path)
    sys.exit(0)
os.unlink(tmp_path)

with open(path, 'w') as f:
    f.write(patched)
print("errors.py patched successfully.")
PYEOF
    else
        echo "Skipping errors.py patch (dev branch already compatible or already patched)."
    fi

    export PYTHONPATH="$PYTHONPATH:$REPO_DIR"
    cd $REPO_DIR
    pip install --force-reinstall setuptools==69.5.1
    python $current_dir/preinstall.py

    # 🔒 Pin mediapipe AFTER preinstall so we override whatever version the
    #    controlnet_aux requirements resolved to. controlnet_aux 0.0.9 uses the
    #    legacy mp.solutions API (mp.solutions.drawing_utils etc.) which was
    #    removed in mediapipe >= 0.10.14. mediapipe 0.10.3 is the newest build
    #    that (a) still has the full solutions namespace and (b) ships cp311
    #    wheels (0.9.x never had cp311 wheels so cannot be used here).
    #    Available cp311 builds start at 0.10.5; use the oldest to stay as
    #    close as possible to the legacy API surface.
    pip install "mediapipe==0.10.5"

    cd $current_dir

    # 🔒 Pin scikit-image to last version compatible with numpy<2 (no-cache avoids
    #    stale wheels), then immediately re-anchor numpy so nothing else bumps it.
    pip install --force-reinstall scikit-image==0.21.0
    pip install numpy==1.26.4

    # 🔒 Pin xformers to the build that targets torch 2.1.x / CUDA 12.1.
    #    Install with --no-deps so pip cannot replace the already-correct
    #    torch 2.1.1+cu121 wheel with the generic PyPI torch 2.1.2 build.
    #    The generic build bundles its own libcusparse.so.12 which references
    #    __nvJitLinkAddData_12_1 — a symbol missing in the pt211 container's
    #    libnvJitLink — causing an ImportError at launch.
    pip install xformers==0.0.23.post1 --no-deps

    # Re-anchor torch stack after xformers in case anything slipped through.
    if [[ "$CUDA_VER" == "cpu" ]]; then
        pip install --force-reinstall torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
    else
        pip install --force-reinstall torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
            --extra-index-url https://download.pytorch.org/whl/$CUDA_VER
    fi

    # 🔒 FINAL re-pin of numpy and pillow.
    #    torchvision's --force-reinstall above re-resolves its own unpinned
    #    numpy and pillow deps, pulling numpy 2.4.4 and pillow 12.x — both of
    #    which break the stack:
    #      numpy 2.4.4 causes a fatal ABI mismatch in scikit-image 0.21.0
    #      ("numpy.dtype size changed"), crashing modules/processing.py and
    #      killing all image generation (txt2img, img2img, inpainting).
    #      pillow 12.x is rejected by gradio 3.41.2 (<11.0) and blendmodes (<10).
    #    These lines must remain the very last pip commands before the sentinel.
    pip install numpy==1.26.4
    pip install "pillow>=9.0.1,<10.0"
    # 🔒 Pin markupsafe: the --force-reinstall torch step pulls markupsafe 3.0.3
    #    via jinja2, but gradio 3.41.2 requires markupsafe~=2.0. 3.x causes
    #    Gradio template rendering errors in the WebUI.
    pip install "markupsafe>=2.0,<3.0"

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

  # --skip-torch-cuda-test is always safe: skips startup CUDA probe on GPU,
  # prevents crash loops on CPU. --no-half required when running without CUDA.
  EXTRA_FLAGS="--skip-torch-cuda-test"
  if [[ "$CUDA_VER" == "cpu" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --no-half"
  fi
  echo "Launch flags: $EXTRA_FLAGS"

  # ✅ Show logs in terminal and save to file
  PYTHONUNBUFFERED=1 service_loop "python webui.py --xformers --port $SD_WEBUI_PORT --subpath sd-webui $auth --controlnet-dir $MODEL_DIR/controlnet/ --enable-insecure-extension-access $EXTRA_FLAGS ${EXTRA_SD_WEBUI_ARGS}" 2>&1 | tee $LOG_DIR/sd_webui.log &
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
