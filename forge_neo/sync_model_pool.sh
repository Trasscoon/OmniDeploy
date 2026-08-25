#!/bin/bash
POOL=/storage/stable-diffusion-models
TMPPOOL=/tmp/stable-diffusion-models

# Rescue any real files sitting in /tmp before bridging
for d in sd vae lora esrgan text_encoder embedding; do
    if [[ -d "$TMPPOOL/$d" && ! -L "$TMPPOOL/$d" ]]; then
        mkdir -p "$POOL/$d"
        cp -rn "$TMPPOOL/$d/"* "$POOL/$d/" 2>/dev/null
    fi
done

# /tmp folders -> /storage (downloads to /tmp land on persistent storage)
rm -rf "$TMPPOOL"
mkdir -p "$TMPPOOL"
for d in sd vae lora esrgan text_encoder embedding; do
    mkdir -p "$POOL/$d"
    ln -sfn "$POOL/$d" "$TMPPOOL/$d"
done

# Bridge ComfyUI-style folders so Forge Neo sees them
link_files() {
    [[ -d "$1" ]] || return 0
    mkdir -p "$2"
    find "$1" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" -o -name "*.bin" \) -exec ln -sf {} "$2/" \;
}
link_files "$POOL/diffusion_models" "$POOL/sd"
link_files "$POOL/checkpoints"      "$POOL/sd"
link_files "$POOL/unet"             "$POOL/sd"
link_files "$POOL/text_encoders"    "$POOL/text_encoder"
link_files "$POOL/loras"            "$POOL/lora"
link_files "$POOL/embeddings"       "$POOL/embedding"
link_files "$POOL/upscale_models"   "$POOL/esrgan"
link_files "$POOL/upscaler"         "$POOL/esrgan"
