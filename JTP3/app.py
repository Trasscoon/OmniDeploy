from io import BytesIO
from threading import Lock

import numpy as np

import torch
from torch import Tensor
from torch.nn import Parameter

import spaces
from huggingface_hub import hf_hub_download
import gradio as gr

from PIL import Image, ImageDraw, ImageFont

import requests

from model import load_model, process_image, patchify_image
from image import unpatchify

device = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE = 16
MAX_SEQ_LEN = 1024


model_lock = Lock()
model, tag_list = load_model(
    hf_hub_download(repo_id="RedRocket/JTP-3", filename="models/jtp-3-hydra.safetensors"),
    device=device
)
model.requires_grad_(False)

tags = {
    tag.replace("_", " ").replace("vulva", "pussy"): idx
    for idx, tag in enumerate(tag_list)
}
tag_list = list(tags.keys())

FONT = ImageFont.load_default(24)

@spaces.GPU(duration=5)
@torch.no_grad()
def run_classifier(image: Image.Image, cam_depth: int):
    patches, patch_coords, patch_valid = patchify_image(image, PATCH_SIZE, MAX_SEQ_LEN)
    patches = patches.unsqueeze(0).to(device=device, non_blocking=True)
    patch_coords = patch_coords.unsqueeze(0).to(device=device, non_blocking=True)
    patch_valid = patch_valid.unsqueeze(0).to(device=device, non_blocking=True)

    patches = patches.to(dtype=torch.bfloat16).div_(127.5).sub_(1.0)
    patch_coords = patch_coords.to(dtype=torch.int32)

    with model_lock:
        features = model.forward_intermediates(
            patches,
            patch_coord=patch_coords,
            patch_valid=patch_valid,
            indices=cam_depth,
            output_dict=True,
            output_fmt='NLC'
        )

        logits = model.forward_head(features["image_features"], patch_valid=patch_valid)
        del features["image_features"]

    features["patch_coords"] = patch_coords
    features["patch_valid"] = patch_valid
    del patches, patch_coords, patch_valid

    probits = logits[0].float().sigmoid_().mul_(2.0).sub_(1.0) # scale to -1 to 1

    values, indices = probits.cpu().topk(250)
    predictions = {
        tag_list[idx.item()]: val.item()
        for idx, val in sorted(
            zip(indices, values),
            key=lambda item: item[1].item(),
            reverse=True
        )
    }

    return features, predictions

@spaces.GPU(duration=5)
@torch.no_grad()
def run_cam(
    display_image: Image.Image,
    image: Image.Image, features: dict[str, Tensor],
    tag_idx: int, cam_depth: int
):
    intermediates = features["image_intermediates"]
    if len(intermediates) < cam_depth:
        features, _ = run_classifier(image, cam_depth)
        intermediates = features["image_intermediates"]
    elif len(intermediates) > cam_depth:
        intermediates = intermediates[-cam_depth:]

    patch_coords = features["patch_coords"]
    patch_valid = features["patch_valid"]

    with model_lock:
        saved_q = model.attn_pool.q
        saved_p = model.attn_pool.out_proj.weight

        try:
            model.attn_pool.q = Parameter(saved_q[:, [tag_idx], :], requires_grad=False)
            model.attn_pool.out_proj.weight = Parameter(saved_p[[tag_idx], :, :], requires_grad=False)

            with torch.enable_grad():
                for intermediate in intermediates:
                    intermediate.requires_grad_(True).retain_grad()
                    model.forward_head(intermediate, patch_valid=patch_valid)[0, 0].backward()
        finally:
            model.attn_pool.q = saved_q
            model.attn_pool.out_proj.weight = saved_p

    cam_1d: Tensor | None = None
    for intermediate in intermediates:
        patch_grad = (intermediate.grad.float() * intermediate.sign()).sum(dim=(0, 2))
        intermediate.grad = None

        if cam_1d is None:
            cam_1d = patch_grad
        else:
            cam_1d.add_(patch_grad)

    assert cam_1d is not None

    cam_2d = unpatchify(cam_1d, patch_coords, patch_valid).cpu().numpy()
    return cam_composite(display_image, cam_2d), features

def cam_composite(image: Image.Image, cam: np.ndarray):
    """
    Overlays CAM on image and returns a PIL image.
    Args:
        image_pil: PIL Image (RGB)
        cam: 2D numpy array (activation map)

    Returns:
        PIL.Image.Image with overlay
    """

    cam_abs = np.abs(cam)
    cam_scale = cam_abs.max()

    cam_rgba = np.dstack((
        (cam < 0).astype(np.float32),
        (cam > 0).astype(np.float32),
        np.zeros_like(cam, dtype=np.float32),
        cam_abs * (0.5 / cam_scale),
    ))  # Shape: (H, W, 4)

    cam_pil = Image.fromarray((cam_rgba * 255).astype(np.uint8))
    cam_pil = cam_pil.resize(image.size, resample=Image.Resampling.NEAREST)

    image = Image.blend(
        image.convert('RGBA'),
        image.convert('L').convert('RGBA'),
        0.33
    )

    image = Image.alpha_composite(image, cam_pil)

    draw = ImageDraw.Draw(image)
    draw.text(
        (image.width - 7, image.height - 7),
        f"{cam_scale.item():.4g}",
        anchor="rd", font=FONT, fill=(32, 32, 255, 255)
    )

    return image

def filter_tags(predictions: dict[str, float], threshold: float):
    predictions = {
        key: value
        for key, value in predictions.items()
        if value >= threshold
    }

    tag_str = ", ".join(predictions.keys())
    return tag_str, predictions

def resize_image(image: Image.Image) -> Image.Image:
    longest_side = max(image.height, image.width)
    if longest_side < 1080:
        return image

    scale = 1080 / longest_side
    return image.resize(
        (
            int(round(image.width * scale)),
            int(round(image.height * scale)),
        ),
        resample=Image.Resampling.LANCZOS,
        reducing_gap=3.0
    )

def image_upload(image: Image.Image):
    display_image = resize_image(image)
    processed_image = process_image(image, PATCH_SIZE, MAX_SEQ_LEN)

    if display_image is not image and processed_image is not image:
        image.close()

    return (
        "", {}, "None", "",
        gr.skip() if display_image is image else display_image, display_image,
        processed_image,
    )

def url_submit(url: str):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    image = Image.open(BytesIO(resp.content))
    display_image = resize_image(image)
    processed_image = process_image(image, PATCH_SIZE, MAX_SEQ_LEN)

    if display_image is not image and processed_image is not image:
        image.close()

    return (
        "", {}, "None",
        display_image, display_image,
        processed_image,
    )

def image_changed(image: Image.Image, threshold: float, cam_depth: int):
    features, predictions = run_classifier(image, cam_depth)
    return *filter_tags(predictions, threshold), features, predictions

def image_clear():
    return (
        "", {}, "None", "",
        None, None,
        None, None, {},
    )

def cam_changed(
    display_image: Image.Image,
    image: Image.Image, features: dict[str, Tensor],
    tag: str, cam_depth: int
):
    if tag == "None":
        return display_image, features

    return run_cam(display_image, image, features, tags[tag], cam_depth)

def tag_box_select(evt: gr.SelectData):
    return evt.value

custom_css = """
.output-class { display: none; }
.inferno-slider input[type=range] {
    background: linear-gradient(to right,
        #000004, #1b0c41, #4a0c6b, #781c6d,
        #a52c60, #cf4446, #ed6925, #fb9b06,
        #f7d13d, #fcffa4
    ) !important;
    background-size: 100% 100% !important;
}
#image_container-image {
    width: 100%;
    aspect-ratio: 1 / 1;
    max-height: 100%;
}
#image_container img {
    object-fit: contain !important;
}
.show-api, .show-api-divider {
    display: none !important;
}
"""

with gr.Blocks(
    title="RedRocket JTP-3 Hydra Demo",
    css=custom_css,
    analytics_enabled=False,
) as demo:
    display_image_state = gr.State()
    image_state = gr.State()
    features_state = gr.State()
    predictions_state = gr.State(value={})

    gr.HTML(
        "<h1 style='display:flex; flex-flow: row nowrap; align-items: center;'>"
        "<a href='https://huggingface.co/RedRocket' target='_blank'>"
        "<img src='https://huggingface.co/spaces/RedRocket/README/resolve/main/RedRocket.png' style='width: 2em; margin-right: 0.5em;'>"
        "</a>"
        "<span><a href='https://huggingface.co/RedRocket' target='_blank'>RedRocket</a> &ndash; JTP-3 Hydra Demo</span>"
        "<span style='font-weight: normal;'>&nbsp;&bull;&nbsp;<a href='https://huggingface.co/RedRocket/JTP-3' target='_blank'>Download</a></span>"
        "</h1>"
    )

    with gr.Row():
        with gr.Column():
            with gr.Column():
                image = gr.Image(
                    sources=['upload', 'clipboard'], type='pil',
                    show_label=False,
                    show_download_button=False,
                    show_share_button=False,
                    elem_id="image_container"
                )

                url = gr.Textbox(
                    label="Upload Image via Url:",
                    placeholder="https://example.com/image.jpg",
                    max_lines=1,
                    submit_btn="⮝",
                )

            with gr.Column():
                cam_tag = gr.Dropdown(
                    value="None", choices=["None"] + tag_list,
                    label="CAM Attention Overlay (You can also click a tag on the right.)", show_label=True
                )
                cam_depth = gr.Slider(
                    minimum=1, maximum=27, step=1, value=1,
                    label="CAM Depth (1=fastest, more precise; 27=slowest, more general)"
                )

        with gr.Column():
            threshold_slider = gr.Slider(minimum=0.00, maximum=1.00, step=0.01, value=0.30, label="Tag Threshold")
            tag_string = gr.Textbox(lines=3, label="Tags", show_label=True, show_copy_button=True)
            tag_box = gr.Label(num_top_classes=250, show_label=False, show_heading=False)

    image.upload(
        fn=image_upload,
        inputs=[image],
        outputs=[
            tag_string, tag_box, cam_tag, url,
            image, display_image_state,
            image_state,
        ],
        show_progress='minimal',
        show_progress_on=[image]
    ).then(
        fn=image_changed,
        inputs=[image_state, threshold_slider, cam_depth],
        outputs=[
            tag_string, tag_box,
            features_state, predictions_state,
        ],
        show_progress='minimal',
        show_progress_on=[tag_box]
    )

    url.submit(
        fn=url_submit,
        inputs=[url],
        outputs=[
            tag_string, tag_box, cam_tag,
            image, display_image_state,
            image_state,
        ],
        show_progress='minimal',
        show_progress_on=[url]
    ).then(
        fn=image_changed,
        inputs=[image_state, threshold_slider, cam_depth],
        outputs=[
            tag_string, tag_box,
            features_state, predictions_state,
        ],
        show_progress='minimal',
        show_progress_on=[tag_box]
    )

    image.clear(
        fn=image_clear,
        inputs=[],
        outputs=[
            tag_string, tag_box, cam_tag, url,
            image, display_image_state,
            image_state, features_state, predictions_state,
        ],
        show_progress='hidden'
    )

    threshold_slider.input(
        fn=filter_tags,
        inputs=[predictions_state, threshold_slider],
        outputs=[tag_string, tag_box],
        trigger_mode='always_last',
        show_progress='hidden'
    )

    cam_tag.input(
        fn=cam_changed,
        inputs=[
            display_image_state,
            image_state, features_state,
            cam_tag, cam_depth,
        ],
        outputs=[image, features_state],
        trigger_mode='always_last',
        show_progress='minimal',
        show_progress_on=[cam_tag]
    )

    cam_depth.input(
        fn=cam_changed,
        inputs=[
            display_image_state,
            image_state, features_state,
            cam_tag, cam_depth,
        ],
        outputs=[image, features_state],
        trigger_mode='always_last',
        show_progress='minimal',
        show_progress_on=[cam_depth]
    )

    tag_box.select(
        fn=tag_box_select,
        inputs=[],
        outputs=[cam_tag],
        trigger_mode='always_last',
        show_progress='hidden',
    ).then(
        fn=cam_changed,
        inputs=[
            display_image_state,
            image_state, features_state,
            cam_tag, cam_depth,
        ],
        outputs=[image, features_state],
        show_progress='minimal',
        show_progress_on=[cam_tag]
    )

if __name__ == "__main__":
    demo.launch()
