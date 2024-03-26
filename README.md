ComfyUI-Champ

# Download pretrained models (Can be placed in any path, as nodes can set absolute paths)

1. Download pretrained weight of base models: 
    - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
    - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

2. Download champ checkpoints:

|-- champ
|   |-- denoising_unet.pth
|   |-- guidance_encoder_depth.pth
|   |-- guidance_encoder_dwpose.pth
|   |-- guidance_encoder_normal.pth
|   |-- guidance_encoder_semantic_map.pth
|   |-- reference_unet.pth
|   `-- motion_module.pth

Set the path at ChampLoader node

## Examples

### base workflow

https://github.com/chaojie/ComfyUI-Champ/blob/main/wf.mp4

https://github.com/chaojie/ComfyUI-Champ/blob/main/wf.json

## champ

[champ](https://github.com/fudan-generative-vision/champ)