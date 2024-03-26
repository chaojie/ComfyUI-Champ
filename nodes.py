import argparse
import logging
import os
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
node_path = folder_paths.get_folder_paths("custom_nodes")[0]
import os.path as osp
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from .models.unet_2d_condition import UNet2DConditionModel
from .models.unet_3d import UNet3DConditionModel
from .models.mutual_self_attention import ReferenceAttentionControl
from .models.guidance_encoder import GuidanceEncoder
from .models.champ_model import ChampModel

from .pipelines.pipeline_aggregation import MultiGuidance2LongVideoPipeline

from .utils.video_utils import resize_tensor_frames, save_videos_grid, pil_list_to_tensor, get_images


def setup_savedir(cfg):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if cfg.exp_name is None:
        savedir = f"results/exp-{time_str}"
    else:
        savedir = f"results/{cfg.exp_name}-{time_str}"
    
    os.makedirs(savedir, exist_ok=True)
    
    return savedir

def setup_guidance_encoder(cfg):
    guidance_encoder_group = dict()
    
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif cfg.weight_dtype == "float8_e4m3fn":
        weight_dtype = torch.float8_e4m3fn
    elif cfg.weight_dtype == "float8_e5m2":
        weight_dtype = torch.float8_e5m2
    else:
        weight_dtype = torch.float32
    
    for guidance_type in cfg.guidance_types:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
        ).to(device="cuda", dtype=weight_dtype)
    
    return guidance_encoder_group

def process_semantic_map(semantic_map_path: Path):
    image_name = semantic_map_path.name
    mask_path = semantic_map_path.parent.parent / "mask" / image_name
    semantic_array = np.array(Image.open(semantic_map_path))
    mask_array = np.array(Image.open(mask_path).convert("RGB"))
    semantic_pil = Image.fromarray(np.where(mask_array > 0, semantic_array, 0))
    
    return semantic_pil

def combine_guidance_data(cfg):
    guidance_types = cfg.guidance_types
    guidance_data_folder = cfg.data.guidance_data_folder
    
    guidance_pil_group = dict()
    for guidance_type in guidance_types:
        guidance_pil_group[guidance_type] = []
        for guidance_image_path in sorted(Path(osp.join(guidance_data_folder, guidance_type)).iterdir()):
            # Add black background to semantic map
            if guidance_type == "semantic_map":
                guidance_pil_group[guidance_type] += [process_semantic_map(guidance_image_path)]
            else:
                guidance_pil_group[guidance_type] += [Image.open(guidance_image_path).convert("RGB")]
    
    # get video length from the first guidance sequence
    first_guidance_length = len(list(guidance_pil_group.values())[0])
    # ensure all guidance sequences are of equal length
    assert all(len(sublist) == first_guidance_length for sublist in list(guidance_pil_group.values()))
    
    return guidance_pil_group, first_guidance_length

def inference(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    ref_image_pil,
    guidance_pil_group,
    video_length,
    width,
    height,
    device="cuda",
    dtype=torch.float16,
):
    reference_unet = model.reference_unet
    denoising_unet = model.denoising_unet
    guidance_types = cfg.guidance_types
    guidance_encoder_group = {f"guidance_encoder_{g}": getattr(model, f"guidance_encoder_{g}") for g in guidance_types}
    
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    pipeline = MultiGuidance2LongVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        **guidance_encoder_group,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device, dtype)
    
    video = pipeline(
        ref_image_pil,
        guidance_pil_group,
        width,
        height,
        video_length,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator
    ).videos
    
    del pipeline
    torch.cuda.empty_cache()
    
    return video

champ_path=f'{node_path}/ComfyUI-Champ'
config_path=f'{champ_path}/configs/inference.yaml'
class ChampLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/stable-diffusion-v1-5"}),
                "vae_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/sd-vae-ft-mse"}),
                "image_encoder_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/sd-image-variations-diffusers/image_encoder"}),
                "motion_module_path": ("STRING", {"default": "/home/admin/ComfyUI/models/champ/motion_module.pth"}),
                "denoising_unet_path": ("STRING", {"default": "/home/admin/ComfyUI/models/champ/denoising_unet.pth"}),
                "reference_unet_path": ("STRING", {"default": "/home/admin/ComfyUI/models/champ/reference_unet.pth"}),
                "depth_path": ("STRING", {"default": "/home/admin/ComfyUI/models/champ/guidance_encoder_depth.pth"}),
                "dwpose_path": ("STRING", {"default": "/home/admin/ComfyUI/models/champ/guidance_encoder_dwpose.pth"}),
                "normal_path": ("STRING", {"default": "/home/admin/ComfyUI/models/champ/guidance_encoder_normal.pth"}),
                "semantic_map_path": ("STRING", {"default": "/home/admin/ComfyUI/models/champ/guidance_encoder_semantic_map.pth"}),
                "weight_dtype": (["fp16","fp32"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("Champ","cfg","vae","image_enc","noise_scheduler",)
    RETURN_NAMES = ("champ","cfg","vae","image_enc","noise_scheduler",)
    FUNCTION = "run"
    CATEGORY = "Champ"

    def run(self,sd_path,vae_path,image_encoder_path,motion_module_path,denoising_unet_path,reference_unet_path,depth_path,dwpose_path,normal_path,semantic_map_path,weight_dtype):
        cfg = OmegaConf.load(config_path)
        OmegaConf.update(cfg, "base_model_path", sd_path)
        OmegaConf.update(cfg, "vae_model_path", vae_path)
        OmegaConf.update(cfg, "image_encoder_path", image_encoder_path)
        OmegaConf.update(cfg, "motion_module_path", motion_module_path)
        OmegaConf.update(cfg, "weight_dtype", weight_dtype)
        
        if cfg.weight_dtype == "fp16":
            weight_dtype = torch.float16
        elif cfg.weight_dtype == "bf16":
            weight_dtype = torch.bfloat16
        elif cfg.weight_dtype == "float8_e4m3fn":
            weight_dtype = torch.float8_e4m3fn
        elif cfg.weight_dtype == "float8_e5m2":
            weight_dtype = torch.float8_e5m2
        else:
            weight_dtype = torch.float32
            
        sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
        if cfg.enable_zero_snr:
            sched_kwargs.update( 
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        noise_scheduler = DDIMScheduler(**sched_kwargs)
        sched_kwargs.update({"beta_schedule": "scaled_linear"})
        
        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            cfg.image_encoder_path,
        ).to(dtype=weight_dtype, device="cuda")
        
        vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
            dtype=weight_dtype, device="cuda"
        )
        
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.base_model_path,
            cfg.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=cfg.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device="cuda")
        
        reference_unet = UNet2DConditionModel.from_pretrained(
            cfg.base_model_path,
            subfolder="unet",
        ).to(device="cuda", dtype=weight_dtype)
        
        guidance_encoder_group = setup_guidance_encoder(cfg)
        
        ckpt_dir = cfg.ckpt_dir
        denoising_unet.load_state_dict(
            torch.load(
                denoising_unet_path,
                map_location="cpu",
            ),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(
                reference_unet_path,
                map_location="cpu",
            ),
            strict=False,
        )
        
        for guidance_type, guidance_encoder_module in guidance_encoder_group.items():
            if guidance_type=="depth":
                guidance_encoder_module.load_state_dict(
                    torch.load(
                        depth_path,
                        map_location="cpu",
                    ),
                    strict=False,
                )
            if guidance_type=="normal":
                guidance_encoder_module.load_state_dict(
                    torch.load(
                        normal_path,
                        map_location="cpu",
                    ),
                    strict=False,
                )
            if guidance_type=="semantic_map":
                guidance_encoder_module.load_state_dict(
                    torch.load(
                        semantic_map_path,
                        map_location="cpu",
                    ),
                    strict=False,
                )
            if guidance_type=="dwpose":
                guidance_encoder_module.load_state_dict(
                    torch.load(
                        dwpose_path,
                        map_location="cpu",
                    ),
                    strict=False,
                )
            
        reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )
            
        model = ChampModel(
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            reference_control_writer=reference_control_writer,
            reference_control_reader=reference_control_reader,
            guidance_encoder_group=guidance_encoder_group,
        ).to("cuda", dtype=weight_dtype)
        
        if cfg.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )
        return (model,cfg,vae,image_enc,noise_scheduler,)


class ChampRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("Champ",),
                "cfg": ("cfg",),
                "vae": ("vae",),
                "image_enc": ("image_enc",),
                "noise_scheduler": ("noise_scheduler",),
                "image": ("IMAGE",),
                "depth_images": ("IMAGE",),
                "normal_images": ("IMAGE",),
                "semantic_map_images": ("IMAGE",),
                "dwpose_images": ("IMAGE",),
                "width": ("INT",{"default":512}),
                "height": ("INT",{"default":512}),
                "video_length": ("INT",{"default":16}),
                "num_inference_steps": ("INT",{"default":20}),
                "guidance_scale": ("FLOAT",{"default":3.5}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Champ"

    def run(self,model,cfg,vae,image_enc,noise_scheduler,image,depth_images,normal_images,semantic_map_images,dwpose_images,width,height,video_length,num_inference_steps,guidance_scale):
        ref_image = 255.0 * image[0].cpu().numpy()
        ref_image_pil = Image.fromarray(np.clip(ref_image, 0, 255).astype(np.uint8))
        ref_image_w, ref_image_h = ref_image_pil.size
        
        guidance_pil_group=dict()
        guidance_pil_group["depth"]=[Image.fromarray(np.clip(255.0*img.cpu().numpy(), 0, 255).astype(np.uint8)) for img in depth_images]
        guidance_pil_group["normal"]=[Image.fromarray(np.clip(255.0*img.cpu().numpy(), 0, 255).astype(np.uint8)) for img in normal_images]
        guidance_pil_group["semantic_map"]=[Image.fromarray(np.clip(255.0*img.cpu().numpy(), 0, 255).astype(np.uint8)) for img in semantic_map_images]
        guidance_pil_group["dwpose"]=[Image.fromarray(np.clip(255.0*img.cpu().numpy(), 0, 255).astype(np.uint8)) for img in dwpose_images]

        OmegaConf.update(cfg, "width", width)
        OmegaConf.update(cfg, "height", height)
        OmegaConf.update(cfg, "num_inference_steps", num_inference_steps)
        OmegaConf.update(cfg, "guidance_scale", guidance_scale)
        
        if cfg.weight_dtype == "fp16":
            weight_dtype = torch.float16
        elif cfg.weight_dtype == "bf16":
            weight_dtype = torch.bfloat16
        elif cfg.weight_dtype == "float8_e4m3fn":
            weight_dtype = torch.float8_e4m3fn
        elif cfg.weight_dtype == "float8_e5m2":
            weight_dtype = torch.float8_e5m2
        else:
            weight_dtype = torch.float32
        
        result_video_tensor = inference(
            cfg=cfg,
            vae=vae,
            image_enc=image_enc,
            model=model,
            scheduler=noise_scheduler,
            ref_image_pil=ref_image_pil,
            guidance_pil_group=guidance_pil_group,
            video_length=video_length,
            width=cfg.width, height=cfg.height,
            device="cuda", dtype=weight_dtype
        )  # (1, c, f, h, w)
        
        result_video_tensor = resize_tensor_frames(result_video_tensor, (ref_image_h, ref_image_w))
        
        return get_images(result_video_tensor)
    
NODE_CLASS_MAPPINGS = {
    "ChampLoader":ChampLoader,
    "ChampRun":ChampRun,
}