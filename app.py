import os
import json
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from nadm.models.unet import UNet3DConditionModel
from nadm.pipelines.pipeline_animation import AnimationPipeline
from nadm.utils.util import save_videos_grid
from nadm.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, \
    convert_ldm_vae_checkpoint
from nadm.utils.convert_lora_safetensor_to_diffusers import convert_lora
import cv2
import numpy as np

sample_idx = 0
scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}

css = """
body, html {
    height: 100%;
    margin: 0;
    padding: 0;
    font-family: 'Arial', sans-serif;
    background: #f5f5dc !important; 
}


button, input, select, textarea {
    background-color: #fff5e6  !important; 
    color: #000 !important; 
    border: 1px solid #fff5e6  !important; 
    padding: 10px !important;
    margin-bottom: 10px !important; 
    border-radius: 4px !important;
}


button:hover {
    background-color: #ccc !important; 
    color: #000 !important; 
}


.panel, .card {
    background-color: #f5f5dc !important;
    padding: 20px !important;
    border-radius: 15px !important;
    border: 1px solid #a0a0a0 !important; 
    box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 20px !important;
    position: relative; 
    overflow: hidden; 
}

.panel::before, .card::before {
    content: '';
    position: absolute;
    top: -10px; right: -10px; bottom: -10px; left: -10px;
    background: linear-gradient(45deg, transparent 15px, #f0e8d9 0), linear-gradient(-45deg, transparent 15px, #f0e8d9 0);
    background-repeat: repeat-x, repeat-x;
    background-size: 100px 100%, 100px 100%;
    background-position: left top, right bottom;
}
.markdown-title h1 {
    text-align: center;
    font-size: 24px; 

"""

class AnimateController:
    def __init__(self):

        # config dirs
        self.basedir = os.getcwd()
        self.stable_diffusion_dir = os.path.join(self.basedir, "models", "StableDiffusion")
        self.motion_module_dir = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(self.basedir, "models", "DreamBooth_LoRA")
        self.savedir = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample = os.path.join(self.savedir, "sample")
        os.makedirs(self.savedir, exist_ok=True)

        self.stable_diffusion_list = []
        self.motion_module_list = []
        self.personalized_model_list = []

        self.refresh_stable_diffusion()
        self.refresh_motion_module()
        self.refresh_personalized_model()

        # config models
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.pipeline = None
        self.lora_model_state_dict = {}

        self.inference_config = OmegaConf.load("configs/inference/inference.yaml")

    def refresh_stable_diffusion(self):
        self.stable_diffusion_list = glob(os.path.join(self.stable_diffusion_dir, "*/"))

    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.personalized_model_list = [os.path.basename(p) for p in personalized_model_list]

    def update_stable_diffusion(self, stable_diffusion_dropdown):
        self.tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_dropdown, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_dropdown, subfolder="text_encoder").cuda()
        self.vae = AutoencoderKL.from_pretrained(stable_diffusion_dropdown, subfolder="vae").cuda()
        self.unet = UNet3DConditionModel.from_pretrained_2d(stable_diffusion_dropdown, subfolder="unet",
                                                            unet_additional_kwargs=OmegaConf.to_container(
                                                                self.inference_config.unet_additional_kwargs)).cuda()
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
            motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
            motion_module_state_dict = motion_module_state_dict[
                "state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
            motion_module_state_dict = {key.replace("module.", ""): value for key, value in
                                        motion_module_state_dict.items()}

            missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)

            print(len(unexpected))
            assert len(unexpected) == 0
            return gr.Dropdown.update()

    def update_base_model(self, base_model_dropdown):
        if self.unet is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.Dropdown.update(value=None)
        else:
            base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
            base_model_state_dict = {}
            with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    base_model_state_dict[key] = f.get_tensor(key)

            converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_model_state_dict, self.vae.config)
            self.vae.load_state_dict(converted_vae_checkpoint)

            converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, self.unet.config)
            self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

            self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)
            return gr.Dropdown.update()

    def update_lora_model(self):
        self.lora_model_state_dict = {}

    def calculate_optical_flow(prev_frame, next_frame):

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)


        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        return flow

    def generate_interpolated_frame(prev_frame, next_frame, flow):
        height, width = prev_frame.shape[:2]
        new_positions = np.dstack(np.meshgrid(np.arange(width), np.arange(height))).astype(np.float32) + flow
        interpolated_frame = cv2.remap(prev_frame, new_positions, None, cv2.INTER_LINEAR)

        return interpolated_frame

    def animate(
            self,
            stable_diffusion_dropdown,
            motion_module_dropdown,
            base_model_dropdown,
            prompt_textbox,
            length_slider,
            seed_textbox
    ):
        if self.unet is None:
            raise gr.Error(f"Please select a pretrained model path.")
        if motion_module_dropdown == "":
            raise gr.Error(f"Please select a motion module.")
        if base_model_dropdown == "":
            raise gr.Error(f"Please select a base DreamBooth model.")

        if is_xformers_available(): self.unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
        vae=self.vae, 
        text_encoder=self.text_encoder, 
        tokenizer=self.tokenizer, 
        unet=self.unet,
        scheduler=EulerDiscreteScheduler(**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        ).to("cuda")

        if self.lora_model_state_dict != {}:
            pipeline = convert_lora(pipeline, self.lora_model_state_dict, alpha=0.8)

        pipeline.to("cuda")

        if seed_textbox != -1 and seed_textbox != "":
            torch.manual_seed(int(seed_textbox))
        else:
            torch.seed()
        seed = torch.initial_seed()

        sample = pipeline(
            prompt_textbox,
            num_inference_steps=25,
            guidance_scale=7.5,
            width=384,
            height=384,
            video_length=length_slider,
        ).videos
        sample_np = sample.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()

        def calculate_optical_flow(prev_frame, next_frame):
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            return flow

        def generate_interpolated_frame(prev_frame, next_frame, flow):
            height, width = prev_frame.shape[:2]
            new_positions = np.dstack(np.meshgrid(np.arange(width), np.arange(height))).astype(np.float32) + flow
            interpolated_frame_from_prev = cv2.remap(prev_frame, new_positions, None, cv2.INTER_LINEAR)
            inverse_flow = -flow
            new_positions_inverse = np.dstack(np.meshgrid(np.arange(width), np.arange(height))).astype(
                np.float32) + inverse_flow
            interpolated_frame_from_next = cv2.remap(next_frame, new_positions_inverse, None, cv2.INTER_LINEAR)
            interpolated_frame = cv2.addWeighted(interpolated_frame_from_prev, 0.5, interpolated_frame_from_next, 0.5,
                                                 0)
            return interpolated_frame

        interpolated_frames = []
        for i in range(len(sample_np) - 1):
            prev_frame = sample_np[i]
            next_frame = sample_np[i + 1]
            flow = calculate_optical_flow(prev_frame, next_frame)
            interpolated_frame = generate_interpolated_frame(prev_frame, next_frame, flow)
            interpolated_frames.extend([prev_frame, interpolated_frame])
        interpolated_frames.append(sample_np[-1])
        interpolated_frames_np = np.array(interpolated_frames)
        interpolated_frames_tensor = torch.from_numpy(interpolated_frames_np).float()
        interpolated_frames_tensor = interpolated_frames_tensor.permute(3, 0, 1, 2)
        sample = interpolated_frames_tensor.unsqueeze(0)

        save_sample_path = os.path.join(self.savedir_sample, f"{sample_idx}.mp4")
        save_videos_grid(sample, save_sample_path)

        sample_config = {
            "prompt": prompt_textbox,
            "sampler": "Euler",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "width": 384,
            "height": 384,
            "video_length": length_slider,
            "seed": seed
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")

        return gr.Video.update(value=save_sample_path)


controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
        """
        <div class="markdown-title">
        <h1>NADM: Noise-Aware Diffusion Model for Landscape Painting Video Generation</h1>
        </div>
        """,
        classname="markdown-title"
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Select Model(Please select in order).
                """
            )
            with gr.Row():
                with gr.Column(scale=2):
                    stable_diffusion_dropdown = gr.Dropdown(
                    label="Pretrained Model Path",
                    choices=controller.stable_diffusion_list,
                    interactive=True,
            )
                stable_diffusion_dropdown.change(fn=controller.update_stable_diffusion,
                                                 inputs=[stable_diffusion_dropdown],
                                                 outputs=[stable_diffusion_dropdown])

                with gr.Column(scale=2):
                    motion_module_dropdown = gr.Dropdown(
                    label="Select motion module",
                    choices=controller.motion_module_list,
                    interactive=True,
                )
                motion_module_dropdown.change(fn=controller.update_motion_module, inputs=[motion_module_dropdown],
                                              outputs=[motion_module_dropdown])

                with gr.Column(scale=2):
                    base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model (required)",
                    choices=controller.personalized_model_list,
                    interactive=True,
                )
                    base_model_dropdown.change(fn=controller.update_base_model, inputs=[base_model_dropdown],
                                           outputs=[base_model_dropdown])
                def update_personalized_model():
                    controller.refresh_personalized_model()
                    return [
                        gr.Dropdown.update(choices=controller.personalized_model_list),
                        gr.Dropdown.update(choices=["none"] + controller.personalized_model_list)
                    ]

            with gr.Column(variant="panel"):
                gr.Markdown("### 2. Configs for NADM.")
                preset_prompts = ["There is a withered tree on the mountaintop.there is another branch with petals falling. Across from it is still a mountain.Birds fly by in the sky.", "A person drives a small boat on a river in the mountains.some vegetation on both sides of the lake, a red-crowned crane flying over the lake", "Enter Your Own Description"]
                prompt_dropdown = gr.Dropdown(label="Choose or enter a prompt", choices=preset_prompts)
                prompt_textbox = gr.Textbox(label="Enter your prompt", visible=False, lines=2)
                def prompt_input_handler(choice):
                    if choice == "Enter Your Own Description":
                        return gr.Textbox.update(visible=True, value="")
                    else:
                        return gr.Textbox.update(visible=False, value=choice)

                prompt_dropdown.change(prompt_input_handler, inputs=[prompt_dropdown], outputs=[prompt_textbox])

            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        pass
                    length_slider = gr.Slider(label="Animation length", value=16, minimum=8, maximum=24, step=1)
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)), inputs=[],
                                          outputs=[seed_textbox])
                
                    generate_button = gr.Button(value="Generate", variant='primary')

                result_video = gr.Video(label="Generated Animation", interactive=False)
            

            generate_button.click(
                fn=controller.animate,
                inputs=[
                    stable_diffusion_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    prompt_textbox,
                    length_slider,
                    seed_textbox,
                ],
                outputs=[result_video]
            )

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.launch(share=True)