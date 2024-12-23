import streamlit as st
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
import os
import uuid
from datetime import datetime

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

@st.cache_resource
def load_outpainter():
    class ImageOutpainter:
        def __init__(self):
            # Download and load models
            config_file = hf_hub_download(
                "xinsir/controlnet-union-sdxl-1.0",
                filename="config_promax.json",
            )

            config = ControlNetModel_Union.load_config(config_file)
            controlnet_model = ControlNetModel_Union.from_config(config)
            model_file = hf_hub_download(
                "xinsir/controlnet-union-sdxl-1.0",
                filename="diffusion_pytorch_model_promax.safetensors",
            )
            state_dict = load_state_dict(model_file)
            model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
                controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
            )
            model.to(device="cuda", dtype=torch.float16)

            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
            ).to("cuda")

            self.pipe = StableDiffusionXLFillPipeline.from_pretrained(
                "SG161222/RealVisXL_V5.0_Lightning",
                torch_dtype=torch.float16,
                vae=vae,
                controlnet=model,
                variant="fp16",
            ).to("cuda")

            self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)

        def can_expand(self, source_width, source_height, target_width, target_height, alignment):
            """Checks if the image can be expanded based on the alignment."""
            if alignment in ("Left", "Right") and source_width >= target_width:
                return False
            if alignment in ("Top", "Bottom") and source_height >= target_height:
                return False
            return True

        def prepare_image_and_mask(self, image, width, height, overlap_percentage=10, 
                                    resize_percentage=100, custom_resize_percentage=50, 
                                    alignment="Middle", 
                                    overlap_left=True, overlap_right=True, 
                                    overlap_top=True, overlap_bottom=True):
            """
            Prepare the image and mask for outpainting.
            
            :param image: PIL Image to be outpainted
            :param width: Target width of the final image
            :param height: Target height of the final image
            :param overlap_percentage: Percentage of image to overlap (default 10%)
            :param resize_option: Resize option ('Full', '50%', '33%', '25%', 'Custom')
            :param custom_resize_percentage: Custom resize percentage if 'Custom' is selected
            :param alignment: Image alignment ('Middle', 'Left', 'Right', 'Top', 'Bottom')
            :param overlap_left: Whether to overlap on the left side
            :param overlap_right: Whether to overlap on the right side
            :param overlap_top: Whether to overlap on the top side
            :param overlap_bottom: Whether to overlap on the bottom side
            :return: tuple of (background image, mask)
            """
            target_size = (width, height)

            # Calculate the scaling factor to fit the image within the target size
            scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            
            # Resize the source image to fit within target size
            source = image.resize((new_width, new_height), Image.LANCZOS)

            

            # Calculate new dimensions based on percentage
            resize_factor = resize_percentage / 100
            new_width = int(source.width * resize_factor)
            new_height = int(source.height * resize_factor)

            # Ensure minimum size of 64 pixels
            new_width = max(new_width, 64)
            new_height = max(new_height, 64)

            # Resize the image
            source = source.resize((new_width, new_height), Image.LANCZOS)

            # Calculate the overlap in pixels based on the percentage
            overlap_x = int(new_width * (overlap_percentage / 100))
            overlap_y = int(new_height * (overlap_percentage / 100))

            # Ensure minimum overlap of 1 pixel
            overlap_x = max(overlap_x, 1)
            overlap_y = max(overlap_y, 1)

            # Calculate margins based on alignment
            margin_x = (target_size[0] - new_width) // 2  # Default to middle alignment
            margin_y = (target_size[1] - new_height) // 2

            if alignment == "Left":
                margin_x = 0
            elif alignment == "Right":
                margin_x = target_size[0] - new_width
            elif alignment == "Top":
                margin_y = 0
            elif alignment == "Bottom":
                margin_y = target_size[1] - new_height

            # Adjust margins to eliminate gaps
            margin_x = max(0, min(margin_x, target_size[0] - new_width))
            margin_y = max(0, min(margin_y, target_size[1] - new_height))

            # Create a new background image and paste the resized source image
            background = Image.new('RGB', target_size, (255, 255, 255))
            background.paste(source, (margin_x, margin_y))

            # Create the mask
            mask = Image.new('L', target_size, 255)
            mask_draw = ImageDraw.Draw(mask)

            # Calculate overlap areas
            white_gaps_patch = 2

            left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
            right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
            top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
            bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
            
            if alignment == "Left":
                left_overlap = margin_x + overlap_x if overlap_left else margin_x
            elif alignment == "Right":
                right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
            elif alignment == "Top":
                top_overlap = margin_y + overlap_y if overlap_top else margin_y
            elif alignment == "Bottom":
                bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height

            # Draw the mask
            mask_draw.rectangle([
                (left_overlap, top_overlap),
                (right_overlap, bottom_overlap)
            ], fill=0)

            return background, mask

        def outpaint(self, image, output_dir, width=720, height=1280, 
                    prompt="", num_inference_steps=8, alignment="Middle", 
                    overlap_percentage=10, resize_percentage=100, 
                    custom_resize_percentage=50,
                    overlap_left=True, overlap_right=True, 
                    overlap_top=True, overlap_bottom=True):
            """
            Outpaint the input image.
            
            :param image: PIL Image to be outpainted
            :param output_dir: Directory to save output images
            :param width: Target width of the final image
            :param height: Target height of the final image
            :param prompt: Optional prompt to guide image generation
            :param num_inference_steps: Number of diffusion steps
            :param alignment: Image alignment
            :param overlap_percentage: Percentage of image to overlap
            :param resize_option: Resize option
            :param custom_resize_percentage: Custom resize percentage
            :param overlap_left: Whether to overlap on the left side
            :param overlap_right: Whether to overlap on the right side
            :param overlap_top: Whether to overlap on the top side
            :param overlap_bottom: Whether to overlap on the bottom side
            :return: Paths to generated images
            """
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            if image.size[0] > image.size[1]:
                image = image.rotate(270, expand=True)
            
            # Check if expansion is possible
            if not self.can_expand(image.width, image.height, width, height, alignment):
                alignment = "Middle"

            # Prepare image and mask
            background, mask = self.prepare_image_and_mask(
                image, width, height, overlap_percentage, 
                resize_percentage, custom_resize_percentage, 
                alignment, overlap_left, overlap_right, 
                overlap_top, overlap_bottom
            )

            # Prepare control net image
            cnet_image = background.copy()
            cnet_image.paste(0, (0, 0), mask)

            # Prepare prompt
            final_prompt = f"{prompt}, high quality, 4k"

            # Encode prompt
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(final_prompt, "cuda", True)

            # Generate the image
            generated_images = list(self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                image=cnet_image,
                num_inference_steps=num_inference_steps
            ))

            # Process generated images
            output_images = []
            for i, image in enumerate(generated_images):
                # Convert image to RGBA
                image = image.convert("RGBA")
                
                # Paste generated image onto background
                cnet_image.paste(image, (0, 0), mask)

                # Save output images
                base_filename = "outpaint"
                output_filename = f"{base_filename}_{i+1}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                
                cnet_image.save(output_path)
                output_images.append(output_path)

            return output_images
    return ImageOutpainter()

outpainter = load_outpainter()

def main():
    st.title("Image Outpainter")
    
    st.sidebar.header("Settings")
    
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    width = st.sidebar.slider("Width", 128, 2048, 960)
    height = st.sidebar.slider("Height", 128, 2048, 1280)

    prompt = st.sidebar.text_input("Prompt", "")

    num_inference_steps = st.sidebar.slider("Inference Steps", 1, 50, 10)
    
    alignment = st.sidebar.selectbox("Alignment", ["Middle", "Left", "Right", "Top", "Bottom"])

    overlap_percentage = st.sidebar.slider("Overlap Percentage", 0, 50, 0)
    resize_percentage = st.sidebar.slider("Resize Percentage", 10, 100, 75)
    #custom_resize_percentage = st.sidebar.slider("Custom Resize Percentage", 10, 100, 50)

    overlap_left = st.sidebar.checkbox("Overlap Left", True)
    overlap_right = st.sidebar.checkbox("Overlap Right", True)
    overlap_top = st.sidebar.checkbox("Overlap Top", True)
    overlap_bottom = st.sidebar.checkbox("Overlap Bottom", True)
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Outpaint"):
             with st.spinner("Outpainting..."):
                # Create unique run ID using timestamp and UUID
                run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
                output_dir = os.path.join("results", run_id)
                
                output_images = outpainter.outpaint(
                     image=image,
                     output_dir=output_dir,
                     width=width,
                     height=height,
                     prompt=prompt,
                     num_inference_steps=num_inference_steps,
                     alignment=alignment,
                     overlap_percentage=overlap_percentage,
                     resize_percentage=resize_percentage,
                     #custom_resize_percentage=custom_resize_percentage,
                     overlap_left=overlap_left,
                     overlap_right=overlap_right,
                     overlap_top=overlap_top,
                     overlap_bottom=overlap_bottom,
                )

             if output_images:
                 st.success("Outpainting complete!")
                 for i, output_image in enumerate(output_images):
                     st.image(Image.open(output_image), caption=f"Generated Image {i+1}", use_container_width=True)
             else:
                 st.error("No output generated.")

if __name__ == "__main__":
    main()