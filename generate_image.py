import os
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import gc
import random
from datetime import datetime

# Setup the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_and_save_image(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, seed, batch_size=1):
    # create folder for each day inference is ran
    today_date = datetime.now().strftime("%m-%d-%Y")
    output_dir = os.path.join("outputs", today_date)

    os.makedirs(output_dir, exist_ok=True)

    # Generate and save images
    images_paths = []
    for _ in range(batch_size):
        current_seed = seed if seed != 0 else random.randint(0, 9999999999)
        generator = torch.Generator(device=device).manual_seed(current_seed)
        print(seed)

        prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device, torch_dtype=torch.float16)
        prior.safety_checker = None
        prior.requires_safety_checker = False

        prior_output = prior(prompt=prompt, generator=generator, width=width, height=height, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)

        del prior
        gc.collect()
        torch.cuda.empty_cache()

        decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.float16).to(device)
        decoder.safety_checker = None
        decoder.requires_safety_checker = False

        images = decoder(image_embeddings=prior_output.image_embeddings.half(), output_type="pil", prompt=prompt, negative_prompt=negative_prompt, guidance_scale=0.0, num_inference_steps=12).images

        # Save and collect image paths
        for i, img in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d%H%S%f")
            image_filename = os.path.join(output_dir, f"{timestamp}_{current_seed}.png")
            img.save(image_filename)
            images_paths.append(image_filename)
        
        del decoder
        gc.collect()
        torch.cuda.empty_cache()
        
    prompt_filename = os.path.join(output_dir, f"{timestamp}_prompt.txt")
    with open(prompt_filename, 'w') as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Negative Prompt: {negative_prompt}\n")
        f.write(f"Width: {width}\n")
        f.write(f"Height: {height}\n")
        f.write(f"Guidance Scale: {guidance_scale}\n")
        f.write(f"Num Inference Steps: {num_inference_steps}\n")



    return images_paths  # Return the list of saved image paths

