import gradio as gr
from generate_image import generate_and_save_image
#import random

def generate_images(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, seed, batch_size,):
    # if seed == 0:
    #     seed = random.randint(0, 9999999999)
    print(f"Using seed: {seed}")
    image_paths = generate_and_save_image(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, seed, batch_size)
    return image_paths

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            image_display = gr.Gallery(show_label=False, allow_preview=True, columns=4, rows=4, scale=2)
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt")
            negative_prompt_input = gr.Textbox(label="Negative Prompt")
            width_slider = gr.Slider(minimum=512, maximum=4096, step=64, label="Width", value=1024)
            height_slider = gr.Slider(minimum=512, maximum=4096, step=64, label="Height", value=1024)
            batch_size_slider = gr.Slider(minimum=1, maximum=50, step=1, label="Batch Size", value=1)
            guidance_scale_slider = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, label="Guidance Scale", value=4.0)
            num_inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, label="Inference Steps", value=20)
            seed_input = gr.Number(label="Seed (enter 0 for random)", value=0)
            generate_button = gr.Button("Generate")
    
    generate_button.click(
        generate_images,
        inputs=[prompt_input, negative_prompt_input, width_slider, height_slider, guidance_scale_slider, num_inference_steps_slider, seed_input, batch_size_slider],
        outputs=image_display
    )

app.launch()
