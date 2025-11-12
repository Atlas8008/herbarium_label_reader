import os
import gradio as gr
import hydra
from PIL import Image
from dotenv import load_dotenv
from webapp.process_request import process_image, process_batch

load_dotenv()
Image.MAX_IMAGE_PIXELS = None

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    llm_choices = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]

    def get_common_inputs():
        return [
            gr.Textbox(label="LLM Prompt", value=cfg.llm.prompt, lines=7),
            gr.Checkbox(label="Enable Label Detection", value=cfg.preprocessors.grounding_dino.enabled),
            gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=cfg.preprocessors.grounding_dino.box_threshold, step=0.01),
            gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=cfg.preprocessors.grounding_dino.text_threshold, step=0.01),
            gr.Textbox(label="Label Detection Prompt", value=cfg.preprocessors.grounding_dino.prompt, lines=7),
            gr.Dropdown(label="LLM Model", choices=llm_choices, value=cfg.llm.model_name),
            gr.Slider(label="Resize Size (px)", minimum=512, maximum=4096, value=4096, step=64, info="If the image is larger than this, the longer side will be resized to fit the specified resolution, keeping the aspect ratio. Higher resolutions typically generate better results, but can increase traffic and cost."),
            gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=getattr(cfg.llm, "temperature", 0.7), step=0.01, info="Determines the determinism of the LLVM, where higher values mean that the model is more \"creative\" (values up to 2.0), while lower values result in the model being more deterministic and reproducible (down to 0)."),
        ]

    # Single image interface
    single_interface = gr.Interface(
        fn=lambda *args: process_image(*args, config=cfg),
        inputs=[gr.Image(type="pil", label="Upload Image")] + get_common_inputs(),
        outputs=[
            gr.JSON(label="Extracted Label Data"),
            gr.Gallery(label="Processed Images"),
        ],
        delete_cache=[1800, 43200],
    )

    # Batch processing interface
    batch_interface = gr.Interface(
        fn=lambda *args: process_batch(*args, config=cfg),
        inputs=[
            gr.File(file_count="multiple", label="Upload Images", file_types=["image"]),
            gr.Number(label="Batch Size", value=1, precision=0, minimum=0, step=1, info="Number of images to process in a batch. Set to 0 to process all images at once. Bigger batches can be faster, but require more memory and can be more inaccurate, especially for large batches."),
            gr.Radio(choices=["csv", "json"], label="Output Format", value="csv"),
        ] + get_common_inputs(),
        outputs=[
            gr.JSON(label="Extracted Label Data"),
            gr.Gallery(label="Processed Images"),
            gr.File(label="Download Results"),
        ],
        delete_cache=[1800, 43200],
    )

    # Create tabbed interface
    demo = gr.TabbedInterface(
        [single_interface, batch_interface],
        tab_names=["Single Image", "Batch Processing"],
        title="Herbarium Label Reader",
    )

    demo.launch(share=True)

if __name__ == "__main__":
    main()