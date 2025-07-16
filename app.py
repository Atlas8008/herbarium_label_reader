import os
import gradio as gr
import hydra
from omegaconf import OmegaConf
from dotenv import load_dotenv
from PIL import Image

from llms import GeminiModel
from preprocessors import GroundingDinoPreprocessor

load_dotenv()

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    # Prepare LLM choices
    llm_choices = ["gemini-2.5-pro", "gemini-2.5-flash"]  # Extend as needed

    def process_image(
        image,
        prompt,
        use_grounding_dino,
        box_threshold,
        text_threshold,
        grounding_prompt,
        llm_model_name,
    ):
        preprocessor = None
        if use_grounding_dino:
            preprocessor = GroundingDinoPreprocessor(
                model_name=cfg.preprocessors.grounding_dino.model_name,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device="cpu",
                multi_output=True,
            )
            processed_images = preprocessor.preprocess(image, grounding_prompt)
            if not isinstance(processed_images, list):
                processed_images = [processed_images]
        else:
            processed_images = [image]

        if llm_model_name.startswith("gemini"):
            llm = GeminiModel(model_name=llm_model_name)
        else:
            return "Unsupported LLM model selected."

        output = llm.prompt(
            processed_images + [prompt]
        )

        output_dict = {}
        for line in output.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                output_dict[k.strip()] = v.strip()
        return output_dict

    iface = gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Textbox(label="LLM Prompt", value=cfg.llm.prompt),
            gr.Checkbox(label="Enable Grounding Dino", value=cfg.preprocessors.grounding_dino.enabled),
            gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=cfg.preprocessors.grounding_dino.box_threshold, step=0.01),
            gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=cfg.preprocessors.grounding_dino.text_threshold, step=0.01),
            gr.Textbox(label="Grounding Dino Prompt", value=cfg.preprocessors.grounding_dino.prompt),
            gr.Dropdown(label="LLM Model", choices=llm_choices, value=cfg.llm.model_name),
        ],
        outputs=gr.JSON(label="Extracted Label Data"),
        title="Herbarium Label Reader",
        description="Upload an image and extract label data using LLMs and optional Grounding Dino preprocessing.",
    )
    iface.launch()

if __name__ == "__main__":
    main()
