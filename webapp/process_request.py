import os
import traceback
import pandas as pd
from PIL import Image
import gradio as gr
from omegaconf import OmegaConf

# Use shared extraction pipeline
from utils.extract_utils import ExtractionPipeline

def _open_image(image_file):
    """Open an image file and convert it to RGB."""
    return Image.open(image_file.name).convert("RGB")

def create_pipeline(
    prompt,
    use_grounding_dino,
    box_threshold,
    text_threshold,
    grounding_prompt,
    llm_model_name,
    resize_size,
    temperature,
    config,
    batch_size = 1,
):
    # Build a small config including the requested llm model, temperature and prompt
    cfg_dict = {
        "preprocessors": {
            "grounding_dino": {
                "enabled": bool(use_grounding_dino),
                "model_name": getattr(config.preprocessors.grounding_dino, "model_name", "IDEA-Research/grounding-dino-tiny"),
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "prompt": grounding_prompt,
                "log_output": False,
                "device": "cpu",
                "max_outputs": getattr(config.preprocessors.grounding_dino, "max_outputs", 5),
            }
        },
        "img_max_size": resize_size,
        "llm": {"model_name": llm_model_name, "temperature": temperature, "prompt": prompt},
        "rate_limit_wait": getattr(config, "rate_limit_wait", True),
        "batch_size": batch_size,
        "batch_prompt": config.batch_prompt if batch_size != 1 else "",
    }

    cfg = OmegaConf.create(cfg_dict)

    return ExtractionPipeline(cfg)


def process_image(
    image,
    *args,
    **kwargs,
):
    """Process an image using an LLVM model.

    This function handles image processing through optional object detection (Grounding DINO) and subsequent
    text analysis using various LLVM models (Gemini, GPT, or Llama). It includes error handling and retry logic
    for LLVM processing.

    Args:
        image: Input image to be processed (numpy array or PIL Image)
        prompt (str): The prompt to be sent to the LLM model
        use_grounding_dino (bool): Whether to use Grounding DINO for object detection
        box_threshold (float): Confidence threshold for Grounding DINO bounding boxes
        text_threshold (float): Text threshold for Grounding DINO
        grounding_prompt (str): Prompt for Grounding DINO object detection
        llm_model_name (str): Name of the LLM model to use ('gemini', 'gpt', or 'llama')
        resize_size (int): Size to resize the image to (if needed)
        temperature (float): Temperature parameter for LLM generation
        config: Configuration object containing model settings

    Returns:
        tuple: (dict, list)
            - dict: Either processed output as key-value pairs or error message
            - list: List of processed images (original or detected regions)

    Raises:
        Various exceptions are caught and returned as error messages in the output dictionary
    """
    try:
        if image is None:
            return {"error": "No image provided"}, []

        if not isinstance(image, list):
            image = [image]

        pipeline = create_pipeline(*args, **kwargs)

        # Call the pipeline directly with the image(s). The pipeline returns parsed results.
        parsed_results, preprocessed_imgs = pipeline(image, add_image_names=False)

        return (parsed_results[0] if parsed_results else {"raw": None}), preprocessed_imgs

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, [image] if image else []

def process_batch(
    images,
    batch_size,
    output_format,
    *args,
    progress=gr.Progress(),
    **kwargs,
):
    if not images:
        return {"error": "No images provided"}, None, None

    results = []
    processed_images = []

    # Process images in batches
    if batch_size <= 0:
        batch_size = len(images)

    pipeline = create_pipeline(
        *args,
        **kwargs,
        batch_size=batch_size,
    )

    for batch_start in progress.tqdm(range(0, len(images), batch_size), desc="Processing images"):
        batch_end = min(batch_start + batch_size, len(images))
        batch_images = images[batch_start:batch_end]

        image_names = [os.path.basename(bi) for bi in batch_images]

        batch_images = [_open_image(image) for image in batch_images]

        try:
            result, imgs = pipeline(batch_images, image_names=image_names)

            results.extend(result)
            processed_images.extend(imgs)
        except Exception as e:
            results.append({"error": f"Error processing image: {str(e)}"})
            traceback.print_exc()

    df = pd.DataFrame(results)

    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)

    if output_format == "csv":
        output_path = os.path.join(temp_dir, "results.csv")
        df.to_csv(output_path, index=False)
    else:  # json
        output_path = os.path.join(temp_dir, "results.json")
        df.to_json(output_path, orient="records", indent=2)

    return results, processed_images, output_path
