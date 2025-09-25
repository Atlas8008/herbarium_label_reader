import os
import time
import traceback
import pandas as pd
from PIL import Image
import gradio as gr
from llms import GeminiModel
from preprocessors import GroundingDinoPreprocessor

def maybe_resize(img, max_size):
    """Resize the image if it exceeds the max size."""
    if img.size[0] > max_size or img.size[1] > max_size:
        img.thumbnail((max_size, max_size))
    return img

def process_image(
    image,
    prompt,
    use_grounding_dino,
    box_threshold,
    text_threshold,
    grounding_prompt,
    llm_model_name,
    resize_size,
    temperature,
    config,
):
    try:
        if image is None:
            return {"error": "No image provided"}, []

        image = maybe_resize(image, resize_size)

        preprocessor = None
        if use_grounding_dino:
            try:
                preprocessor = GroundingDinoPreprocessor(
                    model_name=config.preprocessors.grounding_dino.model_name,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device="cpu",
                    multi_output=True,
                )
                processed_images = preprocessor.preprocess(image, grounding_prompt)
                if not isinstance(processed_images, list):
                    processed_images = [processed_images]
            except Exception as e:
                return {"error": f"Grounding DINO error: {str(e)}"}, [image]
        else:
            processed_images = [image]

        if llm_model_name.startswith("gemini"):
            llm = GeminiModel(model_name=llm_model_name, temperature=temperature)
        elif llm_model_name.startswith("gpt"):
            from llms import OpenAIModel
            llm = OpenAIModel(model_name=llm_model_name, temperature=temperature)
        elif llm_model_name.startswith("llama"):
            from llms import GroqModel
            llm = GroqModel(model_name=llm_model_name, temperature=temperature)
        else:
            return {"error": "Unsupported LLM model selected."}, processed_images

        n_retries = 5
        while True:
            try:
                output = llm.prompt(processed_images + [prompt])
                output_dict = {}
                for line in output.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        output_dict[k.strip()] = v.strip()
                return output_dict, processed_images
            except Exception as e:
                if n_retries > 0:
                    n_retries -= 1
                    print(f"LLM error: {str(e)}, retrying... {n_retries} retries left. Retrying in 60 seconds.")
                    time.sleep(60)  # Wait before retrying
                else:
                    return {"error": f"LLM error after all retries: {str(e)}"}, processed_images

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, [image] if image else []

def process_batch(
    images,
    prompt,
    use_grounding_dino,
    box_threshold,
    text_threshold,
    grounding_prompt,
    llm_model_name,
    resize_size,
    temperature,
    output_format,
    config,
    progress = gr.Progress(),
):
    if not images:
        return {"error": "No images provided"}, None, None

    results = []
    processed_images = []

    for idx, image in enumerate(progress.tqdm(images, desc="Processing images")):
        try:
            image = Image.open(image.name).convert("RGB")

            #progress.update(f"Processing image {idx + 1}/{len(images)}")

            result, imgs = process_image(
                image,
                prompt,
                use_grounding_dino,
                box_threshold,
                text_threshold,
                grounding_prompt,
                llm_model_name,
                resize_size,
                temperature,
                config,
            )
            results.append(result)
            processed_images.extend(imgs)
        except Exception as e:
            results.append({"error": f"Error processing image: {str(e)}"})
            print(f"Error processing image {idx + 1}: {e}")
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
