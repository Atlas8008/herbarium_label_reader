import os
import re
import math
import hydra
import hashlib
import pandas as pd

from PIL import Image
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from llms import GeminiModel, OpenAIModel, GroqModel
from preprocessors import GroundingDinoPreprocessor


task_parser = re.compile(r".*Task (\d+).*")

def maybe_resize(img, max_size):
    """
    Resize the image if it exceeds the max size.
    """
    if img.size[0] > max_size or img.size[1] > max_size:
        img.thumbnail((max_size, max_size))

    return img

# Suppress a specific PIL warning about decompression bombs
Image.MAX_IMAGE_PIXELS = None

OmegaConf.register_new_resolver("sanitize", lambda x: x.replace(" ", "_").replace("/", ".").replace("\\", ""))
OmegaConf.register_new_resolver("shorthash", lambda s: hashlib.md5(s.encode()).hexdigest()[:8])

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Load environment variables
    load_dotenv()

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "extracted_data.csv")

    if os.path.exists(output_csv):
        print(f"Output file {output_csv} already exists. Exiting.")
        return

    preprocessor = None

    print("Loading models...")
    if cfg.preprocessors.grounding_dino.enabled:
        gd_cfg = cfg.preprocessors.grounding_dino
        preprocessor = GroundingDinoPreprocessor(
            model_name=gd_cfg.model_name,
            box_threshold=gd_cfg.box_threshold,
            text_threshold=gd_cfg.text_threshold,
            device=gd_cfg.device,
            multi_output=True,
            max_outputs=gd_cfg.max_outputs,
        )

    if cfg.llm.model_name.startswith("gemini"):
        llm = GeminiModel(
            model_name=cfg.llm.model_name,
            rate_limit_wait=cfg.rate_limit_wait,
        )
    elif cfg.llm.model_name.startswith("gpt"):
        llm = OpenAIModel(
            model_name=cfg.llm.model_name,
            rate_limit_wait=cfg.rate_limit_wait,
        )
    elif cfg.llm.model_name.startswith("llama"):
        llm = GroqModel(
            model_name=cfg.llm.model_name,
            rate_limit_wait=cfg.rate_limit_wait,
        )
    else:
        raise ValueError(f"Unsupported LLM model: {cfg.llm.model_name}")

    # Read image paths from the input file
    with open(cfg.image_list, "r") as f:
        image_paths = f.read().strip().split("\n")[cfg.image_index:cfg.image_index + cfg.n_images]

    results = []

    print(f"\nStarting processing for {len(image_paths)} images...")
    print(image_paths)

    for batch_idx in range(0, int(math.ceil(len(image_paths) / cfg.batch_size))):
        images = []

        batch_image_paths = image_paths[batch_idx * cfg.batch_size:(batch_idx + 1) * cfg.batch_size]

        for i, image_path in enumerate(batch_image_paths):
            #print(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path}")
            print(f"\nProcessing: {image_path}")

            image = Image.open(os.path.join(cfg.dataset_path, image_path)).convert("RGB")

            if preprocessor is not None:
                image = preprocessor.preprocess(image, cfg.preprocessors.grounding_dino.prompt)

                if not isinstance(image, list):
                    image = [image]  # Ensure image is a list for consistent handling

                if cfg.preprocessors.grounding_dino.log_output:
                    image_log_path = os.path.join(output_dir, "dino")
                    os.makedirs(image_log_path, exist_ok=True)


                    for idx, img in enumerate(image):
                        output_image_path = os.path.join(image_log_path, f"{os.path.basename(image_path)}_{idx}.jpg")
                        img.save(output_image_path)
                        print(f"Preprocessed image saved to: {output_image_path}")
            else:
                image = [image]

            image = [maybe_resize(img, cfg.img_max_size) for img in image]

            images.append(image)

        prompt = [cfg.llm.prompt]

        if cfg.batch_size > 1:
            prompt.append(cfg.batch_prompt)

        for i, img_set in enumerate(images):
            if cfg.batch_size > 1:
                prompt.append(f"\n\nTask {i + 1}")

            prompt.extend(img_set)

        n_retries = 20

        while True:
            try:
                outputs = llm.prompt(
                    prompt,
                )

                print("Full LLM output: ", outputs)

                sub_results = []
                batch_idx = 0 # Keep track of the current batch index, in case the model specifies a task number

                output_blocks = outputs.split("\n\n")

                # If there are more output blocks than images, we just take the first len(images) blocks
                if len(output_blocks) > len(images):
                    output_blocks = output_blocks[:len(images)]

                for output in output_blocks:
                    output = output.strip()

                    print(f"LLM output: {output}")

                    lines = output.split("\n")

                    # Check, if model specified a task number
                    match = task_parser.match(lines[0])
                    if match: # If the model specified a task number
                        # Use the specified batch index
                        # This allows the model to specify which task it is processing
                        batch_idx = int(match.group(1)) - 1
                        print(f"Specified batch index: {batch_idx}")
                        lines = lines[1:]  # Skip the first line

                    image_path = batch_image_paths[batch_idx]

                    output_dict = {
                        "source_image": image_path,
                    }

                    for line in lines:
                        k, v = line.split(":", 1)

                        k = k.strip()
                        v = v.strip()

                        output_dict[k] = v

                    sub_results.append(output_dict)
                    batch_idx += 1

                results.extend(sub_results)
                break  # Exit retry loop if successful
            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {e}")
                if n_retries > 0:
                    n_retries -= 1
                    print(f"Retrying... {n_retries} retries left.")
                else:
                    print("Max retries reached.")
                    print(e) # Re-raise the exception after max retries
                    return

    # Final Step: Create a DataFrame and save to CSV
    df = pd.DataFrame(results)

    df.to_csv(output_csv, index=False)
    print(f"\nProcessing complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()