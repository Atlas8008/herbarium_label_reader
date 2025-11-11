import os
import math
import hydra
import hashlib
import pandas as pd

from PIL import Image
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict

# Centralized extraction pipeline
from utils.extract_utils import ExtractionPipeline


def load_image_paths(cfg) -> List[str]:
    """
    Load image paths from the configured image list file.

    Args:
        cfg: configuration object with `image_list`, `image_index`, and `n_images` attributes

    Returns:
        List[str]: image path strings (relative or as listed in the file)
    """
    category = os.path.splitext(os.path.basename(cfg.image_list))[0]

    with open(cfg.image_list, "r") as f:
        image_paths = f.read().strip().split("\n")[
            cfg.image_index : cfg.image_index + cfg.n_images
        ]
    return [os.path.join(category, p) for p in image_paths]

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

    # Initialize the extraction pipeline with config
    pipeline = ExtractionPipeline(cfg)

    # Load image paths
    image_paths = load_image_paths(cfg)

    results = []

    print(f"\nStarting processing for {len(image_paths)} images...")
    print(image_paths)

    # Process images in batches using the pipeline __call__
    for batch_idx in range(0, int(math.ceil(len(image_paths) / cfg.batch_size))):
        batch_image_paths = image_paths[batch_idx * cfg.batch_size:(batch_idx + 1) * cfg.batch_size]

        # Full paths for the pipeline (it accepts paths or PIL images)
        batch_full_paths = [os.path.join(cfg.dataset_path, p) for p in batch_image_paths]

        print(f"\nProcessing batch {batch_idx + 1}: {batch_image_paths}")

        try:
            sub_results = pipeline(batch_full_paths)
            results.extend(sub_results)
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            return

    # Save results to CSV
    pipeline.save_results(results, output_csv)


if __name__ == "__main__":
    main()
