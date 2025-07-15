import os
import hydra
import torch
import argparse
import pandas as pd

from PIL import Image
from omegaconf import DictConfig

from llms import GeminiModel
from preprocessors import GroundingDinoPreprocessor

# Suppress a specific PIL warning about decompression bombs
Image.MAX_IMAGE_PIXELS = None



def analyze_with_gemini(image, prompt, gemini_model):
    """Sends an image and prompt to Gemini and returns the text response."""
    print("  -> Analyzing with Gemini...")
    try:
        response = gemini_model.generate_content([prompt, image])
        # Handle cases where the response might be blocked
        if not response.parts:
             print("  -> Gemini response was empty or blocked.")
             return "Error: Gemini response blocked"
        return response.text
    except Exception as e:
        print(f"  -> An error occurred with the Gemini API: {e}")
        return f"Error: {e}"

def parse_gemini_response(text_response):
    """
    Parses a key-value string from Gemini into a dictionary.
    Assumes format: "Key1: Value1, Key2: Value2, ..."
    """
    parsed_data = {}
    if "Error:" in text_response:
        parsed_data['gemini_error'] = text_response
        return parsed_data

    try:
        parts = [p.strip() for p in text_response.split(',')]
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                parsed_data[key.strip()] = value.strip()
    except Exception as e:
        print(f"  -> Could not parse Gemini response: '{text_response}'. Storing raw text.")
        parsed_data['raw_gemini_response'] = text_response
    return parsed_data

@hydra.main(version_base=None, config_path="config", config_name="main_config")
def main(cfg: DictConfig):
    preprocessor = None

    print("Loading models...")
    if cfg.preprocessors.grounding_dino.enabled:
        gd_cfg = cfg.preprocessors.grounding_dino
        preprocessor = GroundingDinoPreprocessor(
            model_name=gd_cfg.model_name,
            box_threshold=gd_cfg.box_threshold,
            text_threshold=gd_cfg.text_threshold,
            device=gd_cfg.device,
        )

    if cfg.llm.model_name.startswith('gemini'):
        llm = GeminiModel(model_name=cfg.llm.model_name)
    else:
        raise ValueError(f"Unsupported LLM model: {cfg.llm.model_name}")

    # Read image paths from the input file
    with open(cfg.image_list, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    all_results = []

    print(f"\nStarting processing for {len(image_paths)} images...")

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if preprocessor is not None:
            image = preprocessor.preprocess(image, cfg.preprocessors.grounding_dino.prompt)


        # Step 2: Analyze with Gemini
        gemini_text = analyze_with_gemini(processed_image, args.gemini_prompt, gemini_model)

        # Step 3: Parse the Gemini response
        parsed_results = parse_gemini_response(gemini_text)

        # Step 4: Collate all data for this image
        final_row = {
            'source_image': image_path,
            'dino_prompt': args.dino_prompt or 'N/A',
            'bounding_box_found': str(bounding_box) if bounding_box else 'None',
        }
        final_row.update(parsed_results)
        all_results.append(final_row)

    # Final Step: Create a DataFrame and save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        # Reorder columns to have source image first
        cols = ['source_image', 'dino_prompt', 'bounding_box_found'] + [c for c in df.columns if c not in ['source_image', 'dino_prompt', 'bounding_box_found']]
        df = df[cols]

        df.to_csv(args.output_csv, index=False)
        print(f"\nProcessing complete. Results saved to {args.output_csv}")
    else:
        print("\nNo images were processed.")

if __name__ == "__main__":
    main()