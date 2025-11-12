import os
import re
import time
from typing import List, Dict, Optional, Union

import pandas as pd
from pathlib import Path
from PIL import Image
from omegaconf import DictConfig

from llms import GeminiModel, OpenAIModel, GroqModel
from preprocessors import GroundingDinoPreprocessor


task_parser = re.compile(r".*Task (\d+).*")


def maybe_resize(img: Image.Image, max_size: int) -> Image.Image:
    """Resize the image if it exceeds the max size."""
    if img.size[0] > max_size or img.size[1] > max_size:
        img.thumbnail((max_size, max_size))
    return img


class ExtractionPipeline:
    """Object-oriented extraction pipeline that handles preprocessing, LLM initialization, and result parsing."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the ExtractionPipeline with a configuration object.

        Args:
            cfg: DictConfig containing all settings for preprocessors, LLM, batch size, etc.
        """
        self.cfg = cfg
        self.preprocessor = None
        self.llm = None

        self.initialize_preprocessor()
        self.initialize_llm()

    def initialize_preprocessor(self) -> Optional[GroundingDinoPreprocessor]:
        """
        Initialize the preprocessor (Grounding DINO) based on configuration.

        Returns:
            GroundingDinoPreprocessor instance if enabled, else None
        """
        if (getattr(self.cfg, "preprocessors", None) and
            getattr(self.cfg.preprocessors, "grounding_dino", None) and
            self.cfg.preprocessors.grounding_dino.enabled):

            gd_cfg = self.cfg.preprocessors.grounding_dino
            self.preprocessor = GroundingDinoPreprocessor(
                model_name=gd_cfg.model_name,
                box_threshold=gd_cfg.box_threshold,
                text_threshold=gd_cfg.text_threshold,
                device=getattr(gd_cfg, "device", "cpu"),
                multi_output=True,
                max_outputs=getattr(gd_cfg, "max_outputs", 5),
            )
        return self.preprocessor

    def initialize_llm(self) -> object:
        """
        Initialize the LLM model based on configuration.

        This method is flexible and tries multiple constructor signatures to handle
        different model implementations (Gemini, OpenAI, Groq).

        Returns:
            LLM instance (GeminiModel, OpenAIModel, or GroqModel)

        Raises:
            ValueError: If model_name is not set or unsupported
        """
        model_name = self.cfg.llm.model_name if getattr(self.cfg, "llm", None) else None
        if model_name is None:
            raise ValueError("cfg.llm.model_name is required to initialize LLM")

        def _try_construct(model_cls, **kwargs):
            """Try to construct model with given kwargs, fall back to minimal args."""
            try:
                return model_cls(**kwargs)
            except TypeError:
                # Fall back to minimal constructor
                allowed = {}
                if "model_name" in kwargs:
                    allowed["model_name"] = kwargs["model_name"]
                if "rate_limit_wait" in kwargs:
                    allowed["rate_limit_wait"] = kwargs["rate_limit_wait"]
                if "temperature" in kwargs:
                    allowed["temperature"] = kwargs["temperature"]
                return model_cls(**allowed)

        # Build kwargs from config
        kwargs = {"model_name": model_name}
        if getattr(self.cfg, "rate_limit_wait", None) is not None:
            kwargs["rate_limit_wait"] = self.cfg.rate_limit_wait
        if getattr(self.cfg.llm, "temperature", None) is not None:
            kwargs["temperature"] = self.cfg.llm.temperature

        # Instantiate the appropriate model
        if model_name.startswith("gemini"):
            self.llm = _try_construct(GeminiModel, **kwargs)
        elif model_name.startswith("gpt"):
            self.llm = _try_construct(OpenAIModel, **kwargs)
        elif model_name.startswith("llama"):
            self.llm = _try_construct(GroqModel, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM model: {model_name}")

        return self.llm

    def preprocess_image(
        self,
        image: Image.Image,
        image_path: str,
        output_dir: str,
    ) -> List[Image.Image]:
        """
        Preprocess a single image using the configured preprocessor and resize settings.

        Args:
            image: PIL Image object
            image_path: Path to the image file for reference
            output_dir: Directory to save preprocessed images if logging is enabled

        Returns:
            List of preprocessed images
        """
        if self.preprocessor is not None:
            image = self.preprocessor.preprocess(image, self.cfg.preprocessors.grounding_dino.prompt)

            if not isinstance(image, list):
                image = [image]

            if getattr(self.cfg.preprocessors.grounding_dino, "log_output", False):
                image_log_path = os.path.join(output_dir, "dino")
                os.makedirs(image_log_path, exist_ok=True)
                for idx, img in enumerate(image):
                    output_image_path = os.path.join(image_log_path, f"{os.path.basename(image_path)}_{idx}.jpg")
                    img.save(output_image_path)
        else:
            image = [image]

        image = [maybe_resize(img, self.cfg.img_max_size) for img in image]
        return image

    def build_prompt(self, images: List[List[Image.Image]]) -> List:
        """
        Build the prompt for the LLM including system prompt and batch information.

        Args:
            images: List of image lists (one per task in batch)

        Returns:
            List containing prompt components and images
        """
        prompt = [self.cfg.llm.prompt]
        if getattr(self.cfg, "batch_size", 1) > 1:
            prompt.append(self.cfg.batch_prompt)

        for i, img_set in enumerate(images):
            if getattr(self.cfg, "batch_size", 1) > 1:
                prompt.append(f"\n\nTask {i + 1}")
            prompt.extend(img_set)

        return prompt

    def parse_llm_output(self, outputs: str, batch_image_paths: List[str], num_images: int, add_image_names=True) -> List[Dict[str, str]]:
        """
        Parse the LLM output into structured dictionaries.

        Args:
            outputs: Raw string output from the LLM
            batch_image_paths: List of image paths (or provided image names) in the current batch
            num_images: Number of images/tasks expected in the batch
            add_image_names (bool): If True, include a "source_image" key mapping to the image name/path

        Returns:
            List[Dict[str, str]]: List of dictionaries with extracted information for each task/image
        """
        sub_results = []
        batch_idx = 0
        output_blocks = outputs.split("\n\n")

        if len(output_blocks) > num_images:
            output_blocks = output_blocks[:num_images]

        for output in output_blocks:
            output = output.strip()
            lines = output.split("\n")
            if not lines:
                continue

            match = task_parser.match(lines[0])
            if match:
                batch_idx = int(match.group(1)) - 1
                lines = lines[1:]

            image_path = batch_image_paths[batch_idx]
            if add_image_names:
                output_dict = {"source_image": image_path}
            else:
                output_dict = {}

            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    output_dict[k.strip()] = v.strip()

            sub_results.append(output_dict)
            batch_idx += 1

        return sub_results

    def save_results(self, results: List[Dict[str, str]], output_csv: str) -> None:
        """
        Save results to a CSV file.

        Args:
            results: List of result dictionaries
            output_csv: Path to save the CSV file
        """
        if not results:
            # Create an empty DataFrame with a default column so pandas can read it back
            df = pd.DataFrame(columns=["source_image"])
        else:
            df = pd.DataFrame(results)

        df.to_csv(output_csv, index=False)


    def __call__(self, images: List[Image.Image], image_names=None, add_image_names=True, on_error_fn=None):
        """
        Process an image or list of images through the pipeline and return parsed output.

        Args:
            images: A single PIL Image, a path (str/Path), or a list of such items.
            image_names (Optional[List[str]]): Optional list of names to associate with each image
            add_image_names (bool): If True, parsed dictionaries will include a "source_image" key
            on_error_fn (Optional[Callable[[Exception, str], None]]): Optional function to call on errors with signature (Exception, str)

        Returns:
            Tuple[List[Dict[str, str]], List[PIL.Image.Image]]: A tuple where the first element is a list of
            parsed result dictionaries (one per task/image) and the second is a flat list of preprocessed images
            produced by the pipeline.
        """
        # Normalize input into a list
        if images is None:
            return []

        if not isinstance(images, (list, tuple)):
            images = [images]

        # Convert string/Path inputs to PIL Images and capture names
        prepared_inputs: List[Dict[str, Union[Image.Image, str]]] = []
        for idx, item in enumerate(images):
            if isinstance(item, (str, Path)):
                img_path = str(item)
                img = Image.open(img_path).convert("RGB")
                name = os.path.basename(img_path)
            elif isinstance(item, Image.Image):
                img = item
                name = getattr(item, "filename", None) or f"image{idx}"
            else:
                # Attempt to handle file-like objects with a 'name' attribute
                name = getattr(item, "name", f"image{idx}")
                try:
                    img = Image.open(item).convert("RGB")
                except Exception:
                    raise TypeError("Unsupported image input type for ExtractionPipeline.__call__")

            img_name = name if image_names is None else image_names[idx]

            prepared_inputs.append({"image": img, "name": img_name})

        # Ensure models are initialized
        if self.preprocessor is None:
            self.initialize_preprocessor()
        if self.llm is None:
            self.initialize_llm()

        temp_out = os.path.join(os.getcwd(), "tmp")
        os.makedirs(temp_out, exist_ok=True)

        batch_size = getattr(self.cfg, "batch_size", 1) or 1
        results: List[Dict[str, str]] = []
        preprocessed_images = []

        # Process in batches
        for start in range(0, len(prepared_inputs), batch_size):
            batch = prepared_inputs[start : start + batch_size]

            # Preprocess each input in the batch
            batch_image_names: List[str] = []
            batch_images_for_prompt: List[List[Image.Image]] = []
            for entry in batch:
                img = entry["image"]
                name = entry["name"]
                preprocessed = self.preprocess_image(img, name, temp_out)
                batch_images_for_prompt.append(preprocessed)
                batch_image_names.append(name)
                preprocessed_images.extend(preprocessed)

            # Build prompt for the batch
            prompt = self.build_prompt(batch_images_for_prompt)

            # LLM call with retries
            n_retries = getattr(self.cfg, "n_retries", None)
            if n_retries is None:
                n_retries = 5

            while True:
                try:
                    outputs = self.llm.prompt(prompt, on_error_fn=on_error_fn)
                    parsed = self.parse_llm_output(
                        outputs,
                        batch_image_names,
                        len(batch_images_for_prompt),
                        add_image_names=add_image_names,
                    )
                    results.extend(parsed)
                    break
                except Exception as e:
                    n_retries -= 1
                    if n_retries > 0:
                        print(f"Error during LLM call: {e}. Retrying... ({n_retries} attempts left)")
                        if on_error_fn:
                            on_error_fn(e, f"Error during LLM call: {e}. Retrying... ({n_retries} attempts left)")
                        # Backoff before retrying
                        time.sleep(1)
                        continue
                    else:
                        # Raise if out of retries or we shouldn't wait
                        raise

        return results, preprocessed_images
