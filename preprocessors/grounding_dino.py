import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundingDinoPreprocessor:
    def __init__(self, model_name: str, device: str = 'cuda', box_threshold: float = 0.4, text_threshold: float = 0.3, multi_output: bool = False, max_outputs: int = None):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self.device = device

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.multi_output = multi_output
        self.max_outputs = max_outputs

        self.model.to(device)

    def preprocess(self, image: Image.Image, text: str):
        print(f"  -> Running Grounding DINO with prompt: '{text}'")
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold, # Confidence threshold for the box
            text_threshold=self.text_threshold, # Confidence threshold for the label
            target_sizes=[image.size[::-1]] # (height, width)
        )

        # Find the box with the highest score
        if results[0]["scores"].numel() > 0:
            if not self.multi_output:
                best_box_index = results[0]["scores"].argmax()
                box = results[0]["boxes"][best_box_index].tolist()

                print(f"  -> Object found with score: {results[0]['scores'][best_box_index].item():.2f}")

                # Crop the image to the bounding box
                cropped_image = image.crop(box)
                return cropped_image
            else:
                print(f"  -> Found {len(results[0]['boxes'])} objects with scores above the threshold.")
                # Sort boxes by score
                sorted_boxes = sorted(
                    zip(results[0]["boxes"], results[0]["scores"]),
                    key=lambda x: x[1],
                    reverse=True,
                )
                sorted_boxes = list(zip(*sorted_boxes))[0]  # Get only the boxes

                cropped_images = []
                for i in range(len(sorted_boxes)):
                    box = sorted_boxes[i].tolist()
                    cropped_image = image.crop(box)
                    cropped_images.append(cropped_image)
                return cropped_images if self.max_outputs is None else cropped_images[:self.max_outputs]
        else:
            print("  -> No object found by Grounding DINO. Using full image.")
            return image

    def postprocess(self, outputs):
        # Implement postprocessing logic if needed
        return outputs