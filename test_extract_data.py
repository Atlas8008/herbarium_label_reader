"""
Unit tests for ExtractionPipeline class and utilities.

Tests the ExtractionPipeline class:
- __init__
- initialize_preprocessor
- initialize_llm
- load_image_paths
- preprocess_image
- build_prompt
- parse_llm_output
- save_results

And utility functions:
- maybe_resize
- task_parser
"""

import os
import re
import tempfile
import unittest
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

import pandas as pd
from PIL import Image
from omegaconf import OmegaConf

from utils.extract_utils import (
    ExtractionPipeline,
    maybe_resize,
    task_parser,
)
from extract_data import load_image_paths

load_dotenv()

class TestMaybeResize(unittest.TestCase):
    """Test the maybe_resize helper function."""

    def test_resize_image_exceeds_max_size(self):
        """Test that oversized images are resized."""
        # Create a large image
        large_image = Image.new("RGB", (5000, 5000), color="red")
        max_size = 4096

        resized = maybe_resize(large_image, max_size)

        # Check that the image was resized
        self.assertLessEqual(resized.size[0], max_size)
        self.assertLessEqual(resized.size[1], max_size)

    def test_resize_image_within_max_size(self):
        """Test that properly sized images are not resized."""
        small_image = Image.new("RGB", (2048, 2048), color="blue")
        max_size = 4096

        resized = maybe_resize(small_image, max_size)

        # Check that the image was not resized
        self.assertEqual(resized.size[0], 2048)
        self.assertEqual(resized.size[1], 2048)

    def test_resize_maintains_aspect_ratio(self):
        """Test that resizing maintains aspect ratio."""
        wide_image = Image.new("RGB", (5000, 2500), color="green")
        max_size = 4096

        resized = maybe_resize(wide_image, max_size)

        # Check aspect ratio is preserved (approximately)
        original_ratio = 5000 / 2500
        resized_ratio = resized.size[0] / resized.size[1]

        self.assertAlmostEqual(original_ratio, resized_ratio, places=1)


class TestTaskParser(unittest.TestCase):
    """Test the task_parser regex."""

    def test_parse_task_number_valid(self):
        """Test parsing valid task numbers."""
        match = task_parser.match("Task 1: Some content")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "1")

    def test_parse_task_number_larger(self):
        """Test parsing larger task numbers."""
        match = task_parser.match("Task 42: Content here")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "42")

    def test_parse_task_number_no_match(self):
        """Test non-matching strings."""
        match = task_parser.match("This is not a task")
        self.assertIsNone(match)

    def test_parse_task_number_prefix(self):
        """Test parsing task numbers with prefix text."""
        match = task_parser.match("Processing Task 5 now")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "5")


class TestParseLLMOutput(unittest.TestCase):
    """Test the parse_llm_output method on ExtractionPipeline."""

    def test_parse_simple_output(self):
        """Test parsing simple LLM output."""
        llm_output = (
            "Species name: Rosa canina\n"
            "Collection date: 2023-05-15\n"
            "Collector's name: John Doe"
        )
        batch_image_paths = ["image1.jpg"]

        pipeline = object.__new__(ExtractionPipeline)
        results = ExtractionPipeline.parse_llm_output(pipeline, llm_output, batch_image_paths, 1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_image"], "image1.jpg")
        self.assertEqual(results[0]["Species name"], "Rosa canina")
        self.assertEqual(results[0]["Collection date"], "2023-05-15")
        self.assertEqual(results[0]["Collector's name"], "John Doe")

    def test_parse_multiple_outputs(self):
        """Test parsing multiple task outputs separated by double newlines."""
        llm_output = (
            "Species name: Rosa canina\n"
            "Collection date: 2023-05-15\n"
            "\n"
            "Species name: Quercus robur\n"
            "Collection date: 2023-06-20"
        )
        batch_image_paths = ["image1.jpg", "image2.jpg"]

        pipeline = object.__new__(ExtractionPipeline)
        results = ExtractionPipeline.parse_llm_output(pipeline, llm_output, batch_image_paths, 2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["Species name"], "Rosa canina")
        self.assertEqual(results[1]["Species name"], "Quercus robur")

    def test_parse_with_task_numbers(self):
        """Test parsing output where model specifies task numbers."""
        llm_output = (
            "Task 2: Species name: Quercus robur\n"
            "Collection date: 2023-06-20\n"
            "\n"
            "Task 1: Species name: Rosa canina\n"
            "Collection date: 2023-05-15"
        )
        batch_image_paths = ["image1.jpg", "image2.jpg"]

        pipeline = object.__new__(ExtractionPipeline)
        results = ExtractionPipeline.parse_llm_output(pipeline, llm_output, batch_image_paths, 2)

        # Results should be in the order specified by task numbers
        self.assertEqual(len(results), 2)
        # First result should be from task 2
        self.assertEqual(results[0]["source_image"], "image2.jpg")
        # Second result should be from task 1
        self.assertEqual(results[1]["source_image"], "image1.jpg")

    def test_parse_output_with_colons_in_values(self):
        """Test parsing output where values contain colons."""
        llm_output = "Notes: Time collected: 14:30 (afternoon)"
        batch_image_paths = ["image1.jpg"]

        pipeline = object.__new__(ExtractionPipeline)
        results = ExtractionPipeline.parse_llm_output(pipeline, llm_output, batch_image_paths, 1)

        self.assertEqual(results[0]["Notes"], "Time collected: 14:30 (afternoon)")

    def test_parse_empty_values(self):
        """Test parsing output with empty values."""
        llm_output = (
            "Species name: Rosa canina\n"
            "Location: \n"
            "Notes:"
        )
        batch_image_paths = ["image1.jpg"]

        pipeline = object.__new__(ExtractionPipeline)
        results = ExtractionPipeline.parse_llm_output(pipeline, llm_output, batch_image_paths, 1)

        self.assertEqual(results[0]["Species name"], "Rosa canina")
        self.assertEqual(results[0]["Location"], "")
        self.assertEqual(results[0]["Notes"], "")

    def test_parse_excess_output_blocks(self):
        """Test that excess output blocks are trimmed."""
        llm_output = (
            "Species name: Rosa canina\n"
            "\n"
            "Species name: Quercus robur\n"
            "\n"
            "Species name: Extra output"
        )
        batch_image_paths = ["image1.jpg", "image2.jpg"]

        pipeline = object.__new__(ExtractionPipeline)
        results = ExtractionPipeline.parse_llm_output(pipeline, llm_output, batch_image_paths, 2)

        # Should only return 2 results despite 3 output blocks
        self.assertEqual(len(results), 2)


class TestBuildPrompt(unittest.TestCase):
    """Test the build_prompt method of ExtractionPipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_cfg = OmegaConf.create({
            "llm": {
                "model_name": "gemini-2.5-pro",
                "prompt": "You are an expert.",
            },
            "batch_size": 1,
            "batch_prompt": "Process this batch.",
        })
        # Avoid initializing real LLMs during tests
        self.pipeline = object.__new__(ExtractionPipeline)
        self.pipeline.cfg = self.mock_cfg

    def test_build_single_image_prompt(self):
        """Test building prompt with single image."""
        image = Image.new("RGB", (100, 100), color="red")
        images = [[image]]

        result = self.pipeline.build_prompt(images)

        self.assertIn("You are an expert.", result)
        self.assertEqual(result[1], image)

    def test_build_batch_prompt_with_multiple_images(self):
        """Test building batch prompt with multiple images."""
        self.pipeline.cfg.batch_size = 2
        image1 = Image.new("RGB", (100, 100), color="red")
        image2 = Image.new("RGB", (100, 100), color="blue")
        images = [[image1], [image2]]

        result = self.pipeline.build_prompt(images)

        # Should have system prompt, batch prompt, and task labels
        self.assertIn("You are an expert.", result)
        self.assertIn("Process this batch.", result)
        self.assertIn("\n\nTask 1", result)
        self.assertIn("\n\nTask 2", result)

    def test_build_prompt_with_single_batch_size(self):
        """Test that batch prompt is not included when batch_size is 1."""
        self.pipeline.cfg.batch_size = 1
        image = Image.new("RGB", (100, 100), color="red")
        images = [[image]]

        result = self.pipeline.build_prompt(images)

        self.assertNotIn("Process this batch.", result)


class TestSaveResults(unittest.TestCase):
    """Test the save_results method of ExtractionPipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_cfg = OmegaConf.create({
            "llm": {"model_name": "gemini-2.5-pro"}
        })
        # Avoid initializing real LLMs during tests
        self.pipeline = object.__new__(ExtractionPipeline)
        self.pipeline.cfg = self.mock_cfg

    def test_save_results_to_csv(self):
        """Test saving results to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, "results.csv")

            results = [
                {
                    "source_image": "image1.jpg",
                    "Species name": "Rosa canina",
                    "Collection date": "2023-05-15",
                },
                {
                    "source_image": "image2.jpg",
                    "Species name": "Quercus robur",
                    "Collection date": "2023-06-20",
                },
            ]

            self.pipeline.save_results(results, output_csv)

            # Check that file was created
            self.assertTrue(os.path.exists(output_csv))

            # Check that content is correct
            df = pd.read_csv(output_csv)
            self.assertEqual(len(df), 2)
            self.assertListEqual(list(df.columns), ["source_image", "Species name", "Collection date"])
            self.assertEqual(df.iloc[0]["Species name"], "Rosa canina")

    def test_save_results_empty(self):
        """Test saving empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, "results.csv")
            results = []

            self.pipeline.save_results(results, output_csv)

            # Check that file was created
            self.assertTrue(os.path.exists(output_csv))

            # Check that it's empty
            df = pd.read_csv(output_csv)
            self.assertEqual(len(df), 0)

    def test_save_results_with_special_characters(self):
        """Test saving results with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, "results.csv")

            results = [
                {
                    "source_image": "image1.jpg",
                    "Species name": "Orchidaceae spp.",
                    "Notes": "Special chars: !@#$%^&*()",
                },
            ]

            self.pipeline.save_results(results, output_csv)

            # Verify file can be read back
            df = pd.read_csv(output_csv)
            self.assertEqual(df.iloc[0]["Notes"], "Special chars: !@#$%^&*()")


class TestLoadImagePaths(unittest.TestCase):
    """Test the load_image_paths method of ExtractionPipeline."""

    def test_load_image_paths_basic(self):
        """Test loading image paths from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_list_file = os.path.join(tmpdir, "image_list.txt")

            # Create test image list
            with open(image_list_file, "w") as f:
                f.write("image1.jpg\nimage2.jpg\nimage3.jpg\n")

            mock_cfg = OmegaConf.create({
                "image_list": image_list_file,
                "image_index": 0,
                "n_images": 3,
                "llm": {"model_name": "gemini-2.5-pro"},
            })

            result = load_image_paths(mock_cfg)

            self.assertEqual(result, ["image1.jpg", "image2.jpg", "image3.jpg"])

    def test_load_image_paths_with_offset(self):
        """Test loading image paths with offset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_list_file = os.path.join(tmpdir, "image_list.txt")

            with open(image_list_file, "w") as f:
                f.write("image1.jpg\nimage2.jpg\nimage3.jpg\nimage4.jpg\n")

            mock_cfg = OmegaConf.create({
                "image_list": image_list_file,
                "image_index": 1,
                "n_images": 2,
                "llm": {"model_name": "gemini-2.5-pro"},
            })

            result = load_image_paths(mock_cfg)

            self.assertEqual(result, ["image2.jpg", "image3.jpg"])

    def test_load_image_paths_partial(self):
        """Test loading fewer images than available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_list_file = os.path.join(tmpdir, "image_list.txt")

            with open(image_list_file, "w") as f:
                f.write("image1.jpg\nimage2.jpg\nimage3.jpg\n")

            mock_cfg = OmegaConf.create({
                "image_list": image_list_file,
                "image_index": 0,
                "n_images": 2,
                "llm": {"model_name": "gemini-2.5-pro"},
            })

            result = load_image_paths(mock_cfg)

            self.assertEqual(result, ["image1.jpg", "image2.jpg"])


class TestPreprocessImage(unittest.TestCase):
    """Test the preprocess_image method of ExtractionPipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = Image.new("RGB", (100, 100), color="red")
        self.mock_cfg = OmegaConf.create({
            "img_max_size": 4096,
            "preprocessors": {
                "grounding_dino": {
                    "log_output": False,
                    "enabled": False,
                }
            },
            "llm": {
                "model_name": "gemini-2.5-pro",
            },
        })

    def test_preprocess_image_without_preprocessor(self):
        """Test preprocessing when no preprocessor is configured."""
        pipeline = object.__new__(ExtractionPipeline)
        pipeline.cfg = self.mock_cfg
        pipeline.preprocessor = None

        result = pipeline.preprocess_image(
            self.test_image,
            "test_image.jpg",
            "/tmp",
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].size, (100, 100))

    def test_preprocess_image_resizes_oversized(self):
        """Test that preprocessing resizes oversized images."""
        large_image = Image.new("RGB", (5000, 5000), color="blue")
        self.mock_cfg.img_max_size = 4096

        pipeline = object.__new__(ExtractionPipeline)
        pipeline.cfg = self.mock_cfg
        pipeline.preprocessor = None
        result = pipeline.preprocess_image(
            large_image,
            "test_image.jpg",
            "/tmp",
        )

        self.assertLessEqual(result[0].size[0], 4096)
        self.assertLessEqual(result[0].size[1], 4096)

    @patch("utils.extract_utils.GroundingDinoPreprocessor")
    def test_preprocess_image_with_preprocessor(self, mock_preprocessor_class):
        """Test preprocessing with a preprocessor enabled."""
        # Create mock preprocessor
        mock_preprocessor = MagicMock()
        processed_image = Image.new("RGB", (200, 200), color="green")
        mock_preprocessor.preprocess.return_value = processed_image

        self.mock_cfg.preprocessors.grounding_dino.enabled = True
        self.mock_cfg.preprocessors.grounding_dino.prompt = "test prompt"

        pipeline = object.__new__(ExtractionPipeline)
        pipeline.cfg = self.mock_cfg
        pipeline.preprocessor = mock_preprocessor  # Inject the mock

        result = pipeline.preprocess_image(
            self.test_image,
            "test_image.jpg",
            "/tmp",
        )

        # Verify preprocessor was called
        mock_preprocessor.preprocess.assert_called_once()
        self.assertEqual(len(result), 1)


class TestInitializePreprocessor(unittest.TestCase):
    """Test the initialize_preprocessor method of ExtractionPipeline."""

    @patch("utils.extract_utils.GroundingDinoPreprocessor")
    def test_initialize_preprocessor_enabled(self, mock_gd):
        """Test initializing preprocessor when enabled."""
        mock_cfg = OmegaConf.create({
            "preprocessors": {
                "grounding_dino": {
                    "enabled": True,
                    "model_name": "test-model",
                    "box_threshold": 0.25,
                    "text_threshold": 0.3,
                    "device": "cpu",
                    "max_outputs": 5,
                }
            }
        ,
            "llm": {"model_name": "gemini-2.5-pro"}
        })

        # Avoid running __init__ which would initialize LLMs; attach cfg to a bare instance
        pipeline = object.__new__(ExtractionPipeline)
        pipeline.cfg = mock_cfg
        pipeline.preprocessor = None
        result = pipeline.initialize_preprocessor()

        mock_gd.assert_called_once()
        self.assertIsNotNone(result)

    def test_initialize_preprocessor_disabled(self):
        """Test initializing when preprocessor is disabled."""
        mock_cfg = OmegaConf.create({
            "preprocessors": {
                "grounding_dino": {
                    "enabled": False,
                }
            }
        ,
            "llm": {"model_name": "gemini-2.5-pro"}
        })

        pipeline = ExtractionPipeline(mock_cfg)
        #pipeline.preprocessor = None
        result = pipeline.initialize_preprocessor()

        self.assertIsNone(result)


class TestInitializeLLM(unittest.TestCase):
    """Test the initialize_llm method of ExtractionPipeline."""

    @patch("utils.extract_utils.GeminiModel")
    def test_initialize_llm_gemini(self, mock_gemini):
        """Test initializing Gemini model."""
        mock_cfg = OmegaConf.create({
            "llm": {
                "model_name": "gemini-2.5-pro",
            },
            "rate_limit_wait": True,
        })

        pipeline = ExtractionPipeline(mock_cfg)
        llm = pipeline.llm

        mock_gemini.assert_called_once()
        self.assertIsNotNone(llm)

    @patch("utils.extract_utils.OpenAIModel")
    def test_initialize_llm_openai(self, mock_openai):
        """Test initializing OpenAI model."""
        mock_cfg = OmegaConf.create({
            "llm": {
                "model_name": "gpt-4",
            },
            "rate_limit_wait": True,
        })

        pipeline = ExtractionPipeline(mock_cfg)
        llm = pipeline.llm

        mock_openai.assert_called_once()
        self.assertIsNotNone(llm)

    @patch("utils.extract_utils.GroqModel")
    def test_initialize_llm_groq(self, mock_groq):
        """Test initializing Groq model."""
        mock_cfg = OmegaConf.create({
            "llm": {
                "model_name": "llama-3-8b",
            },
            "rate_limit_wait": True,
        })

        pipeline = ExtractionPipeline(mock_cfg)
        llm = pipeline.llm

        mock_groq.assert_called_once()
        self.assertIsNotNone(llm)

    def test_initialize_llm_unsupported(self):
        """Test that unsupported models raise ValueError."""
        mock_cfg = OmegaConf.create({
            "llm": {
                "model_name": "unsupported-model",
            },
            "rate_limit_wait": True,
        })

        # Create an instance without running __init__ (which would itself call initialize_llm)
        pipeline = object.__new__(ExtractionPipeline)
        pipeline.cfg = mock_cfg

        with self.assertRaises(ValueError):
            pipeline.initialize_llm()


if __name__ == "__main__":
    unittest.main()
