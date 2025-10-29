# Herbarium Label Reader

A project for digitizing herbarium specimen labels by combining modern visual grounding and large language models in a single extraction pipeline. The system uses zero-shot object localization (Grounding DINO) to find label regions and large language-and-vision models (LLVMs) to extract and structure textual information from those regions. Supported providers in the codebase include Google (Gemini via google-genai), OpenAI (GPT family), and Groq (LLaMA-family models served via the Groq API).

## Key Features

- Zero-shot label localization using Grounding DINO
- Visual + language extraction using LLMs/LLVMs (Gemini, OpenAI GPTs, LLaMA-family models served via the Groq API)
- Single-image interactive processing and bulk/batch processing
- Exportable structured outputs (CSV/JSON) and side-by-side comparison for evaluation
- Hydra-based experiment orchestration and reproducible output directories

## Project Structure

```
herbarium_label_reader/
├── app.py                 # Gradio web app (single & batch interfaces)
├── extract_data.py        # Hydra-driven experiment runner / batch extractor
├── evaluate.py            # Script to compare extracted CSVs with ground truth
├── evaluate_all.sh        # Helper for running multiple evaluations
├── run_experiments.sh     # Example hydra multirun invocation
├── webapp/                # Webapp helpers (process_request.py)
├── preprocessors/         # Preprocessor implementations (Grounding DINO)
├── llms/                  # Wrappers for Gemini/OpenAI/Groq (LLaMA via Groq API) models
├── requirements.txt
└── config.yaml            # Default hydra configuration
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys and provider credentials:
Create a `.env` file at the project root. The application and scripts call load_dotenv(), and the following environment variables are commonly required depending on which providers you use:
```
# For OpenAI models
OPENAI_API_KEY=...

# For Google Gemini (google-genai)
GEMINI_API_KEY=...

# For Groq
GROQ_API_KEY=...
```

## Running Experiments

To run experiments on a dataset of herbarium labels:

```bash
./run_experiments.sh /path/to/dataset/images /path/to/output_directory
```

Notes:
- `run_experiments.sh` invokes `extract_data.py` under hydra multirun. Outputs go into the hydra sweep directory (see config.yaml hydra.sweep.dir and printed output path).
- `extract_data.py` reads image paths from the list pointed to by config.image_list and the dataset root at config.dataset_path (see config.yaml). Adjust config.yaml or pass overrides via hydra/CLI if needed.

You can run a single extraction job (no sweep) by calling the script with hydra overrides, for example:

```bash
python extract_data.py dataset_path=/absolute/path/to/dataset image_list=data/handwritten.txt n_images=50 llm.model_name=gemini-2.5-pro
```
The script saves a CSV named `extracted_data.csv` inside the hydra-run output directory.

## Evaluating Results

The evaluation script compares extracted data against ground truth data. Use it as follows:

```bash
python evaluate.py \
    --extracted_csv /path/to/results/extracted_data.csv \
    --ground_truth_csv /path/to/ground_truth/label_data.csv \
    --output_csv /path/to/output/evaluation_results.csv
```

## Web Application

The web application provides two main interfaces for processing herbarium labels:

### Single Prediction Interface

- Upload a single image, optionally enable Grounding DINO and tune thresholds and prompt.
- The server optionally runs Grounding DINO to crop/detect label regions (controlled by checkboxes/sliders).
- A selected LLVM receives the image regions plus the prompt and returns a structured, line-wise *key: value* output.
- The UI shows:
  - JSON: structured extracted fields (key:value)
  - Gallery: processed image regions (cropped/annotated)
- Useful for interactive inspection, manual correction, and quick sampling.

### Batch Prediction Interface

- Upload multiple image files at once (multiple-file selection).
- The batch pipeline processes each file sequentially: optional Grounding DINO -> LLM prompt -> result parsing.
- Progress is shown in the UI; on completion the app returns:
  - JSON: list of extraction results for immediate inspection
  - Gallery: all processed regions across the batch
  - Downloadable file: CSV or JSON file with all results

To launch the web app:

```bash
python app.py
```

Access the application at `http://localhost:7860`

### Processing Pipeline

1. (Optional) Grounding DINO finds bounding boxes likely to contain label text.
2. Bounding boxes are cropped and passed (as images) plus the human prompt to the selected LLVM.
3. The LLVM returns structured text (line-wise key:value). The code parses these into columns.
4. Results are saved as CSV/JSON for downstream use or evaluation.

## Data Format

The extracted data is saved in CSV format with the following fields:
- Species Name: Scientific name including genus and species
- Collection Date: Date when the specimen was collected
- Location: Geographic location of collection
- Collector's Name: Name of the person who collected the specimen
- Country/State: Name of the state or country the specimen was collected
- Region: The general natural region where the specimen was found
- Notes: Additional notes found on the image or label

## License

MIT License — see LICENSE file for details.