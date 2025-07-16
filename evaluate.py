import pandas as pd
import argparse
from typing import Callable, Dict
from difflib import SequenceMatcher

def default_string_distance(a: str, b: str) -> float:
    # Returns similarity ratio (1.0 = identical, 0.0 = completely different)
    return SequenceMatcher(None, str(a), str(b)).ratio()

def compare_tables(
    extracted_csv: str,
    ground_truth_csv: str,
    column_map: Dict[str, str],
    metric_fn: Callable[[str, str], float] = default_string_distance,
    output_csv: str = None,
):
    # Load CSVs
    extracted = pd.read_csv(extracted_csv)
    ground_truth = pd.read_csv(ground_truth_csv)

    # Index both tables by source_image
    extracted = extracted.set_index("source_image")
    ground_truth = ground_truth.set_index("source_image")

    # Filter columns
    extracted_cols = list(column_map.keys())
    ground_truth_cols = list(column_map.values())

    # Only keep rows present in both
    common_images = extracted.index.intersection(ground_truth.index)
    extracted = extracted.loc[common_images, extracted_cols]
    ground_truth = ground_truth.loc[common_images, ground_truth_cols]
    ground_truth.columns = extracted_cols  # Rename for comparison

    # Compute metrics
    results = []
    for img in common_images:
        row = {"source_image": img}
        for col in extracted_cols:
            val1 = extracted.at[img, col]
            val2 = ground_truth.at[img, col]
            row[f"{col}_score"] = metric_fn(val1, val2)
            row[f"{col}_extracted"] = val1
            row[f"{col}_ground_truth"] = val2
        results.append(row)

    df_results = pd.DataFrame(results)
    if output_csv:
        df_results.to_csv(output_csv, index=False)
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare extracted herbarium data with ground truth.")
    parser.add_argument("--extracted_csv", required=True, help="Path to extracted data CSV.")
    parser.add_argument("--ground_truth_csv", required=True, help="Path to ground truth CSV.")
    parser.add_argument("--output_csv", default=None, help="Path to save comparison results.")
    parser.add_argument("--column_map", required=True, help="Column mapping as comma-separated pairs, e.g. 'Species name:species,Collector's name:collector'")
    args = parser.parse_args()

    # Parse column_map argument
    column_map = dict(pair.split(":") for pair in args.column_map.split(","))

    df = compare_tables(
        args.extracted_csv,
        args.ground_truth_csv,
        column_map,
        output_csv=args.output_csv,
    )
    print(df)
