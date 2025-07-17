import pandas as pd
import argparse
from typing import Callable, Dict
from difflib import SequenceMatcher

def default_string_distance(a: str, b: str) -> float:
    # Returns similarity ratio (1.0 = identical, 0.0 = completely different)
    return SequenceMatcher(None, str(a), str(b)).ratio()

def compare_species_name(extracted: str, gt: str) -> float:
    # Compare only the two first words of the species name
    extracted_words = extracted.split()
    gt_words = gt.split()

    extracted = " ".join(extracted_words[:2])
    gt = " ".join(gt_words[:2])

    return default_string_distance(extracted, gt)

def compare_collection_date(extracted: str, gt: str) -> float:
    # Only compare the year
    extracted_year = extracted.split("-")[0] # "2023-08-17" -> "2023"
    gt_year = gt.split(" ")[-1] # "17. August 1951" -> "1951"

    return default_string_distance(extracted_year, gt_year)

def compare_collectors_name(extracted: str, gt: str) -> float:
    # Only compare the last name
    extracted_last_name = extracted.split()[-1]  # Get the last word as last name
    gt_last_name = gt.split()[-1]  # Get the last word as last name

    return default_string_distance(extracted_last_name, gt_last_name)

def compare_location(extracted: str, gt: str) -> float:
    # Compare complete location strings, but format gt similarly to extracted

    return default_string_distance(extracted, gt)


metric_fns = {
    "Species name": compare_species_name,
    "Collection date": compare_collection_date,
    "Collector's name": compare_collectors_name,
    #"Country/State": default_string_distance,  # Excluded, because not contained in every image
    "Location": compare_location,
    "Region": default_string_distance,  # Assuming region is a simple string comparison
}

def compare_tables(
    extracted_csv: str,
    ground_truth_csv: str,
    column_map: Dict[str, str],
    output_csv: str = None,
):
    # Load CSVs
    extracted = pd.read_csv(extracted_csv)
    ground_truth = pd.read_csv(ground_truth_csv)

    # Index both tables by source_image
    extracted = extracted.set_index("source_image")
    ground_truth = ground_truth.set_index(column_map["source_image"])

    column_map = column_map.copy()

    del column_map["source_image"]  # Remove source_image from mapping

    # Filter columns
    extracted_cols = list(column_map.keys())
    ground_truth_cols = list(column_map.values())

    # Only keep rows present in both
    common_images = extracted.index.intersection(ground_truth.index)
    extracted = extracted.loc[common_images, extracted_cols]
    ground_truth = ground_truth.loc[common_images, ground_truth_cols]
    ground_truth.columns = extracted_cols  # Rename for comparison

    # Compute metrics
    results = {}

    for col in metric_fns:
        results[f"{col}_similarity"] = extracted[col].combine(ground_truth[col], metric_fns[col]).mean()

    df_results = pd.Series(results)
    df_results["mean"] = df_results.mean()
    if output_csv:
        df_results.to_csv(output_csv, index=False)
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare extracted herbarium data with ground truth.")
    parser.add_argument("--extracted_csv", required=True, help="Path to extracted data CSV.")
    parser.add_argument("--ground_truth_csv", required=True, help="Path to ground truth CSV.")
    parser.add_argument("--output_csv", default=None, help="Path to save comparison results.")
    args = parser.parse_args()

    # Parse column_map argument
    column_map = {
        "source_image": "Bildname",
        "Species name": "Wissenschaftlicher Name",
        "Collection date": "Sammeldatum",
        "Collector's name": "Sammler",
        "Country/State": "Geographische_Zuordnung",
        "Location": "Fundort",
        "Region": "Naturraum",
    }

    df = compare_tables(
        args.extracted_csv,
        args.ground_truth_csv,
        column_map,
        output_csv=args.output_csv,
    )
    print(df)
