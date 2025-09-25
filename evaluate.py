import argparse
import numpy as np
import pandas as pd
import Levenshtein as lev

from typing import Callable, Dict
from difflib import SequenceMatcher
from functools import partial


def default_string_distance(a: str, b: str) -> float:
    # Returns similarity ratio (1.0 = identical, 0.0 = completely different)
    return SequenceMatcher(None, str(a), str(b)).ratio()

def equal(a: str, b: str):
    return a == b

def substring_match(a: str, b: str, threshold: float = 0.8) -> bool:
    """
    Returns True if any substring of b matches a with similarity above threshold.
    """
    if not a or not b:
        return False
    a = str(a)
    b = str(b)
    max_ratio = 0.0
    len_a = len(a)

    for i in range(len(b) - len_a + 1):
        sub = b[i:i + len_a]
        ratio = SequenceMatcher(None, a, sub).ratio()

        if ratio > max_ratio:
            max_ratio = ratio

            if max_ratio >= threshold:
                return True

    return max_ratio >= threshold

def substring_match_levenshtein(a: str, b: str, threshold: int=1) -> int:
    """
    Returns the minimal Levenshtein distance between 'a' and any substring of 'b' of length len(a).
    If 'a' or 'b' is empty, returns len(a) (i.e., the cost to insert all of 'a').
    """
    if not a or not b:
        return len(a) if a else 0
    a = str(a)
    b = str(b)
    len_a = len(a)
    min_dist = None

    for i in range(len(b) - len_a + 1):
        sub = b[i:i + len_a]
        dist = lev.distance(a, sub)
        if min_dist is None or dist < min_dist:
            min_dist = dist

    # If b is shorter than a, compare directly
    if min_dist is None:
        min_dist = lev.distance(a, b)

    return min_dist <= threshold

def compare_species_name(extracted: str, gt: str, dist_fn=default_string_distance) -> float:
    # Compare only the two first words of the species name
    extracted = " ".join(extracted.split(" ")[:2]) if extracted else ""
    gt = " ".join(gt.split(" ")[:2]) if gt else ""

    return dist_fn(extracted, gt)

def compare_collection_date(extracted: str, gt: str, dist_fn=default_string_distance) -> float:
    # Only compare the year
    extracted_year = str(extracted).split("-")[0] # "2023-08-17" -> "2023"
    gt_year = str(gt).split(" ")[-1] # "17. August 1951" -> "1951"

    return dist_fn(extracted_year, gt_year)

def compare_collectors_name(extracted: str, gt: str, dist_fn=default_string_distance) -> float:
    # Only compare the last name
    # Get the last word as last name
    extracted_last_name = extracted.split(" ")[-1] if extracted else ""

    # Get the last word as last name
    gt_last_name = gt.split(" ")[-1] if gt else ""

    return dist_fn(extracted_last_name, gt_last_name)

def compare_location(extracted: str, gt: str, dist_fn=default_string_distance) -> float:
    # Compare complete location strings, but format gt similarly to extracted

    return dist_fn(extracted, gt)


metric_fns_sim = {
    "Species name": compare_species_name,
    "Collection date": compare_collection_date,
    "Collector's name": compare_collectors_name,
    #"Country/State": default_string_distance,  # Excluded, because not contained in every image
    "Location/Place": compare_location,
    "Location/Description": compare_location,
    #"Region": default_string_distance,  # Assuming region is a simple string comparison
}

metric_fns_lev = {
    "Species name": partial(compare_species_name, dist_fn=lev.distance),
    "Collection date": partial(compare_collection_date, dist_fn=lev.distance),
    "Collector's name": partial(compare_collectors_name, dist_fn=lev.distance),
    #"Country/State": default_string_distance,  # Excluded, because not contained in every image
    "Location/Place": partial(compare_location, dist_fn=lev.distance),
    "Location/Description": partial(compare_location, dist_fn=lev.distance),
    #"Region": lev.distance,  # Assuming region is a simple string comparison
}

metric_fns_match = {
    "Species name": partial(compare_species_name, dist_fn=equal),
    "Collection date": partial(compare_collection_date, dist_fn=equal),
    "Collector's name": partial(compare_collectors_name, dist_fn=equal),
    #"Country/State": default_string_distance,  # Excluded, because not contained in every image
    "Location/Place": partial(compare_location, dist_fn=equal),
    "Location/Description": partial(compare_location, dist_fn=equal),
    #"Region": equal,  # Assuming region is a simple string comparison
}

def compare_tables(
    extracted_csv: str,
    ground_truth_csv: str,
    column_map: Dict[str, str],
):
    # Load CSVs
    extracted = pd.read_csv(extracted_csv, keep_default_na=False)
    ground_truth = pd.read_csv(ground_truth_csv, keep_default_na=False)

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
    for col in extracted_cols:
        if col not in extracted.columns:
            extracted[col] = ""
    extracted = extracted.loc[common_images, extracted_cols]
    ground_truth = ground_truth.loc[common_images, ground_truth_cols]
    ground_truth.columns = extracted_cols  # Rename for comparison

    # Ensure both dataframes have the same source_image items
    missing_in_extracted = ground_truth.index.difference(extracted.index)

    # Add missing rows with empty strings
    for idx in missing_in_extracted:
        extracted.loc[idx] = [""] * len(extracted.columns)

    # Sort both dataframes to align rows
    extracted = extracted.sort_index()
    ground_truth = ground_truth.sort_index()

    if extracted.shape != ground_truth.shape:
        print(f"Warning: Dataframes have different shapes after alignment. {extracted.shape} vs {ground_truth.shape}")
        # Align by index, filling missing values with empty strings
        try:
            extracted = extracted.reindex(ground_truth.index, fill_value="")
            ground_truth = ground_truth.reindex(extracted.index, fill_value="")
        except Exception as e:
            print(f"Error aligning dataframes: {e}")
            return pd.Series()

    # Check if both indices are now identical
    assert all(extracted.index == ground_truth.index), "Indices do not match after alignment."
    assert all(s1 == s2 for s1, s2 in zip(extracted.shape, ground_truth.shape)), f"The dataframe shapes do not match. {(extracted.shape, ground_truth.shape)}"

    # Split "Location" columns into two columns based on a delimiter (e.g., ",")
    if "Location" in extracted.columns and "Location" in ground_truth.columns:
        extracted[["Location/Place", "Location/Description"]] = extracted["Location"].str.split(", ", n=1, expand=True)
        ground_truth[["Location/Place", "Location/Description"]] = ground_truth["Location"].str.split(" / ", n=1, expand=True)

        extracted[extracted.isnull()] = ""
        ground_truth[ground_truth.isnull()] = ""

    print(extracted)
    print(ground_truth)

    # Compute metrics
    results = {}

    sim_cols = []
    lev_cols = []

    for col in metric_fns_sim:
        results[f"{col}_similarity"] = extracted[col].combine(ground_truth[col], metric_fns_sim[col]).mean()
        results[f"{col}_levenshtein"] = extracted[col].combine(ground_truth[col], metric_fns_lev[col]).mean()
        results[f"{col}_match"] = extracted[col].combine(ground_truth[col], metric_fns_match[col]).mean()

        sim_cols.append(f"{col}_similarity")
        lev_cols.append(f"{col}_levenshtein")

    results["loc_contained"] = np.mean([el in glp or el in gld for el, glp, gld in zip(extracted["Location/Place"], ground_truth["Location/Place"], ground_truth["Location/Description"])])
    results["loc_similar_contained"] = np.mean([substring_match(el, glp + " " + gld) for el, glp, gld in zip(extracted["Location/Place"], ground_truth["Location/Place"], ground_truth["Location/Description"])])
    results["loc_similar_contained_lev1"] = np.mean([substring_match_levenshtein(el, glp + " " + gld, 1) for el, glp, gld in zip(extracted["Location/Place"], ground_truth["Location/Place"], ground_truth["Location/Description"])])
    results["loc_similar_contained_lev2"] = np.mean([substring_match_levenshtein(el, glp + " " + gld, 2) for el, glp, gld in zip(extracted["Location/Place"], ground_truth["Location/Place"], ground_truth["Location/Description"])])
    results["loc_similar_contained_lev3"] = np.mean([substring_match_levenshtein(el, glp + " " + gld, 3) for el, glp, gld in zip(extracted["Location/Place"], ground_truth["Location/Place"], ground_truth["Location/Description"])])

    df_results = pd.Series(results)
    df_results["mean_similarity"] = df_results[sim_cols].mean()
    df_results["mean_levenshtein"] = df_results[lev_cols].mean()

    # Write side-by-side comparison to a CSV file for this extraction
    comparison_df = pd.DataFrame({
        "source_image": extracted.index,
    })
    for col in extracted.columns:
        comparison_df[f"extracted_{col}"] = extracted[col].values
        comparison_df[f"ground_truth_{col}"] = ground_truth[col].values

    # Save to file (filename based on extracted_csv)
    comparison_output = extracted_csv.replace(".csv", "_comparison.csv")
    comparison_df.to_csv(comparison_output, index=False)

    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare extracted herbarium data with ground truth.")
    parser.add_argument("--extracted_csv", required=True, nargs="+", help="Path to extracted data CSV.")
    parser.add_argument("--ground_truth_csv", required=True, help="Path to ground truth CSV.")
    parser.add_argument("--output_csv", default=None, help="Path to save comparison results.")
    args = parser.parse_args()

    # Parse column_map argument
    column_map = {
        "source_image": "Bildname",
        "Species name": "Spezies Label",
        "Collection date": "Sammeldatum",
        "Collector's name": "Sammler",
        "Country/State": "Geographische_Zuordnung",
        "Location": "Fundort Label",
        "Region": "Naturraum",
        "Notes": "Bemerkung_zur_Pflanze",
    }

    metric_series = {}

    for csv_file in args.extracted_csv:
        s = compare_tables(
            csv_file,
            args.ground_truth_csv,
            column_map,
        )
        metric_series[csv_file] = s

    df_results = pd.DataFrame(metric_series).T

    print(df_results)

    if args.output_csv:
        df_results.to_csv(args.output_csv, index=True)
