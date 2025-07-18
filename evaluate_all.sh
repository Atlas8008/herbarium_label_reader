#!/bin/bash

for folder in $1/*; do
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"
        echo $(basename "$folder") >> "$1/evaluation_results.txt"
        python evaluate.py --extracted_csv "$folder/extracted_data.csv" --ground_truth_csv ~/datasets/herbarium_scans_senckenberg/2024_14\ P\ Anhang_Belegdaten.csv >> "$1/evaluation_results.txt"
        echo >> "$1/evaluation_results.txt"
    else
        echo "Skipping non-directory: $folder"
    fi
done