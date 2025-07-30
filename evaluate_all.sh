#!/bin/bash


python evaluate.py \
    --extracted_csv "$1"/*image_list=data.handwritten.txt*/extracted_data.csv \
    --ground_truth_csv ~/datasets/herbarium_scans_senckenberg/handwritten/label_data.csv \
    --output_csv "$1/evaluation_results_handwritten.csv"

python evaluate.py \
    --extracted_csv "$1"/*image_list=data.printed.txt*/extracted_data.csv \
    --ground_truth_csv ~/datasets/herbarium_scans_senckenberg/printed/label_data.csv \
    --output_csv "$1/evaluation_results_printed.csv"


# for folder in $1/*; do
#     if [ -d "$folder" ]; then
#         echo "Processing folder: $folder"
#         echo $(basename "$folder") >> "$1/evaluation_results.txt"
#         python evaluate.py --extracted_csv "$folder/extracted_data.csv" --ground_truth_csv ~/datasets/herbarium_scans_senckenberg/2024_14\ P\ Anhang_Belegdaten.csv >> "$1/evaluation_results.txt"
#         echo >> "$1/evaluation_results.txt"
#     else
#         echo "Skipping non-directory: $folder"
#     fi
# done