#!/bin/bash

# This script evaluates extracted data against ground truth for both handwritten and printed datasets.

# Get dataset folder via command line argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <experiment_folder> <dataset_folder>"
    exit 1
fi

experiment_folder="$1"
dataset_folder="$2"

# Evaluate handwritten data
python evaluate.py \
    --extracted_csv "$experiment_folder"/*image_list=data.handwritten.txt*/extracted_data.csv \
    --ground_truth_csv "$dataset_folder/handwritten/label_data.csv" \
    --output_csv "$experiment_folder/evaluation_results_handwritten.csv"

# Evaluate printed data
python evaluate.py \
    --extracted_csv "$experiment_folder"/*image_list=data.printed.txt*/extracted_data.csv \
    --ground_truth_csv "$dataset_folder/printed/label_data.csv" \
    --output_csv "$experiment_folder/evaluation_results_printed.csv"
