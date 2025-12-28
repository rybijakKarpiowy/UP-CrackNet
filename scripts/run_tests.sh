#!/usr/bin/bash

# Get folder names from ./saved_model
pwd
folders=($(ls -d ./saved-model/G_D_448_ngf_9*/ | xargs -n 1 basename))

# Loop through each folder and run the Python scripts
for folder in "${folders[@]}"; do
    echo "Processing folder: $folder"
    uv run test_generator.py --model_name "$folder"
    uv run scripts/img_binarization.py --model_name "$folder"
    uv run scripts/calculate_errors.py --approach_name "$folder"
done