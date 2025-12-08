# Declare parameter lists

methods=("bilater_otsu_after_combining_w_edges_and_classifier")
powers=(0.8)
canny_1s=(80 90 100 110 120)
canny_2s=(180 190 200 210 220)

# Loop through all combinations of parameters
for method in "${methods[@]}"; do
  for power in "${powers[@]}"; do
    for canny_1 in "${canny_1s[@]}"; do
      for canny_2 in "${canny_2s[@]}"; do
        echo "Running img_binarization.py with method=$method, power=$power, canny_threshold1=$canny_1, canny_threshold2=$canny_2"
        uv run scripts/img_binarization.py --method "$method" --power "$power" --canny_threshold1 "$canny_1" --canny_threshold2 "$canny_2"
        # Calculate error metrics
        echo "Calculating error metrics for method=$method, power=$power, canny_threshold1=$canny_1, canny_threshold2=$canny_2"
        uv run scripts/calculate_errors.py --approach_name "${method}_p${power}_c1${canny_1}_c2${canny_2}"
      done
    done
  done
done