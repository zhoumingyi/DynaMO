#!/bin/bash

# Check if the user provided an argument
if [ $# -ne 4 ]; then
    echo "Usage: $0 <input_dim> <output_dim> <num_output_tensor> <clear_mode>"
    exit 1
fi

# Get the opname from the user's input
input_dim=$1
output_dim=$2
num_output_tensor=$3
clear_mode=$4

# Function to compare two arrays
compare_arrays() {
    local arr1=("${!1}")
    local arr2=("${!2}")

    if [ ${#arr1[@]} -ne ${#arr2[@]} ]; then
        return 1
    fi

    for ((i=0; i<${#arr1[@]}; i++)); do
        if [ "${arr1[i]}" != "${arr2[i]}" ]; then
            return 1
        fi
    done

    return 0
}

# Function to compare two arrays and compute the accumulated element-wise error
# compare_arrays_diff() {
#     local arr1=("${!1}")
#     local arr2=("${!2}")

#     # Initialize the accumulated error variable
#     local accumulated_error=0

#     if [ ${#arr1[@]} -ne ${#arr2[@]} ]; then
#         echo "Arrays have different lengths"
#         return 1
#     fi

#     for ((i=0; i<${#arr1[@]}; i++)); do
#         # Compute the absolute difference and accumulate the error using awk
#         local diff=$(awk -v a="${arr1[i]}" -v b="${arr2[i]}" 'BEGIN { print (a > b ? a - b : b - a) }')
#         accumulated_error=$(awk -v error="$accumulated_error" -v d="$diff" 'BEGIN { print error + d }')
#         if awk -v error="$accumulated_error" 'BEGIN { exit !(error > 0) }'; then
#             return 1
#         fi
#     done

#     # Return the accumulated error
#     echo "Accumulated Error: $accumulated_error"
#     return 0
# }


# Output CSV file
output_csv="real_opnames.csv"
# echo "opname" > "$output_csv"
if [ "$clear_mode" = "1" ]; then
    echo "clear mode is on"
    > "$output_csv"
fi

# Read oplist.txt file line by line
while read -r line; do
    # Extract opname from the line
    opname=$(echo "$line" | awk '{print $1}' | tr -d '.')

    echo "Processing opname: $opname"

    # Run the initial pintool command
    output=$(~/pin-3.30-98830-g1d7b601b3-gcc-linux/pin -t ./obj-intel64/FindRelatedFunc.so -opname "$opname" -- ../minimal_x86_build/minimal ../obf_model.tflite $input_dim $output_dim $num_output_tensor)

    # Extract the last floating-point numbers from the output
    ori_out=$(echo "$output" | grep -oE '[0-9]+\.[0-9]+(e[-+]?[0-9]+)?' | tail -n $output_dim)

    # Save the float numbers as an array
    ori_out_array=($ori_out)

    # Print the array (optional)
    # echo "Extracted float numbers (ori_out_array) for $opname: ${ori_out_array[@]}"

    # Get the first line of the text file Used_Func.txt, and assign it to the variable func_name
    func_name=$(head -n 1 Used_Func.txt)

    # Run the probe.so pintool with the obtained function name and capture the output
    output=$(~/pin-3.30-98830-g1d7b601b3-gcc-linux/pin -t ./obj-intel64/probe.so -FuncName "$func_name" -- ../minimal_x86_build/minimal ../obf_model.tflite $input_dim $output_dim $num_output_tensor)

    # Extract the last floating-point numbers from the output
    ins_out=$(echo "$output" | grep -oE '[0-9]+\.[0-9]+(e[-+]?[0-9]+)?' | tail -n $output_dim)

    # Save the float numbers as an array
    ins_out_array=($ins_out)

    # Print the array (optional)
    # echo "Extracted float numbers (ins_out_array) for $opname: ${ins_out_array[@]}"

    # Compare the arrays
    if compare_arrays ori_out_array[@] ins_out_array[@]; then
        echo "The arrays for $opname are the same."
    else
        echo "The arrays for $opname are different."
        # Record the opname in the CSV file
        echo "$opname" >> "$output_csv"
    fi

done < ../oplist.txt
