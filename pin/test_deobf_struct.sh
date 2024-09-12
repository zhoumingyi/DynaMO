#!/bin/bash

# Check if the user provided an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <opname>"
    exit 1
fi

# Get the opname from the user's input
opname=$1

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

# Run the initial pintool command
output=$(~/pin-3.30-98830-g1d7b601b3-gcc-linux/pin -t ./obj-intel64/FindRelatedFunc.so -opname $opname -- ../minimal_x86_build/minimal ../obf_model.tflite 150528 40)

# Extract the last 40 floating-point numbers from the output
ori_out=$(echo "$output" | grep -oE '[0-9]+\.[0-9]+(e[-+]?[0-9]+)?' | tail -n 40)

# Save the float numbers as an array
ori_out_array=($ori_out)

# Print the array (optional)
echo "Extracted float numbers (ori_out_array): ${ori_out_array[@]}"

# Get the first line of the text file Used_Func.txt, and assign it to the variable func_name
func_name=$(head -n 1 Used_Func.txt)

# Run the probe.so pintool with the obtained function name and capture the output
output=$(~/pin-3.30-98830-g1d7b601b3-gcc-linux/pin -t ./obj-intel64/probe.so -FuncName $func_name -- ../minimal_x86_build/minimal ../obf_model.tflite 150528 40)

# Extract the last 40 floating-point numbers from the output
ins_out=$(echo "$output" | grep -oE '[0-9]+\.[0-9]+(e[-+]?[0-9]+)?' | tail -n 40)

# Save the float numbers as an array
ins_out_array=($ins_out)

# Print the array (optional)
echo "Extracted float numbers (ins_out_array): ${ins_out_array[@]}"

# Compare the arrays
if compare_arrays ori_out_array[@] ins_out_array[@]; then
    echo "The arrays are the same."
else
    echo "The arrays are different."
fi
