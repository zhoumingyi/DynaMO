import numpy as np
import tensorflow as tf
import argparse
import json
import re
import os

def parse_data(file_content):
    data_types = []
    data_arrays = []
    seen_data = set()

    # Split content by lines
    lines = file_content.splitlines()

    for i, line in enumerate(lines):
        # Check if the line matches the pattern for operator's name and data type
        match = re.match(r'(\w{6})\((\w+), (.+)\)', line)
        if match:
            data_type = match.group(2)
            shape = tuple(map(int, re.findall(r'\d+', match.group(3))))

            # Get the corresponding data line
            data_line = lines[i + 1].strip()

            # Extract data values and convert to numpy array
            data_values = np.fromstring(data_line.strip('()'), sep=',')
            data_array = data_values.reshape(shape)

            # Create a hashable representation of the data to check for uniqueness
            data_key = (data_type, shape, tuple(data_values))

            if data_key not in seen_data:
                seen_data.add(data_key)
                data_types.append((data_type, shape))
                data_arrays.append(data_array)

    return data_types, data_arrays


def extract_weights():
    # Read the content of the file
    with open('extracted_weights.txt', 'r') as file:
        file_content = file.read()

    # Parse the data
    data_types, data_arrays = parse_data(file_content)

    return data_types, data_arrays



def parse_operator_parameters(text):
    pattern = re.compile(r'(\w+)\(([^)]*)\)')
    matches = pattern.findall(text)

    operators = []
    for match in matches:
        op_name = match[0]
        params = match[1].split(', ')
        params = [param for param in params if param]

        operators.append({
            "OpName": op_name,
            "padding": params[0],
            "stride_w": int(params[1]),
            "stride_h": int(params[2]),
            "dilation_w": int(params[3]),
            "dilation_h": int(params[4])
        })
    return operators

def read_operator_parameters_from_file():
    with open("extracted_params.txt", 'r') as file:
        text = file.readlines()

    # Remove duplicates
    unique_lines = list(set(text))

    # Join unique lines back into a single string for parsing
    unique_text = ''.join(unique_lines)

    return parse_operator_parameters(unique_text)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default='skin', help='name of the model')
    # opt = parser.parse_args()

    # Read the content of the file
    with open('extracted_weights.txt', 'r') as file:
        file_content = file.read()

    # Parse the data
    data_types, data_arrays = parse_data(file_content)

    # Print results
    print("Data Types:")
    for data_type in data_types:
        print(data_type)

    print(len(data_arrays))
    # print("\nData Arrays:")
    # for array in data_arrays:
    #     print(array)