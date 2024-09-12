import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import torch
# import argparse
import os

# from extract_weight import extract_weights

import argparse

def get_OpName(input_file):
    op_names = []

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.split('.')
            op_name = parts[0].strip()
            op_names.append(op_name)

    return op_names


def generate_data(model_path, batch_size=1):
    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    model = tf.lite.Interpreter(model_content=model_buffer)
    input_details = model.get_input_details()
    input_tensors = []
    for i in range(len(input_details)):
        # print(input_details[i]['dtype'])
        shape_input = input_details[i]['shape'].astype(np.int32).tolist()
        shape_input[0] = batch_size
        print(shape_input)
        if input_details[i]['dtype'] == np.uint8:
            inputs = torch.ones(size=tuple(shape_input)).to(torch.uint8).numpy()
            print("Data type of this model is uint8")
        elif input_details[i]['dtype'] == np.float32:
            inputs = torch.ones(tuple(shape_input)).numpy()
            print("Data type of this model is float32")
        input_tensors.append(inputs)
    return input_tensors


def model_inference(interpreter, inputs, index=None):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(len(inputs))
    for i in range(len(inputs)):
        interpreter.set_tensor(input_details[0]["index"], np.expand_dims(inputs[i], 0))
        interpreter.invoke()
        if i == 0:
            if index is not None:
                output = interpreter.get_tensor(index)
            else:
                output = interpreter.get_tensor(output_details[0]['index'])
        else:
            output = np.concatenate((output, interpreter.get_tensor(output_details[0]['index'])), axis=0)
    return output

OpNameList = get_OpName('../oplist.txt')
interpreter = tf.lite.Interpreter(
    '../obf_model.tflite', experimental_preserve_all_tensors=True
    )
interpreter.allocate_tensors()

# op_details = interpreter._get_ops_details()
# print(op_details)


# for tensor_details in interpreter.get_tensor_details():
#     print(tensor_details)

for op_details in interpreter._get_ops_details():
    if op_details["op_name"] == OpNameList[0].capitalize():
        input_index = op_details["inputs"]
        if len(input_index) == 1:
            print("Input index: ", input_index[0])
            input_idx = input_index[0]
        else:
            raise ValueError("More than one input index")
        output_index = op_details["outputs"]
        if len(output_index) == 1:
            print("Output index: ", output_index[0])
            output_idx = output_index[0]
        else:
            raise ValueError("More than one output index")

inputs = generate_data('../obf_model.tflite', batch_size=1)[0] * 100.0
op_input = model_inference(interpreter, inputs, index=input_idx)
print(op_input)
op_output = model_inference(interpreter, inputs, index=output_idx)
print(op_output)
output_ori = model_inference(interpreter, inputs, index=None)
print(output_ori)