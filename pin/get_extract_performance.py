import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import argparse
import torch
import os


from extract_weight import extract_weights, read_operator_parameters_from_file
from graph_rebuilding import predict_operator, fully_connected_output_shape, predict_operator_from_arrays

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='fruit', help='name of the model')
parser.add_argument('--input_num', type=int, default=150528, help='name of the model')
parser.add_argument('--output_num', type=int, default=40, help='name of the model')
parser.add_argument('--num_output_tensor', type=int, default=0, help='out tensor ID')
opt = parser.parse_args()


def process_file(input_file):
    op_names = []
    labels = []
    predictions = []

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.split('.')
            op_name = parts[0].strip()
            label_type = parts[1].strip()

            op_names.append(op_name)
            if label_type == "ObfOptions":
                labels.append(1)
                predictions.append(0)
            else:
                labels.append(0)
                predictions.append(0)

    return op_names, labels, predictions


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
        # print(shape_input)
        if input_details[i]['dtype'] == np.uint8:
            inputs = torch.ones(size=tuple(shape_input)).to(torch.uint8).numpy()
            # print("Data type of this model is uint8")
        elif input_details[i]['dtype'] == np.float32:
            inputs = torch.ones(tuple(shape_input)).numpy()
            # print("Data type of this model is float32")
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


def compute_accuracy(op_name_label, op_name_predict, struc_error_count):
    assert len(op_name_label) == len(op_name_predict), "The lists must have the same length"

    correct_predictions = 0
    total_predictions = len(op_name_label)

    for label, predict in zip(op_name_label, op_name_predict):
        if label == predict:
            correct_predictions += 1

    accuracy = correct_predictions / (total_predictions + struc_error_count)
    return accuracy


def get_OpName(input_file):
    op_names = []

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.split('.')
            op_name = parts[0].strip()
            op_names.append(op_name)

    return op_names


def get_RealOpName(input_file):
    op_names = []

    with open(input_file, 'r') as file:
        for line in file:
            # parts = line.split('.')
            op_name = line.strip()
            op_names.append(op_name)

    return op_names


def are_lists_equal(list1, list2):
    error = 0.0
    if len(list1) != len(list2):
        return False, error

    if list1 == []:
        return False, error

    # Sort both lists
    sorted_list1 = sorted(list1, key=lambda x: x.size)
    sorted_list2 = sorted(list2, key=lambda x: x.size)

    error_sign = False
    # Compare elements
    for arr1, arr2 in zip(sorted_list1, sorted_list2):
        if np.max(np.abs((arr1 - arr2))) > 1e-4:
            error_sign = True
            # return False, error
        # else:
        error += np.max(np.abs((arr1 - arr2)))

    if error_sign:
        return False, error/len(list1)/max(np.max(np.abs(arr1[0])), np.max(np.abs(arr1[0])))
    else:
        return True, error/len(list1)/max(np.max(np.abs(arr1[0])), np.max(np.abs(arr1[0])))


model_path = '../tflite_model/'
model_name = opt.model_name + '.tflite'
interpreter = tf.lite.Interpreter(
 os.path.join(model_path, model_name)
)
interpreter.allocate_tensors()

OpNameList = get_OpName('../oplist.txt')
realOpNameList = get_RealOpName('real_opnames.csv')

obf_interpreter = tf.lite.Interpreter(
    '../obf_model.tflite', experimental_preserve_all_tensors=True
    )
obf_interpreter.allocate_tensors()


struc_error_count = 0
input_file = '../oplist.txt'
# Process the input file to get the operation names and labels
op_names, labels, predictions = process_file(input_file)
for i in range(len(realOpNameList)):
    for j in range(len(op_names)):
        if realOpNameList[i] == op_names[j]:
            if labels[j] == 1:
                struc_error_count += 1
print("struc_error_count: ", struc_error_count)
# realOpNameList = []
# for i in range(len(op_names)):
#     if labels[i] == 0:
#         realOpNameList.append(op_names[i])


obf_outID = []
for i, op_details in enumerate(obf_interpreter._get_ops_details()):
    if op_details["op_name"].lower() not in realOpNameList:
        output_index = op_details["outputs"]
        # print(op_details)
        obf_outID.append(output_index[0])

    # if op_details["op_name"].lower() == "byylyx":
    #     print("Real Op: ", op_details["op_name"].lower())
    #     print(op_details)

# print("obf_outID: ", obf_outID)

output_idx_all = []
for i, op_details in enumerate(interpreter._get_ops_details()):
    output_idx = op_details["outputs"]
    for j in range(len(output_idx)):
        output_idx_all.append(output_idx[j])


correct = 0.
total = 0.

op_name_label = []
op_name_predicted = []

wea_error = 0.0

for i, op_details in enumerate(interpreter._get_ops_details()):
    # data_types = []
    ori_data_arrays = []
    if op_details["op_name"] != "DELEGATE":
        # total += 1
        # print("original op: ", op_details["op_name"])
        # print(OpNameList[i])
        input_list = op_details["inputs"].tolist()
        # print("input_list: ", input_list)
        input_list.pop(0)

        remove_list = []
        for j in range(len(input_list)):
            if input_list[j] in output_idx_all:
                remove_list.append(input_list[j])
        for j in range(len(remove_list)):
            input_list.remove(remove_list[j])
        # print("input_list: ", input_list)
        for j in range(len(input_list)):
            weights = interpreter.get_tensor(input_list[j])
            # print(weights)
            ori_data_arrays.append(weights)
        os.system("~/pin-3.30-98830-g1d7b601b3-gcc-linux/pin -t \
                  ./obj-intel64/extract_weights.so -opname " + OpNameList[i] + " -- \
                  ../minimal_x86_build/minimal \
                  ../obf_model.tflite " + str(opt.input_num) + " " + str(opt.output_num) +
                  " " + str(opt.num_output_tensor) + " > /dev/null 2>&1")

        os.system("~/pin-3.30-98830-g1d7b601b3-gcc-linux/pin -t \
                  ./obj-intel64/extract_params.so -opname " + OpNameList[i] + " -- \
                  ../minimal_x86_build/minimal \
                  ../obf_model.tflite " + str(opt.input_num) + " " + str(opt.output_num) +
                  " " + str(opt.num_output_tensor) + " > /dev/null 2>&1")
        extracted_data_types, extracted_data_arrays = extract_weights()
        if len(ori_data_arrays) != 0:
            total += 1
            # for j in range(len(ori_data_arrays)):
                # print("extracted weights shape: ", extracted_data_arrays[0].shape)
                # print("original weights shape: ", ori_data_arrays[0].shape)
                # print("max difference: ", np.max(extracted_data_arrays[j] - ori_data_arrays[j]))
        # print(extracted_data_arrays[1])
        correct_sign, wea_error_op = are_lists_equal(ori_data_arrays, extracted_data_arrays)
        wea_error += wea_error_op
        # print("max error: ", wea_error_op)
        if correct_sign:
            correct += 1
        # else:
        #     print("====================================")
        #     print("fails to extract the weights")
        #     print("failed Op: ", OpNameList[i])
        #     print(extracted_data_arrays)
        #     print("====================================")

        inputs = generate_data('../obf_model.tflite', batch_size=1)[0] * 0.5
        for obf_op_details in obf_interpreter._get_ops_details():
            if obf_op_details["op_name"] == OpNameList[i].capitalize():
                # print("obf op: ", obf_op_details["op_name"])
                input_index = obf_op_details["inputs"].tolist()
                # print("before remove input id: ", input_index)

                remove_list = []
                for j in range(len(input_index)):
                    if input_index[j] in obf_outID:
                        remove_list.append(input_index[j])
                for j in range(len(remove_list)):
                    input_index.remove(remove_list[j])

                # print("after remove input id: ", input_index)

                if len(input_index) == 1:
                    # print("Input index: ", input_index[0])
                    input_idx = input_index[0]
                    op_input = model_inference(obf_interpreter, inputs, index=input_idx)
                    # print("input shape: ", op_input.shape)
                else:
                    print("More than one input index: ", obf_op_details["op_name"])
                    # input_idx = input_index[0]
                    input_data_list = []
                    # print("op_name: ", obf_op_details["op_name"])
                    # print("input_idx: ", input_idx)
                    for k in range(len(input_index)):
                        input_data_list.append(model_inference(obf_interpreter,
                                                            inputs, index=input_index[k]))
                    # print("input shape: ", op_input.shape)
                    # print("ID: ", input_index)
                    # op_name_predicted.append("CONCATENATE")
                    # op_name_label.append(op_details["op_name"])
                output_index = obf_op_details["outputs"]
                if len(output_index) == 1:
                    # print("Output index: ", output_index[0])
                    output_idx = output_index[0]
                    op_output = model_inference(obf_interpreter, inputs, index=output_idx)
                    # print("output shape: ", op_output.shape)
                else:
                    print("More than one output index: ", obf_op_details["op_name"])
                    # output_idx = output_index[0]
                    # op_output = model_inference(obf_interpreter, inputs, index=output_idx)
                    # print("output shape: ", op_output.shape)
        if len(input_index) == 1 and len(output_index) == 1:
            input_shape = op_input.shape
            output_shape = op_output.shape
            if len(extracted_data_arrays) != 0:
                weight_shape = extracted_data_arrays[0].shape
                bias_shape = extracted_data_arrays[1].shape
                # print("input_shape: ", input_shape)
                # print("output_shape: ", output_shape)
                # print("weight_shape: ", weight_shape)
                # print("bias_shape: ", bias_shape)
                if len(read_operator_parameters_from_file()) != 0:
                    op_parameters = read_operator_parameters_from_file()[0]
                    # print(op_parameters)
                    strides = (op_parameters['stride_w'], op_parameters['stride_h'])
                    padding = op_parameters['padding']
                    dilation_rate = (op_parameters['dilation_w'], op_parameters['dilation_h'])
                    predicted_op = predict_operator(input_shape,
                       output_shape, weight_shape, bias_shape,
                       strides, padding, dilation_rate)
                    # print("Unknown operator predict_operator")
                    # print("predicted op name: ", predicted_op)
                    # print("label: ", op_details["op_name"])
                    op_name_label.append(op_details["op_name"])
                    op_name_predicted.append(predicted_op)
                else:
                    # Check FullyConnected
                    fully_connected_out_shape = fully_connected_output_shape(input_shape,
                                                weight_shape, bias_shape)
                    if fully_connected_out_shape == output_shape:
                        # print("predicted op name: FullyConnected")
                        # print("label: ", op_details["op_name"])
                        op_name_label.append(op_details["op_name"])
                        op_name_predicted.append("FULLY_CONNECTED")
                    else:
                        # print("Unknown operator")
                        # print("label: ", op_details["op_name"])
                        op_name_label.append(op_details["op_name"])
                        op_name_predicted.append("Unknown operator")
            else:
                predicted_op = predict_operator_from_arrays(op_input, op_output)
                # print("Unknown operator predict_operator_from_arrays")
                # print("label: ", op_details["op_name"])
                op_name_label.append(op_details["op_name"])
                op_name_predicted.append(predicted_op)
        elif len(input_index) > 1 and len(output_index) == 1:
            # Check if all arrays have the same shape
            first_shape = input_data_list[0].shape
            if all(array.shape == first_shape for array in input_data_list):
                # Perform element-wise sum
                sum_result = np.sum(input_data_list, axis=0)

                # Check if the sum result is equal to the output
                if np.array_equal(sum_result, op_output):
                    op_name_predicted.append("ADD")
                    op_name_label.append(op_details["op_name"])
                else:
                    op_name_predicted.append("CONCATENATION")
                    op_name_label.append(op_details["op_name"])
            else:
                op_name_predicted.append("CONCATENATION")
                op_name_label.append(op_details["op_name"])



print(f"Weights Extract Rate: {correct/total * 100:.2f}%")
print(f"Weights Extract Acc: {wea_error/correct}")

print("Op Name Label: ", op_name_label)
print("Op Name Predicted: ", op_name_predicted)
accuracy = compute_accuracy(op_name_label, op_name_predicted, struc_error_count)
print(f"Name Identification Rate: {accuracy * 100:.2f}%")