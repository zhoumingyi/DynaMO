import numpy as np
from extract_weight import read_operator_parameters_from_file

candidates = ['AveragePool2D', 'Softmax', 'Logistic', 'MaxPool2D', 'Reshape', 'Relu', 'Squeeze', 'Mean']

def conv2d_output_shape(input_shape, weight_shape, bias_shape, strides, padding, dilation_rate):
    batch_size, in_height, in_width, in_channels = input_shape
    out_channels, filter_height, filter_width, _ = weight_shape
    d_filter_height = filter_height + (filter_height - 1) * (dilation_rate[0] - 1)
    d_filter_width = filter_width + (filter_width - 1) * (dilation_rate[1] - 1)
    if padding == 'kTfLitePaddingValid':
        out_height = (in_height - d_filter_height + strides[0]) // strides[0]
        out_width = (in_width - d_filter_width + strides[1]) // strides[1]
    elif padding == 'kTfLitePaddingSame':
        out_height = (in_height + strides[0] - 1) // strides[0]
        out_width = (in_width + strides[1] - 1) // strides[1]
    else:
        raise ValueError("Unsupported padding type")
    return (batch_size, out_height, out_width, out_channels)


def depthwise_conv2d_output_shape(input_shape, weight_shape, bias_shape, strides, padding, dilation_rate):
    batch_size, in_height, in_width, in_channels = input_shape
    in_channels, filter_height, filter_width, filter_channel = weight_shape
    d_filter_height = filter_height + (filter_height - 1) * (dilation_rate[0] - 1)
    d_filter_width = filter_width + (filter_width - 1) * (dilation_rate[1] - 1)
    if padding == 'kTfLitePaddingValid':
        out_height = (in_height - d_filter_height + strides[0]) // strides[0]
        out_width = (in_width - d_filter_width + strides[1]) // strides[1]
    elif padding == 'kTfLitePaddingSame':
        out_height = (in_height + strides[0] - 1) // strides[0]
        out_width = (in_width + strides[1] - 1) // strides[1]
    else:
        raise ValueError("Unsupported padding type")
    return (batch_size, out_height, out_width, in_channels * filter_channel)


def fully_connected_output_shape(input_shape, weight_shape, bias_shape):
    if len(list(input_shape)) == 2:
        batch_size, input_dim = input_shape
    elif len(list(input_shape)) == 4:
        batch_size, _, _, input_dim = input_shape
    output_dim, input_dim = weight_shape
    return (batch_size, output_dim)


def predict_operator(input_shape, output_shape, weight_shape, bias_shape, strides, padding, dilation_rate):
    # Check Conv2D
    conv2d_out_shape = conv2d_output_shape(input_shape, weight_shape, bias_shape, strides, padding, dilation_rate)
    # print("shape of conv2d: ", conv2d_out_shape)
    if conv2d_out_shape == output_shape:
        return "CONV_2D"

    # Check DepthwiseConv2D
    depthwise_conv2d_out_shape = depthwise_conv2d_output_shape(input_shape, weight_shape, bias_shape, strides, padding, dilation_rate)
    # print("shape of depthconv2d: ", depthwise_conv2d_out_shape)
    if depthwise_conv2d_out_shape == output_shape:
        return "DEPTHWISE_CONV_2D"

    return "Unknown operator"


def predict_operator_from_arrays(input_array, output_array):

    input_shape = input_array.shape
    output_shape = output_array.shape

    # # AveragePool2D and MaxPool2D typically reduce the spatial dimensions
    # if len(input_shape) == 4 and len(output_shape) == 4:
    #     if input_shape[1] > output_shape[1] and input_shape[2] > output_shape[2] and input_shape[3] == output_shape[3]:
    #         print("output_array: ", output_array.shape)
    #         print("input_array: ", input_array.shape)
    #         print("max: ", np.max(input_array, axis=(1, 2), keepdims=True).shape)
    #         if np.all(np.isclose(output_array, np.mean(input_array, axis=(1, 2), keepdims=True))):
    #             return 'AVERAGE_POOL_2D'
    #         elif np.all(np.isclose(output_array, np.max(input_array, axis=(1, 2), keepdims=True))):
    #             return 'MAX_POOL_2D'


    # # AveragePool2D and MaxPool2D typically reduce the spatial dimensions
    # if len(input_shape) == 4 and len(output_shape) == 4:
    #     if input_shape[3] == output_shape[3] and input_shape[0] == output_shape[0]:
    #         if input_shape[1] > output_shape[1] and input_shape[2] > output_shape[2]:
    #             pool_size = input_shape[1] // output_shape[1]
    #             print("output_array: ", output_array.shape)
    #             print("input_array: ", input_array.shape)
    #             if np.all(np.isclose(output_array, np.mean(input_array.reshape(input_shape[0], output_shape[1], pool_size, output_shape[2], pool_size, -1), axis=(2, 4)))):
    #                 return 'AVERAGE_POOL_2D'
    #             elif np.all(np.isclose(output_array, np.max(input_array.reshape(input_shape[0], output_shape[1], pool_size, output_shape[2], pool_size, -1), axis=(2, 4)))):
    #                 return 'MAX_POOL_2D'

    def is_pooling_operation(pooling_func, input_array, output_array, pool_size, strides):
        for i in range(0, input_array.shape[1] - pool_size + 1, strides):
            for j in range(0, input_array.shape[2] - pool_size + 1, strides):
                if not np.allclose(
                    output_array[:, i // strides, j // strides, :],
                    pooling_func(input_array[:, i:i + pool_size, j:j + pool_size, :], axis=(1, 2))
                ):
                    return False
        return True

    # Check pooling operations
    if len(input_shape) == 4 and len(output_shape) == 4:
        if input_shape[3] == output_shape[3] and input_shape[0] == output_shape[0]:
            for pool_size in range(2, min(input_shape[1], input_shape[2]) + 1):
                for strides in range(1, pool_size + 1):
                    if (input_shape[1] - pool_size) % strides == 0 and (input_shape[2] - pool_size) % strides == 0:
                        if output_shape[1] == (input_shape[1] - pool_size) // strides + 1 and output_shape[2] == (input_shape[2] - pool_size) // strides + 1:
                            if is_pooling_operation(np.mean, input_array, output_array, pool_size, strides):
                                return 'AVERAGE_POOL_2D'
                            elif is_pooling_operation(np.max, input_array, output_array, pool_size, strides):
                                return 'MAX_POOL_2D'

    # Softmax typically transforms the last dimension into a probability distribution
    if len(input_shape) == len(output_shape) and input_shape[:-1] == output_shape[:-1]:
        if np.allclose(np.sum(output_array, axis=-1), 1) and np.all(output_array >= 0):
            return 'SOFTMAX'

    # Logistic (Sigmoid) applies element-wise sigmoid function
    if input_shape == output_shape:
        if np.all((output_array >= 0) & (output_array <= 1)):
            if np.allclose(output_array, 1 / (1 + np.exp(-input_array)), atol=1e-6):
                return 'LOGISTIC'
            if np.allclose(output_array, np.maximum(input_array, 0)):
                return 'RELU'

    # Reshape changes the shape of the input array without changing its data
    if np.prod(input_shape) == np.prod(output_shape):
        return 'RESHAPE'

    # Squeeze removes dimensions of size 1 from the input array
    if input_array.squeeze().shape == output_shape:
        return 'SQUEEZE'

    # Mean reduces the dimensions by computing the mean along specified axes
    if np.all(np.isclose(output_array, np.mean(input_array, axis=tuple(range(1, len(input_shape))), keepdims=False))):
        return 'MEAN'

    return "Unknown operator"

if __name__ == "__main__":

    # Example usage
    file_path = 'operators.txt'  # Path to your text file
    operators = read_operator_parameters_from_file()

    # Assuming input, output shapes and weights are known
    input_shape = (1, 28, 28, 3)
    output_shape = (1, 26, 26, 64)
    weight_shape = (3, 3, 3, 64)
    bias_shape = (64,)

    for operator in operators:
        strides = (operator['stride_w'], operator['stride_h'])
        padding = operator['padding']
        dilation_rate = (operator['dilation_w'], operator['dilation_h'])

        predicted_op = predict_operator(input_shape, output_shape, weight_shape, bias_shape, strides, padding, dilation_rate)
        print(f"Operator in file: {operator['OpName']} - Predicted operator: {predicted_op}")



