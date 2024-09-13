/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <cstring>
#include <memory>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace randopname {

constexpr int kInputTensor = 0;
// constexpr int kShapeTensor = 1;
constexpr int kOutputTensor = 0;
const float obf_value=1.0;

const float act_max=99.0;
const float act_min=-99.0;

TfLiteIntArray* GetOutputShape(TfLiteContext*, TfLiteNode*);


TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
//   TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const float* input_data = GetTensorData<float>(input);
  float* output_data = GetTensorData<float>(output);

  // Get the number of elements in the input tensor
  int num_elements = NumElements(input);

  // Copy the values from input_data to output_data
  for (int i = 0; i < num_elements; ++i) {
    if ((input_data[i] * obf_value) > act_max){
      output_data[i] = act_max;
    } else if ((input_data[i] * obf_value) < act_min)
    {
      output_data[i] = act_min;
    } else {
      output_data[i] = input_data[i] * obf_value;
    }
  }

  return kTfLiteOk;
}

}  // namespace reshape

TfLiteRegistration* Register_randopname() {
  static TfLiteRegistration r = {nullptr, nullptr, randopname::Prepare,
                                 randopname::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
