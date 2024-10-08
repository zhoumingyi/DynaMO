/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdio>
#include <iostream>
#include <sys/time.h>
// #include <time.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
// #include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/optimized/multithreaded_conv.h"
#include "tensorflow/lite/kernels/eigen_support.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

using namespace tflite;

int main(int argc, char* argv[]) {
// extern "C" const float* tflite_minimal(char* path, float* input_v, int num_input, int num_output) {
  const int num_input = atoi(argv[2]);
  const int num_output = atoi(argv[3]);
  const int num_output_tensot = atoi(argv[4]);
  const char* filename = argv[1];
  float input_v[num_input]={0};
  for(int i = 0; i < (num_input); i++)
  {
    input_v[i] = 100;
  }
  std::cout << "TFLite model: " << filename << std::endl;
  // Load model
  // timeval t_start, t_end;
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  // gettimeofday( &t_start, NULL);
  interpreter->AllocateTensors();
  float* input = interpreter->typed_input_tensor<float>(0);
  memcpy(input, input_v, 1*num_input*sizeof(float));
  interpreter->Invoke();
  float* output = interpreter->typed_output_tensor<float>(num_output_tensot);
  float* output_f = new float[num_output];
  memcpy(output_f, output, 1*num_output*sizeof(float));

  for(int i = 0; i < (num_output); i++)
  {
    std::cout << output_f[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
