#!/bin/bash

# run the model obfuscator
cd ..
bash build_obf.sh fruit
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
# run the model deobfuscator
cd pin
./deobf_struct.sh 150528 40 0 1
python get_struc_performance.py | tee -a attack.txt
python get_extract_performance.py --model_name=fruit --input_num=150528 --output_num=40 --num_output_tensor=0 | tee -a attack.txt

# run the model obfuscator
cd ..
bash build_obf.sh skin
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
# run the model deobfuscator
cd pin
./deobf_struct.sh 150528 4 0 1
python get_struc_performance.py | tee -a attack.txt
python get_extract_performance.py --model_name=skin --input_num=150528 --output_num=4 --num_output_tensor=0 | tee -a attack.txt

# run the model obfuscator
cd ..
bash build_obf.sh mobilenet
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
# run the model deobfuscator
cd pin
./deobf_struct.sh 150528 1001 0 1
python get_struc_performance.py | tee -a attack.txt
python get_extract_performance.py --model_name=mobilenet --input_num=150528 --output_num=1001 --num_output_tensor=0 | tee -a attack.txt

# run the model obfuscator
cd ..
bash build_obf.sh mnasnet
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
# run the model deobfuscator
cd pin
./deobf_struct.sh 150528 1001 0 1
python get_struc_performance.py | tee -a attack.txt
python get_extract_performance.py --model_name=mnasnet --input_num=150528 --output_num=1001 --num_output_tensor=0 | tee -a attack.txt

# run the model obfuscator
cd ..
bash build_obf.sh squeezenet
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
# run the model deobfuscator
cd pin
./deobf_struct.sh 150528 1001 0 1
python get_struc_performance.py | tee -a attack.txt
python get_extract_performance.py --model_name=squeezenet --input_num=150528 --output_num=1001 --num_output_tensor=0 | tee -a attack.txt

# run the model obfuscator
cd ..
bash build_obf.sh efficientnet
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
# run the model deobfuscator
cd pin
./deobf_struct.sh 150528 100 0 1
python get_struc_performance.py | tee -a attack.txt
python get_extract_performance.py --model_name=efficientnet --input_num=150528 --output_num=1000 --num_output_tensor=0 | tee -a attack.txt

# run the model obfuscator
cd ..
bash build_obf.sh depth_estimation
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
# run the model deobfuscator
cd pin
./deobf_struct.sh 196608 65536 0 1
# ./deobf_struct.sh 307200 185094 1 0
python get_struc_performance.py | tee -a attack.txt
python get_extract_performance.py --model_name=depth_estimation --input_num=196608 --output_num=65536 --num_output_tensor=0 | tee -a attack.txt


# run the model obfuscator
cd ..
bash build_obf.sh lenet
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
# run the model deobfuscator
cd pin
./deobf_struct.sh 6144 1 0 1
python get_struc_performance.py | tee -a attack.txt
python get_extract_performance.py --model_name=lenet --input_num=6144 --output_num=1 --num_output_tensor=0 | tee -a attack.txt

# run the model obfuscator
cd ..
bash build_obf.sh posenet
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
# run the model deobfuscator
cd pin
./deobf_struct.sh 272163 6647  0 1
./deobf_struct.sh 272163 25024 1 0
./deobf_struct.sh 272163 391   2 0
./deobf_struct.sh 272163 13294 3 0
python get_struc_performance.py | tee -a attack.txt
python get_extract_performance.py --model_name=posenet --input_num=272163 --output_num=6647 --num_output_tensor=0 | tee -a attack.txt

# run the model obfuscator
cd ..
bash build_obf.sh ssd
cd minimal_x86_build/ && rm -rf * &&cd ..
cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build . -j && cd ..
# run the model deobfuscator
cd pin
./deobf_struct.sh 307200 185094  0 1
# ./deobf_struct.sh 307200 8136  1 0
./deobf_struct.sh 307200 185094  1 0
python get_struc_performance.py | tee -a attack.txt
python get_extract_performance.py --model_name=ssd --input_num=307200 --output_num=8136 --num_output_tensor=0 | tee -a attack.txt