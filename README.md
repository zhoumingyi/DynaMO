# DynaMO
Code for ASE2024 Paper: DynaMO: Protecting Mobile DL Models through Coupling Obfuscated DL Operators. 
Note that for testing our codes, you should have a Ubuntu desktop (RAM > 32 GB). Note that you may will experience some configure issues because the tool compilations may be different on differene devices. If so, please let me know through the Issue (indicate your device information), I will try my best to fix it.

## Download ModelObfuscator

### 0. Preparation: download this project:

```
git clone https://github.com/zhoumingyi/DynaMO.git
cd DynaMO
```


### 1. Preparation: build the environment of ModelObfuscator:

(0) Download the code of ModelObfuscator:

```
git clone https://github.com/zhoumingyi/ModelObfuscator.git
cd ModelObfuscator
```

(1) The dependency can be found in `environment.yml`. To create the conda environment:

```
conda env create -f environment.yml
conda activate code275
```

Install the Flatbuffer:

```
conda install -c conda-forge flatbuffers
```

(if no npm) install the npm:

```
sudo apt-get install npm
```

Install the jsonrepair:

```
npm install -g jsonrepair
```

Note that the recommend version of gcc and g++ is 9.4.0.


(2) Download the source code of the TensorFlow. Here we test our tool on v2.9.1.

```
wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.9.1.zip
```

Unzip the file:

```
unzip v2.9.1
```

(3) Download the Bazel:

```
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
```

You can test the Bazel:

```
which bazel
```

It should return:

```
# in ubuntu
/usr/local/bin/bazel
```

(4) Configure the build:

```
cd tensorflow-2.9.1/
./configure
cd ..
```

You can use the default setting (just type Return/Enter for every option).

(5) Copy the configurations and script to the source code:  

```
cp ./files/kernel_files/* ./tensorflow-2.9.1/tensorflow/lite/kernels/
cp ./files/build_files/build.sh ./tensorflow-2.9.1/
cp -r ../minimal ./tensorflow-2.9.1/tensorflow/lite/examples/
```

Note that you can mofify the maximal number of jobs in the 'build.sh' script. Here I set it as `--jobs=14`. 


## Test the DLModelParser

### 1. Copy the DLModelParser:

```
cp -r ../pin ./
```

### 2. Install the Intel Pin:

Install the Intel Pin in your home directory:

```
cp ../pin-3.30-98830-g1d7b601b3-gcc-linux.tar.gz ~/
```

unzip the Intel Pin:

```
tar -zxvf ~/pin-3.30-98830-g1d7b601b3-gcc-linux.tar.gz
```

### 3. Install the Intel Pin:

Copy some essential files to the ModelObfuscator project

```
cp -r ../other_build/* ./
```

### 4. Build the DLModelParser (based on Pin)

```
cd pin
make all TARGET=intel64
```

### 5. Test the DLModelParser

```
bash deobf.sh
```

Then, you will see the results (will be saved in the file 'attack.txt'). 

## Test the DynaMO

### 0. Go back to the ModelObfuscator directory

### 1. Use the file in the DynaMO/DynaMO_build to replace the files in the ModelObfuscator's source projects. It applies our DynaMO algorithm.

```
cp../DynaMO_build/model_parser.py ./
cp../DynaMO_build/obfuscation.py ./
cp../DynaMO_build/ObfOptions.cc ./tfl_source_file
```

### 2. Then, redo the test process in the last section to see the defending performance.

```
cd pin
bash deobf.sh
```