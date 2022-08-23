# Master_thesis

This folder will group all the ressources needed to replicate the attacks performed during my master's thesis at the University of Radboud (Nijmegen). This thesis will end my 3-year-long journey at the Ecole Centrale de Lyon (Ecully).

<div align="center">
<img src="./images/logo-ecl-rectangle-quadri-print.jpg" width="400"><img src="./images/logoradboud.png" width=400>
</div>

The preliminary report written to introduce the subject to my supervisors at ECL can be found [here](Preliminary_report.pdf)

The main focus of this thesis is to recover hyperparameters and weights of a Neural Network implemented on a Xilinx zcu104 through Side Channel Attacks.

## Setup the Board

[Xilinx Tutorial](https://github.com/Xilinx/Vitis-AI) will walk you through the setup of Vitis AI on the board which allows users to easily run NN on the FPGA.

The embedded CPU will run PetaLinux as RTOS or Bare Metal are not available for the Deep Processing Unit (DPU) IP on this board yet.

Vitis AI has the ability to run any Network which has been developped with [Pytorch](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Design_Tutorials/09-mnist_pyt), Tensorflow or Caffe. Once converted and implemented on the board, [Vitis AI Runtime](https://github.com/Xilinx/Vitis-AI/tree/master/demo/VART) allows the user to controll the execution via python or C++ API.

The image given by Xilinx for the evaluation board does not allow GPIOs access. To enable them, a new image must be build. [GPIO_trigger_setup](./GPIO_trigger_setup.md) will guide you through the process of building the image and explain how to use the GPIOs as a trigger. The modified image allowing DPU use and GPIOs access can be found in the [platform_files](./platform_files) folder


## Setup

The setup is shown on the next Figure. It is described in more details in the thesis.

<div align="center">
<img src="./images/overleaf/setup/setup.jpg" width="400">
</div>

## Recording traces

In order to record traces, several algorithm must be implemented in order to synchronise the recordings. The following Figure shows the basic principle of the system. The master coordinator is the collect_pico.py programm running on the Mako computer. This programm is responsible for lanching the app.py (through ssh) which runs on the ZCU board and is responsible for the inference of the Neural Network. The inference only occurs when the picoscope is ready and the computer sends a signal to the ZCU board through a TCP message. 
The collect_pico.py programm outputs a TRS file which contains all the traces. Several algorithms have been implemented to convert and process these traces.

<div align="center">
<img src="./images/overleaf/setup/collect_traces.jpg" width="400">
</div>


## Vitis-AI/Simple_CNNs
Most commands cited here must be run in the Vitis-AI environment. Use the following commands inside the Viti-AI folder to enter it.

```sh
./docker_run.sh xilinx/vitis-ai-cpu:latest
conda activate vitis-ai-pytorch
```

The folders inside Simple_CNNs corresponds to different kind of architectures that have been tested. All the folders have the same basic principle which corresponds to the example given by Xilinx to implement the MNIST example through pytorch. This example has been modified so that the network is not trained to save time. They contain all the scripts required to build and compile .xmodel files. 

The architecture of the network implemented is stored in the common.py file of each folder.

The builder.sh script has been create to automatically create the model with pytorch, quantize it and compile it with the Vitis-AI environment.

### list

The list folder allows to build and compile multiple architectures in one go. The architecture are single convolution layers which parameters are stored in a csv file inside the folder. all the models can the be built with the multi_builder.sh script which compiles all the architectures and store them in the build/compiled_model folder.

### singleC

This folder contains tools to generate a lot of networks to assess single weight leakage.
To save time, it has been decided not to quantize random models because it takes too much time. The randomness is added by modifying the binary quantized model. This means that the creation of a model takes less than one second which is really efficient.

The generator.py script compiles these models based on an original network which have been quantized. The weights of the model have been identified in the file with a hex compare software. The generator then replaces the weights with random or fixed weights depending on the application.

The file_handler.py script is mandatory to deal with the transfer of the xmodels to the ZCU104 board.

The idea is to launch both the generator.py and file_handler.py to handle models creation and transfer while the collect_pico_weights.py collects the traces.



