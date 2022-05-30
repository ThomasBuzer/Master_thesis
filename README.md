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


## Preliminary results

After changing the Clock frequencies given to the DPU (100 and 200 MHz instead of 300 and 600MHz) for easier measurements, the first measurements show an activity of the DPU at 100Mhz when an application is running. Two applications have been tested including a resnet50 image recognition network and a 4 Layer CNN trained on the MNIST dataset. These two examples come from the Vitis Tutorial Library.
