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

After changing the Clock frequencies given to the DPU (100 and 200 MHz instead of 300 and 600MHz) for easier measurements, the first measurements show an activity of the DPU at 100Mhz when an application is running. Two applications have been tested including a resnet50 image recognition network and a 3-4-and 5 layer CNN trained on the MNIST dataset. These two examples come from the Vitis Tutorial Library. 

The measuring setup is the following :
<div align="center">
<img src="./images/30_05/setup.jpg" width="600">
</div>

The ZCU104 communicates through USB with TeraTerm and the Picoscope also uses USB to communicate with the PC. The trigger wire (linked to the LED) is plugged on the B channel of the scope and the RF-U 2,5-2 probe to the A channel through the PA303 amplifier. After some measurements, the probe has been aimed at the capacitor located on Y16 on the back of the board. 

### MNIST example

The networks trained on the MNIST dataset allow fast inferences (<50ms). On the following figure, the trigger is set high before the dpu.execute_async() command and set low after the dpu.wait(). This allows to identify the limits of the DPU activity. 

<div align="center">
<img src="./images/30_05/MNIST_5L_2.jpg" width="600">
</div>

The graph shows a peak of activity within the trigger. The spectrum shows a lot of low frequency (<40MHz) noise and a apike at 100MHz which is one of the clocks entering the DPU. The following graphs show the background noise (probe in the air) and the signal when the DPU is not working : just the trigger is running.

<div align="center">
<img src="./images/30_05/background.jpg" width="400"><img src="./images/30_05/trigger_1.jpg" width="400">
</div>

The intensity of the 100MHz peak does not change when the DPU is running or not but it disapears on the background noise. The peaks around 100MHz shared on tall the graphs seems to be local radio stations (Oranje Radio 94,2 MHz). 

Zooming on the peak observed on the first graph shows interesting features. It shows frequency peaks around 100 and 200 MHz which corresponds to the frequencies of the running DPU and it shows some patterns which could be identified as computing layers. The following graphs show no activity from the DPU against the DPU running the MNIST classification on a different timescale than previously shown.

<div align="center">
<img src="./images/30_05/trigger_Zoomed_1.jpg" width="400"><img src="./images/30_05/MNIST_5L_Zoomed_1.jpg" width="400">
</div>

To confirm that the patterns oberved on the Scope of the 5 layer CNN, a 4-layer, a 3-layer, and a bigger 5-layer  CNN were implemented. To precisely record the activity of the layers. The following graphs show the envelope of the trace averaged on 10k traces for the 3, 4 and 5-layer CNN.

<div align="center">
<img src="./images/30_05/MNIST_3L_superposed.jpg" width="500"><img src="./images/30_05/MNIST_4L_superposed.jpg" width="500"><img src="./images/30_05/MNIST_5L_superposed.jpg" width="500">
</div>
