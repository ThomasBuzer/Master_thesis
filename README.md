# Master_thesis

This folder will group all the ressources needed to replicate the attacks performed during my master's thesis at the University of Radboud (Nijmegen). This thesis will end my 3-year-long journey at the Ecole Centrale de Lyon (Ecully).

<img src="./images/logo-ecl-rectangle-quadri-print.jpg" width="400"><img src="./images/logoradboud.png" width=400>

The preliminary report written to introduce the subject to my supervisors at ECL can be found [here](Preliminary_report.pdf)

The main focus of this thesis is to recover hyperparameters and weights of a Neural Network implemented on a Xilinx zcu104 through Side Channel Attacks.

## Setup the Board

[Xilinx Tutorial](https://github.com/Xilinx/Vitis-AI) will walk you through the setup of Vitis AI on the board which allows users to easily run NN on the FPGA.

The embedded CPU will run PetaLinux as RTOS or Bare Metal are not available for the Deep Processing Unit (DPU) IP on this board yet.

Vitis AI has the ability to run any Network which has been developped with [Pytorch](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Design_Tutorials/09-mnist_pyt), Tensorflow or Caffe. Once converted and implemented on the board, [Vitis AI Runtime](https://github.com/Xilinx/Vitis-AI/tree/master/demo/VART) allows the user to controll the execution via python or C++ API.

### GPIO Trigger

The prebuild image burned on the SD card does not allow easy access to the GPIO pins. To enable this, the whole image has to be rebuild with [This](https://github.com/Xilinx/Vitis-Tutorials/tree/2021.2/Vitis_Platform_Creation/Introduction/02-Edge-AI-ZCU104) tutorial guiding through the steps. [This](https://www.youtube.com/watch?v=CHsidFIXUEE) youtube video helps understanding the steps to add GPIO pins.

## Design Flow

The idea of the setup is to modify the base image given by Xilinx to enable the LEDs (easier than the PMOD gpios). First download this [image](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms/2021-2.html). In the **hw** folder will be the **hw.xsa** file which can be opened with vitis through the **open_hw_platform** in the TCL console. 

Adding the LEDs is as simple as adding a AXI_GPIO and connecting it to the AXI Interconnect "interconnect_axihpm0fdp"


## ERRORS you might encounter

Using an earlier version of the Vitis environment (prior to 2022.1), you **WILL** have to apply this [patch](https://support.xilinx.com/s/article/76960?language=en_US) for the IP checker to work properly. 

**step 2 :** you might want to build the petalinux software offline. [This](https://support.xilinx.com/s/article/2021-1-PetaLinux?language=en_US) post guides you through the process.

**step 4 :** During the build of the Vitis project (step 4-4) you might encounter these problems:
- **source file does not exist** : [this](https://support.xilinx.com/s/question/0D52E00006hpRgHSAU/vitis-ide-error-v-60602-source-file-does-not-exist?language=en_US) post should fix it by changing the path to the default boards.
- opencv2 not found : Download the base image for the zcu104 [here](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms/2021-2.html) in which can be found the proper version of opencv. Provide it to the Vitis environement (explanation [here](https://support.xilinx.com/s/question/0D52E00006hpPCUSA2/vitis-vision-libraries-error-on-build?language=en_US)). The default installation path for opencv2 is **petalinux_sdk_2021.2/sysroots/cortexa72-cortexa53-xilinx-linux/usr/local/include/opencv4** for the Includes and  **petalinux_sdk_2021.2/sysroots/cortexa72-cortexa53-xilinx-linux/usr/local/lib** for the Library references.





WIP
