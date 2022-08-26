#!/bin/bash
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
TARGET=zcu104
echo "-----------------------------------------"
echo "COMPILING MODEL FOR ZCU104.."
echo "-----------------------------------------"


BUILD=./build
LOG=./build/logs

compile() {
  vai_c_xir \
  --xmodel      ${BUILD}/quant_model/CNN_int.xmodel \
  --arch        $ARCH \
  --net_name    CNN_${TARGET} \
  --output_dir  ${BUILD}/compiled_model
}

compile 2>&1 | tee ${LOG}/compile_$TARGET.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"



