#!/bin/bash

while [[ $# -gt 0 ]]; do
	case $1 in
		-f|--folder)
			FOLDER="$2"
			shift # past argument
			shift # past value
			;;
		-*|--*)
			echo "Unknown option $1"
			exit 1
			;;
	esac
done

export WF=./$FOLDER
export BUILD=${WF}/build
export LOG=${BUILD}/logs

mkdir -p ${LOG}
python -u ${WF}/train.py -d ${BUILD} 2>&1 | tee ${LOG}/train.log


python -u ${WF}/quantize.py -d ${BUILD} --quant_mode calib 2>&1 | tee ${LOG}/quant_calib.log
python -u ${WF}/quantize.py -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log

source ${WF}/compile.sh zcu104 ${BUILD} ${LOG}
