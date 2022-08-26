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
export BUILD=${WF}build/
export LOG=${BUILD}logs/

mkdir -p ${LOG}
LEN_ARCH=$(wc -l ${WF}/arch_validation.csv | awk '{ print $1 }')
#echo $LEN_ARCH

for((i=0;i<$LEN_ARCH; i++))
do
echo "Going for the model $i"
python -u ${WF}/train.py -wf ${WF} -d ${BUILD} -ln ${i} 2>&1 | tee ${LOG}/train.log


python -u ${WF}/quantize.py -wf ${WF} -ln ${i} -d ${BUILD} --quant_mode calib 2>&1 | tee ${LOG}/quant_calib.log
python -u ${WF}/quantize.py -wf ${WF} -ln ${i} -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log

source ${WF}/compile.sh zcu104 ${BUILD} ${LOG}

python ${WF}/move.py -wf ${WF} -cf ${BUILD}/compiled_model/ -ln ${i}

echo "Model $i is done and saved !"
echo $i
done
