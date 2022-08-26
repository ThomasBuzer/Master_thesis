import trsfile
import numpy as np
from tqdm import tqdm

x_val = np.load("x_val.npy")
y_val = np.load("y_val.npy")

for i in tqdm(range(10)):
    traceFileName = "trs_files/layer_"+str(y_val[i*25][0]) + '_' + str(y_val[i*25][1])+ '_' + str(y_val[i*25][2])+ '_'+ str(y_val[i*25][3])+ '_'+ str(y_val[i*25][4])+".trs"

    headers = {
            trsfile.Header.NUMBER_SAMPLES:250000,
            trsfile.Header.LENGTH_DATA:int(0), #Header_data_length,
            trsfile.Header.SAMPLE_CODING:trsfile.SampleCoding.FLOAT,
            trsfile.Header.LABEL_X:"s",
            trsfile.Header.LABEL_Y:"V",
        }
    traceFile = trsfile.trs_open(traceFileName, mode='w', headers=headers)

    for j in range(25):
        traceFile.append(trsfile.Trace(trsfile.SampleCoding.FLOAT, x_val[i*25+j]))
    traceFile.close()



