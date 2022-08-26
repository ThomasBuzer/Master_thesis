###
###corr_align.py: Loads a trs dataset and align traces based on correlation on a pattern in a reference trace.
###
import sys
import numpy as np
import trsfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fftpack import fft, ifft, fftshift




if __name__=="__main__":
    #If True, will not save traces but only log alignement
    DRY_RUN = False

    window_size = 100
    step_size = 10

    #Load traceset
    traceFileName = sys.argv[1]
    traceset = trsfile.open(traceFileName, 'r')
    Nt, Ns = traceset.get_header(trsfile.Header.NUMBER_TRACES), traceset.get_header(trsfile.Header.NUMBER_SAMPLES)
    # al_traces = np.zeros((100,Ns))

    #Create an empty traceset to store aligned traces
    if DRY_RUN is False:
        headers = traceset.get_headers().copy()
        headers.update({trsfile.Header.NUMBER_TRACES:0})#Create a tracefile with zero traces, because it is not "possible" to overwrite existing traces in .trs dataset.
        headers.update({trsfile.Header.NUMBER_SAMPLES:(Ns-window_size)//step_size})
        headers.update({trsfile.Header.SAMPLE_CODING:trsfile.SampleCoding.FLOAT})
        al_traceset = trsfile.open(traceFileName.split(".")[0] + "_ABSWR.trs", 'w', headers=headers)

    # Define trace portion to apply correlattion


    i = 0
    for trace in tqdm(traceset[:1000000]):
        t = np.array(trace.samples)*1
        new_trace= []

        #ABS and WIN_RES
        for j in (range(int((len(trace)-window_size)/step_size))):
            result = np.average(np.absolute(t[step_size*j:step_size*j+window_size]))
            new_trace += [result]

        if DRY_RUN is False:
                coin = int(trace.parameters.serialize()[0])
                parameters = trsfile.parametermap.TraceParameterMap()
                parameters["BYTE"]=trsfile.traceparameter.ByteArrayParameter([coin])
                al_traceset.append(trsfile.Trace(trsfile.SampleCoding.FLOAT, new_trace, parameters=parameters))#, data=trace.data))
        i+=1

    al_traceset.close()
    traceset.close()
