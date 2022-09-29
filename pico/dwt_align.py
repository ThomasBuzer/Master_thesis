###
###corr_align.py: Loads a trs dataset and align traces based on correlation on a pattern in a reference trace.
###
import sys
import numpy as np
import trsfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fftpack import fft, ifft, fftshift

#Use fft to compute the normalized cross-corelation between a template x and a trace y.
def normalized_cross_correlation_using_fft(x,y):
    x = x - np.average(x)/np.std(x)
    y = y - np.average(y)/np.std(y)
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))

    #normalize the output
    cSumA = np.sum(y)
    cSumA2 = np.sum(np.square(y))
    sigmaA = np.sqrt(cSumA2-(cSumA**2)/len(x))
    sigmaT = np.std(x)*np.sqrt(len(x)-1)
    nXcc = (cc - cSumA*np.mean(x))/(sigmaT*sigmaA)
    return fftshift(nXcc)

# Computes the normalized cross-corelation between two misaligned traces and outputs the best shift between them.
# max_shift is the maximum shift allowed. It should limit the border cases.
# The smaller the template trace, the faster the computation (no surprise).
# (Use with care.)
def compute_shift(ref, trace, start=0, end=None, max_shift=None):
    if end==None:
        end = len(trace)
    pad = end-start
    if max_shift == None:
        y = np.pad(trace, (pad,),'constant',constant_values=(0))
        x = np.pad(ref[start:end], (start+pad,len(trace)-end+pad),'constant',constant_values=(0))
    else:
        # y = np.pad(trace[start-max_shift:end+max_shift], (pad,),'constant',constant_values=(0))
        # x = np.pad(np.pad(ref[start:end], (start,len(trace)-end),'constant',constant_values=(0))[start-max_shift:end+max_shift], (pad,),'constant',constant_values=(0))
        y = np.pad(trace[start-max_shift:end+max_shift], (pad,),'constant',constant_values=(0))
        x = np.pad(ref[start:end], (max_shift + pad),'constant',constant_values=(0))
    assert len(x) == len(y)
    c = normalized_cross_correlation_using_fft(x,y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    if max_shift != None:
        assert max_shift <= zero_index
        c = c[zero_index-max_shift:zero_index+max_shift]
        zero_index = max_shift
    shift = zero_index - np.argmax(c)
    return shift, c.max()


if __name__=="__main__":
    #If True, will not save traces but only log alignement
    DRY_RUN = False

    #Load traceset
    traceFileName = sys.argv[1]
    traceset = trsfile.open(traceFileName, 'r')
    Nt, Ns = traceset.get_header(trsfile.Header.NUMBER_TRACES), traceset.get_header(trsfile.Header.NUMBER_SAMPLES)
    # al_traces = np.zeros((100,Ns))

    #Create an empty traceset to store aligned traces
    if DRY_RUN is False:
        headers = traceset.get_headers().copy()
        headers.update({trsfile.Header.NUMBER_TRACES:0})#Create a tracefile with zero traces, because it is not "possible" to overwrite existing traces in .trs dataset.
        al_traceset = trsfile.open(traceFileName.split(".")[0] + "_align.trs", 'w', headers=headers)

    #Open log file to store the value of shift and correlation coefficient for every trace
    log = open(traceFileName.split(".")[0] + "_align.txt", 'w')

    # Define trace portion to apply correlattion
    sample_coding = traceset.get_header(trsfile.Header.SAMPLE_CODING)
    index = 0                       #Index of the reference trace
    ref = traceset[index].samples   #Reference trace
    start = 5001#50000                     #Start of the segement to align from reference trace
    end = 30000#150000                     #End of the segement to align from reference trace
    threshold = 0.7               #The minimum correlation expected to consider the alignment successful
    max_shift = 5000               #The maximum shift allowed.

    i = 0
    min_corr = 1.0
    _, max_corr = compute_shift(ref, ref, start, end, max_shift=max_shift)
    for trace in tqdm(traceset[:1000000]):
        t = trace.samples
        shift_f, corr_f = compute_shift(ref, t, start, end, max_shift=max_shift)
        if DRY_RUN is False:
            if corr_f/max_corr >= threshold:
                #save the aligned trace by rotating the samples given shift_f.
                shift_trace = np.roll(t,-shift_f)
                coin = int(trace.parameters.serialize()[0])
                parameters = trsfile.parametermap.TraceParameterMap()
                parameters["BYTE"]=trsfile.traceparameter.ByteArrayParameter([coin])
                al_traceset.append(trsfile.Trace(sample_coding, shift_trace, parameters=parameters))#, data=trace.data))
        log.write("trace {}\t\t: shift:{}, corr:{}\n".format(i,shift_f,corr_f/max_corr))
        if min_corr > corr_f/max_corr:
            min_corr = corr_f/max_corr
        i+=1
    print("min_corr=",min_corr)

    log.close()
    # traceset.close()
    if DRY_RUN is False:
        al_traceset.close()

        al_traceset = trsfile.trs_open(traceFileName.split(".")[0] + "_align.trs", 'r')
        n_al_traces = al_traceset.get_header(trsfile.Header.NUMBER_TRACES)
        print(n_al_traces, traceset.get_header(trsfile.Header.NUMBER_TRACES))
        for i in range(1, n_al_traces):
            plt.plot(al_traceset[i], color='gray')
            plt.plot(al_traceset[0],color='red', alpha=0.3)
            plt.show()
        al_traceset.close()
        pass
