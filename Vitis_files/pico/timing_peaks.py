import argparse
import trsfile
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', type=str, default='./traces/trace.trs', help='Path to trs file. Default is ./traces/trace.trs')
args = ap.parse_args()


def peak_width(trace, timeout, threshold):
        start = 0
        uptime = 0
        decay = timeout
        for counter in range(len(trace)):
                #detect first peak
                if trace[counter] > threshold and start == 0:
                        start = counter
                if trace[counter] > threshold and start != 0:
                        decay = timeout
                        uptime += 1
                if decay == 0:
                        #print("Length is "+ str(counter-timeout-start) + "points")
                        #print("Start is "+ str(start))
                        #print("End is "+ str(counter-timeout))
                        #plt.plot(trace[start-100:counter])
                        #plt.show()
                        return counter-timeout-start, counter, uptime
                if start !=0 and trace[counter] < threshold:
                        decay -= 1
                counter += 1
        return 0, 0, 0

def all_peaks(trace, timeout=30, threshold=35):
        width = timeout
        peaks = [0]*200
        uptimes = [0]*200
        i=0
        while(width != 0):
                width, end, uptime = peak_width(trace, timeout, threshold)
                trace = trace[end:]
                peaks[i] = width
                uptimes[i] = uptime
                i+=1
        return peaks, uptimes

with trsfile.open(args.file, 'r') as traces:
        # Show all headers
        for header, value in traces.get_headers().items():
            print(header, '=', value)
            if(str(header) == "Header.SCALE_X"):
                scale_X = int(value * 1e9)
        print()
        

        list_peaks = np.asarray([[0]*200])
        list_uptimes = np.asarray([[0]*200])
        counter = 1
        processors = 10


        #for trace in traces[:10]:
        for i in range(len(traces)//processors):
            pool = Pool(processors)
            runs = pool.map(all_peaks, traces[i*processors:(i+1)*processors])

            pool.close()
            pool.join()
            for run in runs:
                peaks, uptimes = np.array([run[0]]), np.array([run[1]])
                list_peaks = np.append(list_peaks, peaks, axis=0)
                list_uptimes = np.append(list_uptimes, uptimes, axis=0)
            print(str((i+1)*processors)+" Done !")

        #rescale for time
        list_peaks *= scale_X
        list_uptimes *= scale_X

        print("Peaks look like this :")
        print(list_peaks)
        print("Peaks array shape : " + str(np.shape(list_peaks)))       
        print()
        print("Uptimes look like this :")
        print(list_uptimes)
        print("Uptimes array shape : "+ str(np.shape(list_uptimes)))
        print()


np.savetxt('peaks.csv', list_peaks, fmt= '%d', delimiter=',')
np.savetxt('uptimes.csv', list_uptimes, fmt= '%d', delimiter=',')
