import argparse
from email.headerregistry import HeaderRegistry

from requests import head
import trsfile
import matplotlib.pyplot as plt
import os

# ap = argparse.ArgumentParser()
# ap.add_argument('-f', '--file', type=str, default='./traces/trace.trs', help='Path to trs file. Default is ./traces/trace.trs')
# args = ap.parse_args()

files = os.listdir("./traces")

for f in files:
    if ".trs" in f:
        with trsfile.open("./traces/"+f, 'r') as traces:
            # Show all headers
            
            for header, value in traces.get_headers().items():
                if 'SAMPLES' in str(header):
                    print(f)
                    print(header, '=', value)
            #print()

            #show first trace 
            #plt.plot(traces[1])
            #plt.show()

            # Iterate over the first 25 traces
            # for i, trace in enumerate(traces):
            #     plt.plot(trace)
            #     plt.show()
