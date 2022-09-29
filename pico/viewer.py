import argparse
import trsfile
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', type=str, default='./traces/trace.trs', help='Path to trs file. Default is ./traces/trace.trs')
args = ap.parse_args()

with trsfile.open(args.file, 'r') as traces:
	# Show all headers
    for header, value in traces.get_headers().items():
        print(header, '=', value)
    print()

    #show first trace 
    #plt.plot(traces[1])
    #plt.show()
    definitions = trsfile.parametermap.TraceParameterDefinitionMap()
    definitions["BYTE"] = trsfile.traceparameter.TraceParameterDefinition(trsfile.traceparameter.ParameterType.BYTE, 1, 0)

	# Iterate over the first 25 traces
    for i, trace in enumerate(traces):
        #print(trace.get_input())
        trace = traces[i].samples
        #coin = traces[i].parameters.serialize()[0]
        #print(i, trace, "COIN is ", coin, len(trace))
        plt.plot(np.array(trace))
        plt.show()
