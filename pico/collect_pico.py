
"""
collect_pico.py : collect traces with picoscope
Use picoscope probe for the trigger (trigger is not always sensed with lecroy probes).
"""
import sys
import os
import time
import numpy as np
import serial
import matplotlib.pyplot as plt
import trsfile

import signal
import threading
from SSHLibrary import SSHLibrary
import socket
from tqdm import tqdm
import argparse
from picosdk.ps5000 import ps5000 as ps5

# from lecroy3 import *
from pico import *

DIVIDER = '-----------------------------------------'

COLLECT_TRACES = True
TTEST_STATUS = False
plot_refresh_rate = 100
ALGORITHM = 0 #0:AEAD, 1:Hash
PATH="/home/root/target_zcu104/"
BOARD_IP = '131.174.142.183'
TCP_PORT = 5005
TCP_BUFFER_SIZE = 8096

#setup ssh connection with target
ssh = SSHLibrary()
ssh.open_connection(BOARD_IP)
ssh.login("root", "root")

#Closes files and scope if interrupted
def handler(signum, frame):
    msg = "Ctrl-c was pressed. Exiting"
    print(msg, end="", flush=True)
    print("")
    if COLLECT_TRACES:
        try:
            traceFile.close()
        except:
            print("TRS File not closed properly")
        try:
            scope.disconnect()
        except:
            print("Scope not closed properly")
        try:
            ssh.close_connection()
        except:
            print("SSH pipe not closed properly")
        try:
            s.close()
        except:
            print("TCP pipe not closed properly")
    exit(1)

def ssh_thread(app, n_traces, model_name):
    out = ssh.execute_command("python3 "+PATH+app+" --model "+PATH+model_name+".xmodel --image_dir "+PATH+"images/ -n "+str(n_traces), return_rc=False, return_stdout=True, output_during_execution=True)

def data_back_thread(timeout):
    data_string = ""
    counter = 0
    while(']}' not in data_string):
        if(counter > timeout):
            print("Timeout")
            return data_string
        data_string += str(s.recv(8096))
        time.sleep(0.001)
        counter += 1
    print("out of loop")
    return data_string


def collect(app, n_traces, model):

    # program start
    start_time = time.time()

    #links interrupt signal to handler
    signal.signal(signal.SIGINT, handler)

    # number of trace to collect
    #n_traces = 2500

    #launch FPGA command in a thread
    print("setup ssh connection with target") 
    ssh = SSHLibrary() 
    ssh.open_connection(BOARD_IP) 
    ssh.login("root", "root")

    print("Started remote computation on target")
    ssh_th=threading.Thread(target=ssh_thread, args=[app, n_traces, model])
    ssh_th.start()
    #time.sleep(5)
    if not ssh_th.is_alive():
        print("Thread died for some reason")
        return

    # Configure output files
    if COLLECT_TRACES:
        current_time = str(int(time.time()))
        traceFileName = "traces/singleW/" + model + "_" + "{}_{}".format(n_traces, current_time) + ".trs"
        #commFileName = "traces/" + "Xoodyak_FVR" + "{}_{}".format(n_traces, current_time) + ".txt"
        #commFile = open(commFileName, "w")
    else:
        n_traces = 10e9 #Set big number of traces to keep computation going when setting parameters in the picoscope GUI.

    ## Scope parameters (Consider a 10div timespace and 10div voltage space)
    #voltDiv = 20 * 10e-3 #V/div
    voltRange = 100 * 1e-3 #V
    voltDiv = voltRange / 10
    timeRange = 0.2e-3 #s
    timeDiv = timeRange / 10 #s/div
    sampleRate = 1e9 #S/s

    triggerChannel = 0
    voltRangeTrigger = 100 * 1e-3 #V/div
    threshold = 40 * 1e-3 #V
    ratio_pre_trigger=0.3
    trigger_delay = 0 * 1e-3 #s
    trigger_delay = int(trigger_delay*sampleRate//8)


    threshold = int( threshold * 32768 / (voltRangeTrigger))
    n_samples=int(10*timeDiv*sampleRate) #time when trigger goes off * sampling frequency (maximum is 10*timeDiv*sampleRate?)

    if COLLECT_TRACES:
        ## Init Scope
        scope = pico5000()
        scope.connect()
        voltDiv, timeDiv, sampleRate = scope.setChannel(0, voltDiv, sampleRate, timeDiv, n_samples)
        #scope.setChannel(1, voltRangeTrigger/10, sampleRate/100, timeDiv, n_samples)
        print("Scope settings:\n\tvoltDiv: {:e}\n\tvoltRange: {}\n\ttimeDiv: {:e}\n\tsampleRate: {:e}".format(voltDiv,scope.voltRange,timeDiv,sampleRate))
        scope.setTriggerChannel(triggerChannel, threshold= threshold,enable=1, delay=trigger_delay, timeout=10000, ratio_points_pre_trigger=ratio_pre_trigger)#, threshold=threshold, timeout=10000, ratio_points_pre_trigger=ratio_pre_trigger, delay=trigger_delay) #4 is Ext Channel


    if COLLECT_TRACES:
        headers = {
            trsfile.Header.NUMBER_SAMPLES:int(n_samples),
            trsfile.Header.LENGTH_DATA:int(0), #Header_data_length,
            trsfile.Header.SAMPLE_CODING:trsfile.SampleCoding.BYTE,
            trsfile.Header.LABEL_X:"s",
            trsfile.Header.LABEL_Y:"V",
            trsfile.Header.SCALE_X:10*timeDiv/n_samples,
            trsfile.Header.SCALE_Y:10*voltDiv/np.iinfo(np.uint8).max,
        }
        traceFile = trsfile.trs_open(traceFileName, mode='w', headers=headers)

    # Setup the Trigger link to the ZCU board
    print("Setup trigger")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    while 1:
            try :
                s.connect((BOARD_IP, TCP_PORT))
                break
            except:
                time.sleep(0.5)
    #data_thread=threading.Thread(target=data_back_thread, args=[1000])
   
    print("Entering loop")
    # Main loop
    for numTotal in tqdm(range(int(n_traces)), ascii=True, desc='Capture'):
        ## Set Scope trigger
        if COLLECT_TRACES: 
            scope.arm()
        try:
            ## Get data from Scope
            if COLLECT_TRACES:
                #print("Sending trigger")
                while(s.sendall(b'1]}') != None):
                    time.sleep(1)
                    print("Missed Connection ! Trying again")
                channel_out, channel_out_interpreted = scope.getNativeSignalBytes()
                #print("Collected")
                ## Write data and trace in .TRS file
                traceFile.append(trsfile.Trace(trsfile.SampleCoding.BYTE, channel_out)) #, data=(coin).to_bytes(1, 'big') + data_output))
                #print("Pico ready !")                

        except Exception as ex:
            print("ERROR: ", ex)


    print("Done: " + str(numTotal+1))
    print("Total time: " + str(time.time() - start_time))

    if COLLECT_TRACES:
        # Close files
        traceFile.close()
        #commFile.close()
        ## Close scope and serial port
        scope.disconnect()
        #close ssh
        ssh.close_connection()
        s.close()

def run_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--app',   type=str,  default='app_server_big_picture.py',       help='App.py to use on the ZCU board. Default is app_server_big_picture.py')
    ap.add_argument('-n', '--n_traces',   type=str,  default=5,       help='Number of traces to capture. Default is 5')
    ap.add_argument('-m', '--model',   type=str,  default='CNN_detect_Relu_8C',       help='.xmodel to use for the inference (name without the .xmodel). Default is CNN_detect_Relu_8C')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--app           : ',args.app)
    print ('--n_traces      : ',args.n_traces)
    print ('--model         : ',args.model)
    print(DIVIDER)
    
    collect(args.app, args.n_traces, args.model)
    return


if __name__ == '__main__':
    run_main()
