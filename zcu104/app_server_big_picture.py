'''
Copyright 2020 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse

import subprocess
import socket
_divider = '-------------------------------'

PATH="/home/root/target_zcu104/"

### setup the variables for communication with host
TCP_IP = ''
TCP_PORT = 5005
BUFFER_SIZE = 8096

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((TCP_IP, TCP_PORT))
s.listen()

def preprocess_fn(image_path, fix_scale):
    '''
    Image pre-processing.
    Opens image as grayscale, adds channel dimension, normalizes to range 0:1
    and then scales by input quantization scaling factor
    input arg: path of image file
    return: numpy array
    '''
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.reshape(28,28,1)
    image = cv2.resize(image, (100, 100), interpolation = cv2.INTER_AREA)

    image = image * (1/255.0) * fix_scale
    image = image.astype(np.int8)
    return image


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id,start,dpu,img):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    #output_fixpos = outputTensors[0].get_attr("fix_point")
    #output_scale = 1 / (2**output_fixpos)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    ids=[]
    ids_max = 1
    outputData = []
    conn, addr = s.accept()

    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        data_string = ""
        ### WAIT FOR COMMANDS FROM THE HOST
        while 1:
            data_string = str(conn.recv(BUFFER_SIZE))[2:]
            if(not(data_string)):
                break
            while("]}" not in data_string):
                #print(data_string)
                data_string += str(conn.recv(BUFFER_SIZE))
            print("Data recieved :", data_string[:-3])
            if(data_string[:-3] == "1"):
                break
            elif(data_string[:-3] == "9"):
                #vart.vart_destroy_runner(dpu)
                conn.sendall(b'Exit DPU')
                print("Exiting")
                exit(0)


        ####TRIG
        subprocess.call(PATH+"trigger_up.sh")
        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData[len(ids)])
        ids.append((job_id,runSize,start+count))

        #print('this is batch', count)



        count = count + runSize
        if count<n_of_images:
            if len(ids) < ids_max-1:
                continue
        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]
            '''store output vectors '''
            for j in range(ids[index][1]):
                # we can avoid output scaling if use argmax instead of softmax
                # out_q[write_index] = np.argmax(outputData[0][j] * output_scale)
                #np.savetxt("crop9.csv", outputData[index][0][j][:,:,0], delimiter=',')
                out_q[write_index] = np.argmax(outputData[index][0][j])
                write_index += 1
        ids=[]
        conn.sendall(b']}')
        subprocess.call(PATH+"trigger_down.sh")


def app(image_dir,threads,model, n_images):

    listimage=os.listdir(image_dir)

    ### set the input to only 1 image
    listimage=[listimage[10]]*n_images

    #limit the number of images
    #listimage=listimage[:n_images]

    runTotal = len(listimage)

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    ''' preprocess images '''
    print (_divider)
    print('Pre-processing',runTotal,'images...')
    img = []
    for i in range(runTotal):
        path = os.path.join(image_dir,listimage[i])
        img.append(preprocess_fn(path, input_scale))

    '''run threads '''
    print (_divider)
    print('Starting',threads,'threads...')

    time1 = time.time()
    runDPU(1, 0, all_dpu_runners[0], img)
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print (_divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))


    ''' post-processing '''
    classes = ['zero','one','two','three','four','five','six','seven','eight','nine']
    correct = 0
    wrong = 0
    for i in range(len(out_q)):
        #print(out_q)
        prediction = classes[out_q[i]]
        ground_truth, _ = listimage[i].split('_',1)
        if (ground_truth==prediction):
            correct += 1
        else:
            wrong += 1
    accuracy = correct/len(out_q)
    print('Correct:%d, Wrong:%d, Accuracy:%.4f' %(correct,wrong,accuracy))
    print (_divider)
    return



# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d', '--image_dir', type=str, default=PATH+'images', help='Path to folder of images. Default is images')
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='CNN_zcu102.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')
  ap.add_argument('-n', '--n_images', type=int, default=100, help='Number of images processed 1-10k')
  args = ap.parse_args()

  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)
  print (' --n_images : ', args.n_images)

  app(args.image_dir,args.threads,args.model, args.n_images)

if __name__ == '__main__':
  main()