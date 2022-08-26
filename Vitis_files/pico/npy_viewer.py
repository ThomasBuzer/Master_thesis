from array import array
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

DIVIDER = '\n______________________\n'

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', type=str, default='./traces/trace.trs', help='Path to trs file. Default is ./traces/trace.trs')
ap.add_argument('-lf', '--label_file', type=str, default='./traces/trace.trs', help='Path to trs file. Default is ./traces/trace.trs')
args = ap.parse_args()

array = np.load(args.file)
labels = np.load(args.label_file)

counter = 0

while 1:
    #i = random.randint(0, len(array))
    i = counter * 5
    print(DIVIDER)
    print("Labels are : ", labels[i])
    print(DIVIDER)
    plt.plot(array[i])
    plt.show()
    counter += 1


