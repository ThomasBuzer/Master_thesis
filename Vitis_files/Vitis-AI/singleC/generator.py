import os
import time
import mmap
import random
import subprocess

def hex_to_uint(hex):
    value = 0
    for i, h in enumerate(reversed(hex)):
        print(h, (2**(8*i)))
        value += h * (2**(8*i))
    return value

def int_to_hex(integer):
    if integer >= 0:
        hexa = hex(integer)[2:]
        return '0'*(4-len(hexa))+hexa
    else:
        return hex(int('1'+bin(32768+integer)[2:], 2))[2:]

def hex_to_2ints(hex):
    return [int(hex[:2], 16), int(hex[2:], 16)]

def generate_random_weigth():
    return [0, 0] + hex_to_2ints(int_to_hex(random.randint(-int_range, int_range)))

def generate_random_weigths(n):
    return [generate_random_weigth() for i in range(n)]


max_models = 1000
compiled_dir = "./build/compiled_model"

full_random_dir = os.path.join(compiled_dir, "full_random")
fixed_weight_dir = os.path.join(compiled_dir, "fixed_weight")

file = open("./build/quant_model/CNN_int.xmodel", 'r+b')

mm = mmap.mmap(file.fileno(), 0)

start= 677 #640
length = 400 #18

int_range= 10000
fixed_weight = [0, 0, 209, 108]
fixed_weights = generate_random_weigths(length-1)

# hexs = [mm[start+i*4:start+(i+1)*4] for i in range(length)]
# print(hexs)
#
# weights = generate_random_weigths(length)
# print(weights)
#
# for i in range(length):
#     #print(mm[start+i*4], weights[i][0])
#     mm[start+i*4] = weights[i][0]
#     mm[start+i*4+1] = weights[i][1]
#     mm[start+i*4+2] = weights[i][2]
#     mm[start+i*4+3] = weights[i][3]

# hexs = [mm[start+i*4:start+(i+1)*4] for i in range(length)]
# print(hexs)



while 1:
    
    flip = random.random()
    if(flip < 0.5):
        #FIXED
        if(len(os.listdir(fixed_weight_dir))>max_models):
            continue
        weights = [fixed_weight] + fixed_weights#generate_random_weigths(length - 1)
    else:
        if(len(os.listdir(full_random_dir))>max_models):
            continue
        weights = [generate_random_weigth()] + fixed_weights
    
    for i in range(length):
        mm[start+i*4] = weights[i][0]
        mm[start+i*4+1] = weights[i][1]
        mm[start+i*4+2] = weights[i][2]
        mm[start+i*4+3] = weights[i][3]
    
    
    subprocess.run("./compile.sh")
    if(flip < 0.5):
        os.replace("./build/compiled_model/CNN_zcu104.xmodel", "./build/compiled_model/fixed_weight/fixed_"+str(int(time.time()*100))+".xmodel")
        print("saving : fixed_"+str(int(time.time()*100)) )
    else :
        os.replace("./build/compiled_model/CNN_zcu104.xmodel", "./build/compiled_model/full_random/random_"+str(int(time.time()*100))+".xmodel")    
        print("saving : random_"+str(int(time.time()*100)) )
    time.sleep(0.0001)




mm.close()

