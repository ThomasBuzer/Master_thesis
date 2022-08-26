import os
import time
import random
import subprocess
from SSHLibrary import SSHLibrary

max_models = 1000
compiled_dir = "./build/compiled_model"

full_random_dir = os.path.join(compiled_dir, "full_random")
fixed_weight_dir = os.path.join(compiled_dir, "fixed_weight")

BOARD_IP = '131.174.142.183'
ZCU_PATH="/home/root/target_zcu104/"
ssh = SSHLibrary()
ssh.open_connection(BOARD_IP)
ssh.login("root", "root")


while(1):
    random = os.listdir(full_random_dir)
    random_ZCU_dir = ssh.execute_command("ls "+ZCU_PATH+"full_random/* | wc -l")
    if(len(random) > 10 and int(random_ZCU_dir) < max_models):
        print("Copying : ", os.path.join(full_random_dir, random[0]))
        print(full_random_dir+"/*")
        subprocess.run(["scp", "-r", full_random_dir+"", "root@131.174.142.183:/home/root/target_zcu104/"])
        for f in random:
            os.remove(os.path.join(full_random_dir, f))
    fixed = os.listdir(fixed_weight_dir)
    fixed_ZCU_dir = ssh.execute_command("ls "+ZCU_PATH+"fixed_weight/* | wc -l")
    if(len(fixed) > 10 and int(fixed_ZCU_dir) < max_models):
        print("Copying : ", os.path.join(fixed_weight_dir, fixed[0]))
        subprocess.run(["scp", "-r", fixed_weight_dir+"", "root@131.174.142.183:/home/root/target_zcu104/"])
        for f in fixed:
            os.remove(os.path.join(fixed_weight_dir, f))
    print("random : mako : ", len(random), "ZCU_board : ", random_ZCU_dir)
    print("fixed : mako : ", len(fixed), "ZCU_board : ", fixed_ZCU_dir)
    time.sleep(1)