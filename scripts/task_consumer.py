

import time
import os
import sys
import argparse

METHOD_COMMAND_TEMPLATE = {
    ### NeRF-LLFF
    "1": "CUDA_VISIBLE_DEVICES=##GPU## python train.py -s data/nerf_llff_fewshot_resize/##SCENE## --eval --port 631##GPU## --model_path ##EXPPATH## --resolution 1 --kshot ##SHOT## --seed ##SEED##",
    "2": "CUDA_VISIBLE_DEVICES=##GPU## python train.py -s data/nerf_llff_fewshot_resize/##SCENE## --eval --port 631##GPU## --model_path ##EXPPATH## --resolution 1 --kshot ##SHOT## --seed ##SEED## --depth --usedepthReg", # (ours)

    ### MipNeRF360
    "11": "CUDA_VISIBLE_DEVICES=##GPU## python train.py -s data/mipnerf360_fewshot/##SCENE## --eval --port 631##GPU## --model_path ##EXPPATH## --resolution 1 --kshot ##SHOT## --seed ##SEED##",
    "12": "CUDA_VISIBLE_DEVICES=##GPU## python train.py -s data/mipnerf360_fewshot/##SCENE## --eval --port 631##GPU## --model_path ##EXPPATH## --resolution 1 --kshot ##SHOT## --seed ##SEED## --depth --usedepthReg", # (ours)
    
    ### DTU
    "21": "CUDA_VISIBLE_DEVICES=##GPU## python train.py -s data/DTU_fewshot/##SCENE##/##SCENE## --eval --port 631##GPU## --model_path ##EXPPATH## --resolution 1 --kshot ##SHOT## --seed ##SEED## --white_background",
    "22": "CUDA_VISIBLE_DEVICES=##GPU## python train.py -s data/DTU_fewshot/##SCENE##/##SCENE## --eval --port 631##GPU## --model_path ##EXPPATH## --resolution 1 --kshot ##SHOT## --seed ##SEED## --depth --usedepthReg --white_background", # (ours)
    
    ### NeRF-synthetic
    "31": "CUDA_VISIBLE_DEVICES=##GPU## python train.py -s data/nerf_synthetic_fewshot/##SCENE## --eval --port 631##GPU## --model_path ##EXPPATH## --resolution 1 --kshot ##SHOT## --seed ##SEED## --white_background",
    "32": "CUDA_VISIBLE_DEVICES=##GPU## python train.py -s data/nerf_synthetic_fewshot/##SCENE## --eval --port 631##GPU## --model_path ##EXPPATH## --resolution 1 --kshot ##SHOT## --seed ##SEED## --white_background --depth --usedepthReg", # (ours)
}
EXPERIMENT_PATH = "./output/baseline"
SEED_LIST_PATH = "./scripts/seed_list.txt"

def get_seed_list():
    with open(SEED_LIST_PATH, "r") as fp:
        retval = []
        while True:
            line = fp.readline()
            if not line:
                break
            seed = line.strip()
            retval.append(seed)
    
    return retval

SEED_LIST = get_seed_list()

def get_full_exp_path(base, method, scene, shot, seed):
    return os.path.join(base, f"method{method}", scene, f"{shot}_shot", seed)
def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        return "error"
    return "ok"

def get_task(task_list_path):
    with open(task_list_path, "r") as fp:
        while True:
            line = fp.readline()
            if not line:
                return None

            method, scene, shot, seed_id = line.strip().split("#")
            seed = SEED_LIST[int(seed_id)]
            full_exp_path = get_full_exp_path(EXPERIMENT_PATH, method, scene, shot, seed)
            if not os.path.exists(full_exp_path):
                return method, scene, shot, seed
            
import result_cleaner

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--tasklist", type=str, default="./scripts/all_tasks.txt")
    parser.add_argument("--rest", type=int, default=10)
    args = parser.parse_args()

    gpu = str(args.gpu)
    rest = args.rest

    while True:
        task = get_task(task_list_path=args.tasklist)
        
        if task is None:
            print("No task left. Bye")
            break
        
        method, scene, shot, seed = task
        print(f"GRABBED TASK: method={method} scene={scene} #shot={shot} seed={seed}")

        if method not in METHOD_COMMAND_TEMPLATE:
            raise NotImplementedError
        
        command = METHOD_COMMAND_TEMPLATE[method]
        
        exp_path = get_full_exp_path(EXPERIMENT_PATH, method, scene, shot, seed)
        
        command = command.replace("##GPU##", gpu)
        command = command.replace("##SCENE##", scene)
        command = command.replace("##EXPPATH##", exp_path)
        command = command.replace("##SHOT##", shot)
        command = command.replace("##SEED##", seed)

        cmd_return = do_system(command)

        if cmd_return == "error":
            pass
            os.makedirs(get_full_exp_path(EXPERIMENT_PATH, method, scene, shot, seed), exist_ok=True)
            # do_system(f"rm -rf {exp_path}")
            # exit(1)
        
        elif cmd_return == "ok":
            # CLEAR if not first seed experiment, except for metric
            
            # print(f"Rest for {rest}secs for GPU cooling...")
            # time.sleep(rest)
            if seed != SEED_LIST[0]:

                result_cleaner.clear_result_path(exp_path)
        else:
            raise NotImplementedError
        
        print(f"Rest for {rest}secs for GPU cooling...")
        time.sleep(rest)