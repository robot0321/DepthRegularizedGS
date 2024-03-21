####### Dataset settings #######
## NeRF-LLFF
SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
SHOTS = [2, 3, 5]

## MipNeRF360
# SCENES = ["bicycle", "bonsai", "counter", "garden", "kitchen", "stump"]
# SHOTS = [6,9,12,18,24]

## DTU
# SCENES = ['scan31', 'scan40', 'scan34', 'scan55', 'scan110', 'scan45', 'scan63', 'scan114', 'scan82', 'scan41']
# SCENES = ['scan65', 'scan106', 'scan118']
# SHOTS = [3,6,9,12]

## NeRF-synthetic
# SCENES = ["chair", "drums", "hotdog", "lego"]
# SHOTS = [3,6,9,12,18]
###############################

### The method number is describe in the `scripts/task_consumer.py` (see METHOD_COMMAND_TEMPLATE -- e.g. 1: LLFF/3DGS, 2: LLFF/Ours)
METHODS = [1, 2] # [1, 2, 11, 12, 21, 22, 31, 32] 

### The list of seeds are written in `scripts/seed_list.txt`. You can add/edit the seeds as you like.
SEED_IDS = [0, 1] # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

### the method numbers which you want to run first 
METHOD_URGENT = [] 

if __name__ == "__main__":
    with open(f"scripts/all_tasks.txt", "w") as fp:
        for shot in SHOTS:
            for scene in SCENES:
                for seed_id in SEED_IDS:
                    for method in METHODS:
                        if method in METHOD_URGENT:    
                            print(f"{method}#{scene}#{shot}#{seed_id}", file=fp)
                        else: # not in Method URGENT
                            print(f"{method}#{scene}#{shot}#{seed_id}", file=fp)


