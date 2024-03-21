
import os
import sys
from task_consumer import get_seed_list, get_full_exp_path, EXPERIMENT_PATH
from task_producer import SCENES, SHOTS, SEED_IDS
import numpy as np 

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=int, default=1)
    parser.add_argument("--gather", type=str, default="mean", 
                        help="if 'max', grab maximum statistic within target iteration range. if 'mean', grab an average instead.")
    args = parser.parse_args()
    
    SEED_LIST = get_seed_list()
    method = args.method
    output = {}

    with open(os.path.join(EXPERIMENT_PATH, f"method{method}", f"metric_{args.gather}.txt"), "w") as outfile:

        for scene in SCENES:
            output[scene] = {}
            print(f"=====[{scene}]=====", file=outfile)
            for shot in SHOTS:
                print(f"  [{shot}-SHOT]", file=outfile)
                output[scene][shot] = {}
                
                for seed_id in SEED_IDS:
                    seed = SEED_LIST[seed_id]
                    full_exp_path = get_full_exp_path(EXPERIMENT_PATH, method, scene, shot, seed)
                    metric_path = os.path.join(full_exp_path, "metric.txt")
                    try:
                        with open(metric_path, "r") as fp:
                            while True:
                                line = fp.readline()
                                if not line:
                                    break
                                
                                it, psnr, ssim, lpips = line.strip().split("_")
                                it = int(it)
                                # if not (iter_lb <= it <= iter_ub):
                                #     continue
                                psnr, ssim, lpips = float(psnr), float(ssim), float(lpips)

                                if it in output[scene][shot]:
                                    psnr_sum, ssim_sum, lpips_sum, cnt = output[scene][shot][it]
                                else:
                                    psnr_sum, ssim_sum, lpips_sum, cnt = 0., 0., 0., 0
                                psnr_sum += psnr
                                ssim_sum += ssim
                                lpips_sum += lpips
                                cnt += 1

                                output[scene][shot][it] = (psnr_sum, ssim_sum, lpips_sum, cnt)
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        print(e)
                        exit(0)
                
                # gather.
                output[scene][shot]["psnr"] = []
                output[scene][shot]["ssim"] = []
                output[scene][shot]["lpips"] = []
                
                for it, result in output[scene][shot].items():
                    if isinstance(result, list):
                        continue
                    psnr_sum, ssim_sum, lpips_sum, cnt = result
                    if cnt != 0:
                        psnr = psnr_sum / cnt
                        ssim = ssim_sum / cnt
                        lpips = lpips_sum / cnt
                    output[scene][shot]["psnr"].append(psnr)
                    output[scene][shot]["ssim"].append(ssim)
                    output[scene][shot]["lpips"].append(lpips)
                    

                if len(output[scene][shot]['psnr']) == 0:
                    print("None", file=outfile)
                    continue

                for key in ("psnr","ssim","lpips"):
                    if args.gather == "max":
                        output[scene][shot][key] = max(output[scene][shot][key]) if key != "lpips" else min(output[scene][shot][key])
                    elif args.gather == "mean":
                        length = len(output[scene][shot][key])
                        output[scene][shot][key] = sum(output[scene][shot][key]) / length
                    else:
                        raise NotImplementedError

                print(f"PSNR:{output[scene][shot]['psnr']:6.2f} SSIM:{output[scene][shot]['ssim']:6.3f} LPIPS:{output[scene][shot]['lpips']:6.3f}", file=outfile)
                
            print("",file=outfile)

        
        # Average.
        print("============================\n", file=outfile)

        print(f"=====[Average]=====", file=outfile)
        for shot in SHOTS:
            print(f"  [{shot}-SHOT]", file=outfile)
            psnr_all = 0.0
            ssim_all = 0.0
            lpips_all = 0.0
            cnt_all = 0
            for scene in SCENES:
                if isinstance(output[scene][shot]['psnr'], float):
                    psnr_all += output[scene][shot]['psnr']
                    ssim_all += output[scene][shot]['ssim']
                    lpips_all += output[scene][shot]['lpips']
                    cnt_all += 1
            if cnt_all != 0:
                psnr = psnr_all / cnt_all
                ssim = ssim_all / cnt_all
                lpips = lpips_all / cnt_all

                print(f"PSNR:{psnr:6.2f} SSIM:{ssim:6.3f} LPIPS:{lpips:6.3f}", file=outfile)
            else:
                print(f"[None]", file=outfile)