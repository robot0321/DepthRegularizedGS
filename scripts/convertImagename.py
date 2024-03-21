import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgfolder", type=str) ### e.g.) data/nerf_llff_fewshot_resize/fern/images
    args = parser.parse_args()
    
    imglist = sorted(os.listdir(args.imgfolder))
    for i, imgname in enumerate(imglist):
        os.system(f"mv {os.path.join(args.imgfolder, imgname)} {os.path.join(args.imgfolder, str(i).zfill(5))}.png")
    