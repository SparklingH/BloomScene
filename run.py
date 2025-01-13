# -*- coding: utf-8 -*-
import os
import argparse
import torch
import datetime
import json
import random
import numpy as np
from PIL import Image
from bloomscene import BloomScene
from utils.metrics import clip_score_and_iqa, brisque_and_niqe_score


def fix_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    ### option
    parser = argparse.ArgumentParser(description='Arguments for BloomScene')
    # Input options
    parser.add_argument('--image', '-img', type=str, default='examples/01_childroom.png', help='Input image for scene generation')
    parser.add_argument('--text', '-t', type=str, default='examples/01_childroom.txt', help='Text prompt for scene generation')
    parser.add_argument('--neg_text', '-nt', type=str, default='', help='Negative text prompt for scene generation')

    # Camera options
    parser.add_argument('--campath_gen', '-cg', type=str, default='rotate360', choices=['rotate360'], help='Camera extrinsic trajectories for scene generation')
    parser.add_argument('--campath_render', '-cr', type=str, default='rotate360', choices=['rotate360'], help='Camera extrinsic trajectories for video rendering')

    # Inpainting options
    parser.add_argument('--seed', type=int, default=1, help='Manual seed for running Stable Diffusion inpainting')
    parser.add_argument('--diff_steps', type=int, default=50, help='Number of inference steps for running Stable Diffusion inpainting')

    # Save options
    parser.add_argument('--save_dir', '-s', type=str, default='', help='Save directory')

    # DPR options
    parser.add_argument('--dep_value', action='store_true', help='Pixel-level depth regularization or not')
    parser.add_argument('--dep_domin', action='store_true', help='Distribution-level depth regularization or not')
    parser.add_argument('--dep_smooth', action='store_true', help='Depth smoothness regularization or not')
    parser.add_argument('--dep_value_lbd', type=float, default=0.7, help='Depth regularization..')
    parser.add_argument('--dep_domin_lbd', type=float, default=0.1, help='Depth regularization..')
    parser.add_argument('--dep_smooth_lbd', type=float, default=1.0, help='Depth regularization..')

    # SCC options
    parser.add_argument("--n_features", type=int, default = 4)
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--lambdae", type=float, default = 0.002)
    parser.add_argument("--testing_iterations", nargs="+", type=int, default=[2990])
    parser.add_argument("--saving_iterations", nargs="+", type=int, default=[2990])


    args = parser.parse_args()

     
    fix_random_seed(args.seed)

    ### input (example)
    rgb_cond = Image.open(args.image).resize((512,512))

     
    if args.text.endswith('.txt'):
        with open(args.text, 'r') as f:
            txt_cond = f.readline()
    else:
        txt_cond = args.text

    if args.neg_text.endswith('.txt'):
        with open(args.neg_text, 'r') as f:
            neg_txt_cond = f.readline()
    else:
        neg_txt_cond = args.neg_text

    # Make default save directory if blank
     
    if args.save_dir == '':
        img_name = os.path.splitext(os.path.basename(args.image))[0]
        now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        args.save_dir = f'./outputs/{img_name}_{args.campath_gen}_{args.seed}_{now_str}'
        
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
     
    with open(os.path.join(args.save_dir, "settings.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)


    bs = BloomScene(args, save_dir=args.save_dir)
    start_time = datetime.datetime.now()
    print("start..", start_time.strftime('%Y-%m-%d  %H:%M:%S'))
    
    bs.create(rgb_cond, txt_cond, neg_txt_cond, args.campath_gen, args.seed, args.diff_steps)
    end_time = datetime.datetime.now()
    print("end..", end_time.strftime('%Y-%m-%d  %H:%M:%S'))

    bs.render_video(args.campath_render)


    # # eval
    image_folder = os.path.join(args.save_dir, "eval", "render_rgb")
    clip_score_and_iqa(image_folder=image_folder, text=txt_cond, out_path=args.save_dir)
    brisque_and_niqe_score(image_folder=image_folder, out_path=args.save_dir)
