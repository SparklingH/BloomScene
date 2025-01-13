import os
import time
import datetime
import imageio
import sys
import wandb
import torch
import torchvision
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from random import randint
from os import makedirs
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import griddata as interp_grid
from scipy.ndimage import minimum_filter, maximum_filter
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler

wandb.init(project='BloomScene_final')


from arguments import GSParams, CameraParams
from gaussian_renderer import render, prefilter_voxel
from scene import Scene, GaussianModel
from utils.loss import l1_loss, ssim, DepthLoss, DepthLossType, CMD, bilateral_filter
from utils.camera import load_json
from utils.depth import colorize
from utils.trajectory import get_pcdGenPoses
from utils.general import LatentStorer
from utils.encodings import get_binary_vxl_size


bit2MB_scale = 8 * 1024 * 1024
run_codec = True
get_kernel = lambda p: torch.ones(1, 1, p * 2 + 1, p * 2 + 1).to('cuda')
t2np = lambda x: (x[0].permute(1, 2, 0).clamp_(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
np2t = lambda x: (torch.as_tensor(x).to(torch.float32).permute(2, 0, 1) / 255.0)[None, ...].to('cuda')
pad_mask = lambda x, padamount=1: t2np(
    F.conv2d(np2t(x[..., None]), get_kernel(padamount), padding=padamount))[..., 0].astype(bool)


class BloomScene:
    def __init__(self, args, save_dir=None):
        self.args = args        
        self.opt = GSParams()   
        self.cam = CameraParams()   
        self.save_dir = save_dir 
        self.root = 'outputs'
        self.default_model = 'SD1.5 (default)'
        self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

         
        self.gaussians = GaussianModel(
            self.opt.feat_dim,       
            self.opt.n_offsets,      
            self.opt.voxel_size,     
            self.opt.update_depth,          
            self.opt.update_init_factor,        
            self.opt.update_hierachy_factor,     
            self.opt.use_feat_bank,              
            n_features_per_level=self.args.n_features,  
            log2_hashmap_size=self.args.log2,           
            log2_hashmap_size_2D=self.args.log2_2D,    
        )

         
        bg_color = [1, 1, 1] if self.opt.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
        
        self.rgb_model = StableDiffusionInpaintPipeline.from_pretrained(
            "./models--runwayml--stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        ).to('cuda')
        self.rgb_model.scheduler = DDIMScheduler.from_config(self.rgb_model.scheduler.config)
        self.latent_storer = LatentStorer()

        # monocular depth model
        self.d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')

        self.controlnet = None
        self.lama = None
        self.current_model = self.default_model


    def rgb(self, prompt, image, negative_prompt='', generator=None, num_inference_steps=50, mask_image=None):
        image_pil = Image.fromarray(np.round(image * 255.).astype(np.uint8))
        mask_pil = Image.fromarray(np.round((1 - mask_image) * 255.).astype(np.uint8))
        if self.current_model == self.default_model:
            return self.rgb_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,  
                image=image_pil,
                mask_image=mask_pil,
                callback_steps=num_inference_steps-1,
                callback=self.latent_storer
            ).images[0]

        kwargs = {
            'negative_prompt': negative_prompt,
            'generator': generator,
            'strength': 0.9,
            'num_inference_steps': num_inference_steps,
            'height': self.cam.H,
            'width': self.cam.W,
        }

        image_np = np.round(np.clip(image, 0, 1) * 255.).astype(np.uint8)
        mask_sum = np.clip((image.prod(axis=-1) == 0) + (1 - mask_image), 0, 1)
        mask_padded = pad_mask(mask_sum, 3)
        masked = image_np * np.logical_not(mask_padded[..., None])

        if self.lama is not None:
            lama_image = Image.fromarray(self.lama(masked, mask_padded).astype(np.uint8))
        else:
            lama_image = image

        mask_image = Image.fromarray(mask_padded.astype(np.uint8) * 255)
        control_image = self.make_controlnet_inpaint_condition(lama_image, mask_image)

        return self.rgb_model(
            prompt=prompt,
            image=lama_image,
            control_image=control_image,
            mask_image=mask_image,
            callback_steps=num_inference_steps-1,
            callback=self.latent_storer,
            **kwargs,
        ).images[0]


    def d(self, im):
        return self.d_model.infer_pil(im)      


    def make_controlnet_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image


    def create(self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.traindata = self.generate_pcd(rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps)
        self.scene = Scene(self.traindata, self.gaussians, self.opt)     
        self.x_bound_min, self.x_bound_max = self.training()   
        self.save_image()
        self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    
    def save_ply(self, fpath=None):
        if fpath is None:
            dpath = os.path.join(self.root, self.timestamp)
            fpath = os.path.join(dpath, 'gsplat.ply')
            os.makedirs(dpath, exist_ok=True)
        if not os.path.exists(fpath):
            self.gaussians.save_ply(fpath)
        else:
            self.gaussians.load_ply(fpath)
        return fpath


    def render_video(self, preset):
        videopath = os.path.join(self.save_dir, f'{preset}.mp4')
        depthpath = os.path.join(self.save_dir, f'depth_{preset}.mp4')
        if os.path.exists(videopath) and os.path.exists(depthpath):
            return videopath, depthpath
        
        if not hasattr(self, 'scene'):
            views = load_json(os.path.join('cameras', f'{preset}.json'), self.cam.H, self.cam.W)
        else:
            views = self.scene.getPresetCameras(preset)
        
        framelist = []
        depthlist = []
        dmin, dmax = 1e8, -1e8

        iterable_render = views
        
        for idx, view in enumerate(iterable_render):
            voxel_visible_mask = prefilter_voxel(view, self.gaussians, self.opt, self.background)
            results = render(view, self.gaussians, self.opt, self.background, visible_mask=voxel_visible_mask)

            frame, depth = results['render'], results['depth']
            output_360_rgb_path = os.path.join(self.save_dir, 'eval', '360_rgb')
            if not os.path.exists(output_360_rgb_path):
                os.makedirs(output_360_rgb_path)
            torchvision.utils.save_image(frame, os.path.join(output_360_rgb_path, '{0:05d}.png'.format(idx)))
            
            framelist.append(
                np.round(frame.permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
            
            depth = (depth * (depth > 0)).detach().cpu().numpy()
            dmin_local = depth.min().item()
            dmax_local = depth.max().item()
            if dmin_local < dmin:
                dmin = dmin_local
            if dmax_local > dmax:
                dmax = dmax_local
            depthlist.append(depth)


        depthlist = [colorize(depth) for depth in depthlist]
        if not os.path.exists(videopath):
            imageio.mimwrite(videopath, framelist, fps=30, quality=8)
        if not os.path.exists(depthpath):
            imageio.mimwrite(depthpath, depthlist, fps=30, quality=8)
        return videopath, depthpath


    def training(self):
        if not self.scene:
            raise('Build 3D Scene First!')
        
        iterable_gauss = range(1, self.opt.iterations + 1) 
        self.gaussians.update_anchor_bound()
        self.gaussians.training_setup(self.opt)
        pbar = tqdm(iterable_gauss, miniters=10, file=sys.stdout)

        log_time_sub = 0
        for iteration in pbar:
            torch.cuda.empty_cache()
            self.gaussians.update_learning_rate(iteration)

            # Pick a random Camera
            viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
             
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, self.gaussians, self.opt, self.background)
            retain_grad = (iteration < self.opt.update_until and iteration >= 0)

            render_pkg = render(viewpoint_cam, self.gaussians, self.opt, self.background, scaling_modifier = 1.0, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
            bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param = render_pkg["bit_per_param"], render_pkg["bit_per_feat_param"], render_pkg["bit_per_scaling_param"], render_pkg["bit_per_offsets_param"]

            # torch.cuda.synchronize(); t_start_log = time.time()
            if iteration % 20 == 0 and bit_per_param is not None:
                ttl_size_feat_MB = bit_per_feat_param.item() * self.gaussians.get_anchor.shape[0] * self.gaussians.feat_dim / bit2MB_scale
                ttl_size_scaling_MB = bit_per_scaling_param.item() * self.gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
                ttl_size_offsets_MB = bit_per_offsets_param.item() * self.gaussians.get_anchor.shape[0] * 3 * self.gaussians.n_offsets / bit2MB_scale
                ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

                wandb.log({
                    "iteration": iteration,
                    "bit_per_feat_param": bit_per_feat_param.item(),
                    "anchor_num": self.gaussians.get_anchor.shape[0],
                    "ttl_size_feat_MB": ttl_size_feat_MB,
                    "bit_per_scaling_param": bit_per_scaling_param.item(),
                    "ttl_size_scaling_MB": ttl_size_scaling_MB,
                    "bit_per_offsets_param": bit_per_offsets_param.item(),
                    "ttl_size_offsets_MB": ttl_size_offsets_MB,
                    "bit_per_param": bit_per_param.item(),
                    "ttl_size_MB": ttl_size_MB
                })

                with torch.no_grad():
                    grid_masks = self.gaussians._mask.data
                    binary_grid_masks = (torch.sigmoid(grid_masks) > 0.01).float()
                    mask_1_rate, mask_size_bit, mask_size_MB, mask_numel = get_binary_vxl_size(binary_grid_masks + 0.0)  

                # Logging mask information to wandb
                wandb.log({
                    "iteration": iteration,
                    "mask_1_rate": mask_1_rate,
                    "mask_numel": mask_numel,
                    "mask_size_MB": mask_size_MB
                })
            # torch.cuda.synchronize(); t_end_log = time.time()
            # t_log = t_end_log - t_start_log
            # log_time_sub += t_log

            # Loss
            gt_image = viewpoint_cam.original_image.cuda() 
            Ll1 = l1_loss(image, gt_image)
            loss_rgb = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss = loss_rgb
            
            scaling_reg = render_pkg["scaling"].prod(dim=1).mean()
            loss += 0.01*scaling_reg

            if bit_per_param is not None:
                loss = loss + self.args.lambdae * bit_per_param

            loss_dep_value = torch.tensor(0.0)
            loss_dep_domin = torch.tensor(0.0)
            loss_depth_smooth = torch.tensor(0.0)
            original_depth = viewpoint_cam.original_depth.cuda() 
            max_ori_depth = original_depth.max()
            min_ori_depth = original_depth.min()
            original_depth = (original_depth - min_ori_depth) / (max_ori_depth - min_ori_depth + 1e-8)
            render_depth = render_pkg['depth']               
            max_render_depth = render_depth.max()
            min_render_depth = render_depth.min()
            render_depth = (render_depth - min_render_depth) / (max_render_depth - min_render_depth + 1e-8) 

            if self.args.dep_value:
                this_image = gt_image.permute(2, 1, 0).unsqueeze(0) 
                this_ori_depth = original_depth.unsqueeze(0).unsqueeze(-1) 
                this_render_depth = render_depth.unsqueeze(-1) 
                depth_loss_type = DepthLossType.HuberL1
                depthloss = DepthLoss(depth_loss_type=depth_loss_type)
                loss_dep_value = self.args.dep_value_lbd * depthloss.forward(this_render_depth, this_ori_depth, this_image)
                loss = loss + loss_dep_value

            if self.args.dep_domin:
                target_depth = original_depth.unsqueeze(0).unsqueeze(0)  
                source_depth = render_depth.unsqueeze(0)    
                loss_dep_domin = self.args.dep_domin_lbd * CMD().forward(source_depth, target_depth)
                loss = loss + loss_dep_domin

            if self.args.dep_smooth:
                smooth_depth = render_depth  
                loss_depth_smooth = self.args.dep_smooth_lbd * bilateral_filter(smooth_depth, spatial_sigma=2.0, color_sigma=5.0)
                loss = loss + loss_depth_smooth

            loss.backward()
            str_descrip = f'Iteration {iteration:03d} :' \
                    + f' l_rgb: {loss_rgb:.6f}' \
                    + f' l_dep_value: {(loss_dep_value):.6f}' \
                    + f' l_dep_domin: {loss_dep_domin:.6f}' \
                    + f' l_dep_smooth: {loss_depth_smooth:.6f}'
            pbar.set_description(str_descrip)
            
            with torch.no_grad():
                # Log and save
                self.training_report(iteration)
                if (iteration in self.args.saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    self.gaussians.save_mlp_checkpoints(os.path.join(self.args.save_dir, "checkpoint.pth"))
                    self.gaussians.save_ply(os.path.join(self.args.save_dir, "gsplat.ply"))

                # Densification
                if iteration < self.opt.update_until and iteration > self.opt.start_stat: 
                    self.gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                    if iteration not in range(1000, 1500):  # let the model get fit to quantization
                        # densification
                        if iteration > self.opt.update_from and iteration % self.opt.update_interval == 0:  
                            self.gaussians.adjust_anchor(check_interval=self.opt.update_interval, success_threshold=self.opt.success_threshold, grad_threshold=self.opt.densify_grad_threshold, min_opacity=self.opt.min_opacity)
                elif iteration == self.opt.update_until:
                    del self.gaussians.opacity_accum
                    del self.gaussians.offset_gradient_accum
                    del self.gaussians.offset_denom
                    torch.cuda.empty_cache()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)
                
        return self.gaussians.x_bound_min, self.gaussians.x_bound_max


    def training_report(self, iteration):
        # Report test and samples of training set
        if iteration in self.args.testing_iterations:
            self.gaussians.eval()
            if 1:
                if iteration == self.args.testing_iterations[-1]:
                    with torch.no_grad():
                        log_info = self.gaussians.estimate_final_bits()
                    if run_codec:  # conduct encoding and decoding
                        with torch.no_grad():
                            bit_stream_path = os.path.join(self.args.save_dir, 'bitstreams')
                            os.makedirs(bit_stream_path, exist_ok=True)
                            # conduct encoding
                            patched_infos = self.gaussians.conduct_encoding(pre_path_name=bit_stream_path)
                            # conduct decoding
                            self.gaussians.conduct_decoding(pre_path_name=bit_stream_path, patched_infos=patched_infos)
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            self.gaussians.train()


    def render_set(self, name, views):
        render_path = os.path.join(self.args.save_dir, name, "render_rgb")
        makedirs(render_path, exist_ok=True)

        t_list = []
        visible_count_list = []
        name_list = []
        per_view_dict = {}
        for idx, view in enumerate(tqdm(views, desc="Rendering progress (Save eval image..)")):
            # torch.cuda.synchronize(); t_start = time.time()
            voxel_visible_mask = prefilter_voxel(view, self.gaussians, self.opt, self.background)
            render_pkg = render(view, self.gaussians, self.opt, self.background, visible_mask=voxel_visible_mask)
            # torch.cuda.synchronize(); t_end = time.time()
            # t_list.append(t_end - t_start)

            # renders
            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            visible_count = (render_pkg["radii"] > 0).sum()
            visible_count_list.append(visible_count)
            name_list.append('{0:05d}'.format(idx) + ".png")
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
        return t_list, visible_count_list


    def render_sets(self, x_bound_min=None, x_bound_max=None):
        with torch.no_grad():
            self.gaussians.load_mlp_checkpoints(os.path.join(self.args.save_dir, "checkpoint.pth"))
            self.gaussians.eval()
            if x_bound_min is not None:
                self.gaussians.x_bound_min = x_bound_min
                self.gaussians.x_bound_max = x_bound_max

            t_train_list, _  = self.render_set("eval", self.scene.getEvalCameras())
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            wandb.log({"eval_fps":train_fps.item(), })


    def save_image(self):
        self.render_sets(self.x_bound_min, self.x_bound_max)


    def generate_pcd(self, rgb_cond, prompt, negative_prompt, pcdgenpath, seed, diff_steps):
        generator=torch.Generator(device='cuda').manual_seed(seed)

        w_in, h_in = rgb_cond.size   
        if w_in/h_in > 1.1 or h_in/w_in > 1.1: # if height and width are similar, do center crop
            in_res = max(w_in, h_in)
            image_in, mask_in = np.zeros((in_res, in_res, 3), dtype=np.uint8), 255*np.ones((in_res, in_res, 3), dtype=np.uint8)
            image_in[int(in_res/2-h_in/2):int(in_res/2+h_in/2), int(in_res/2-w_in/2):int(in_res/2+w_in/2)] = np.array(rgb_cond)
            mask_in[int(in_res/2-h_in/2):int(in_res/2+h_in/2), int(in_res/2-w_in/2):int(in_res/2+w_in/2)] = 0

            image2 = np.array(Image.fromarray(image_in).resize((self.cam.W, self.cam.H))).astype(float) / 255.0
            mask2 = np.array(Image.fromarray(mask_in).resize((self.cam.W, self.cam.H))).astype(float) / 255.0

            image_curr = self.rgb(
                prompt=prompt,
                image=image2,
                negative_prompt=negative_prompt, generator=generator,
                mask_image=mask2,
            )

        else: # if there is a large gap between height and width, do inpainting
            if w_in > h_in:
                image_curr = rgb_cond.crop((int(w_in/2-h_in/2), 0, int(w_in/2+h_in/2), h_in)).resize((self.cam.W, self.cam.H))
            else: # w <= h
                 
                image_curr = rgb_cond.crop((0, int(h_in/2-w_in/2), w_in, int(h_in/2+w_in/2))).resize((self.cam.W, self.cam.H))

        render_poses = get_pcdGenPoses(pcdgenpath)   
        depth_curr = self.d(image_curr) 
        center_depth_list =[]
        center_depth = np.mean(depth_curr[h_in//2-10:h_in//2+10, w_in//2-10:w_in//2+10])
        center_depth_list.append(center_depth)
          
        # Iterative scene generation
        H, W, K = self.cam.H, self.cam.W, self.cam.K
        x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        edgeN = 2
        edgemask = np.ones((H-2*edgeN, W-2*edgeN))
        edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN))) 

        ### initialize 
        R0, T0 = render_poses[0,:3,:3], render_poses[0,:3,3:4]   
        pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))   
        new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32)  
        new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)  
        pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()
        iterable_dream = range(1, len(render_poses))   

        for i in iterable_dream: 
            torch.cuda.empty_cache()
            R, T = render_poses[i,:3,:3], render_poses[i,:3,3:4]
            pts_coord_cam2 = R.dot(pts_coord_world) + T   
            pixel_coord_cam2 = np.matmul(K, pts_coord_cam2)    

            valid_idx = np.where(np.logical_and.reduce((pixel_coord_cam2[2]>0, 
                                                        pixel_coord_cam2[0]/pixel_coord_cam2[2]>=0,          
                                                        pixel_coord_cam2[0]/pixel_coord_cam2[2]<=W-1, 
                                                        pixel_coord_cam2[1]/pixel_coord_cam2[2]>=0,          
                                                        pixel_coord_cam2[1]/pixel_coord_cam2[2]<=H-1)))[0]   
            pixel_coord_cam2 = pixel_coord_cam2[:2, valid_idx]/pixel_coord_cam2[-1:, valid_idx]              
            round_coord_cam2 = np.round(pixel_coord_cam2).astype(np.int32)                                   
            x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
            grid = np.stack((x,y), axis=-1).reshape(-1,2)

            image2 = interp_grid(pixel_coord_cam2.transpose(1,0), pts_colors[valid_idx], grid, method='linear', fill_value=0).reshape(H,W,3)
            image2 = edgemask[...,None]*image2 + (1-edgemask[...,None])*np.pad(image2[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')
            round_mask2 = np.zeros((H,W), dtype=np.float32)
            round_mask2[round_coord_cam2[1], round_coord_cam2[0]] = 1   
            round_mask2 = maximum_filter(round_mask2, size=(9,9), axes=(0,1))
            image2 = round_mask2[...,None]*image2 + (1-round_mask2[...,None])*(-1)
            mask2 = minimum_filter((image2.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))         
            image2 = mask2[...,None]*image2 + (1-mask2[...,None])*0                          

            mask_hf = np.abs(mask2[:H-1, :W-1] - mask2[1:, :W-1]) + np.abs(mask2[:H-1, :W-1] - mask2[:H-1, 1:])
            mask_hf = np.pad(mask_hf, ((0,1), (0,1)), 'edge')
            mask_hf = np.where(mask_hf < 0.3, 0, 1)                        
            border_valid_idx = np.where(mask_hf[round_coord_cam2[1], round_coord_cam2[0]] == 1)[0]  # use valid_idx[border_valid_idx] for world1

            # inpaint
            image_curr = self.rgb(
                prompt=prompt, image=image2, 
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=diff_steps,
                mask_image=mask2 
            )
            depth_curr = self.d(image_curr)                          
            center_depth = np.mean(depth_curr[h_in//2-10:h_in//2+10, w_in//2-10:w_in//2+10])
            center_depth_list.append(center_depth)


            ### depth optimize
            t_z2 = torch.tensor(depth_curr)
            sc = torch.ones(1).float().requires_grad_(True)          
            optimizer = torch.optim.Adam(params=[sc], lr=0.001)      

            for idx in range(100):
                trans3d = torch.tensor([[sc,0,0,0], [0,sc,0,0], [0,0,sc,0], [0,0,0,1]]).requires_grad_(True)
                coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1], round_coord_cam2[0]].reshape(3,-1))       
                coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))                                                  
                coord_world2_warp = torch.cat((coord_world2, torch.ones((1,valid_idx.shape[0]))), dim=0)          
                coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
                coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]
                loss = torch.mean((torch.tensor(pts_coord_world[:,valid_idx]).float() - coord_world2_trans)**2)   

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    

            with torch.no_grad():
                coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1, border_valid_idx], round_coord_cam2[0, border_valid_idx]].reshape(3,-1))
                coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
                coord_world2_warp = torch.cat((coord_world2, torch.ones((1, border_valid_idx.shape[0]))), dim=0)
                coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
                coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]

            trans3d = trans3d.detach().numpy()
            pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
            camera_origin_coord_world2 = - np.linalg.inv(R).dot(T).astype(np.float32)                                                
            new_pts_coord_world2 = (np.linalg.inv(R).dot(pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)               
            new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
            new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
            new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]                                               
            new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]


            vector_camorigin_to_campixels = coord_world2_trans.detach().numpy() - camera_origin_coord_world2                 
            vector_camorigin_to_pcdpixels = pts_coord_world[:,valid_idx[border_valid_idx]] - camera_origin_coord_world2      
            compensate_depth_coeff = np.sum(vector_camorigin_to_pcdpixels * vector_camorigin_to_campixels, axis=0) / np.sum(vector_camorigin_to_campixels * vector_camorigin_to_campixels, axis=0)
            compensate_pts_coord_world2_correspond = camera_origin_coord_world2 + vector_camorigin_to_campixels * compensate_depth_coeff.reshape(1,-1)   
            compensate_coord_cam2_correspond = R.dot(compensate_pts_coord_world2_correspond) + T
            homography_coord_cam2_correspond = R.dot(coord_world2_trans.detach().numpy()) + T
            compensate_depth_correspond = compensate_coord_cam2_correspond[-1] - homography_coord_cam2_correspond[-1]
            compensate_depth_zero = np.zeros(4)
            compensate_depth = np.concatenate((compensate_depth_correspond, compensate_depth_zero), axis=0)

            pixel_cam2_correspond = pixel_coord_cam2[:, border_valid_idx]
            pixel_cam2_zero = np.array([[0,0,W-1,W-1],[0,H-1,0,H-1]])
            pixel_cam2 = np.concatenate((pixel_cam2_correspond, pixel_cam2_zero), axis=1).transpose(1,0)
            masked_pixels_xy = np.stack(np.where(1-mask2), axis=1)[:, [1,0]]                                            
            new_depth_linear, new_depth_nearest = interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy), interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy, method='nearest')
            new_depth = np.where(np.isnan(new_depth_linear), new_depth_nearest, new_depth_linear)

            pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]         
            x_nonmask, y_nonmask = x.reshape(-1)[np.where(1-mask2.reshape(-1))[0]], y.reshape(-1)[np.where(1-mask2.reshape(-1))[0]]
            compensate_pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x_nonmask*new_depth, y_nonmask*new_depth, 1*new_depth), axis=0))   
            new_warp_pts_coord_cam2 = pts_coord_cam2 + compensate_pts_coord_cam2

            new_pts_coord_world2 = (np.linalg.inv(R).dot(new_warp_pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
            new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
            new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
            new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
            new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]
            
            pts_coord_world = np.concatenate((pts_coord_world, new_pts_coord_world2), axis=-1) ### Same with inv(c2w) * cam_coord (in homogeneous space)
            pts_colors = np.concatenate((pts_colors, new_pts_colors2), axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_coord_world.T.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(pts_colors.astype(np.float32))
        o3d.io.write_point_cloud(os.path.join(self.args.save_dir, "point_cloud.ply"), pcd)


        yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
        traindata = {
            'camera_angle_x': self.cam.fov[0],
            'W': W,
            'H': H,
            'pcd_points': pts_coord_world,
            'pcd_colors': pts_colors,
            'frames': [],
        }

        internel_render_poses = get_pcdGenPoses('hemisphere', {'center_depth': center_depth_list})        
        iterable_align = range(len(render_poses))
        for i in iterable_align:
            for j in range(int(len(internel_render_poses) / len(render_poses))):
                torch.cuda.empty_cache()
                inter_idx = (int(len(internel_render_poses) / len(render_poses))) * i + j
                print(f'{inter_idx+1} / {len(internel_render_poses)}')       

                Rw2i = render_poses[i,:3,:3]
                Tw2i = render_poses[i,:3,3:4]
                Ri2j = internel_render_poses[inter_idx,:3,:3]
                Ti2j = internel_render_poses[inter_idx,:3,3:4]

                Rw2j = np.matmul(Ri2j, Rw2i)             
                Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j
                Rj2w = np.matmul(yz_reverse, Rw2j).T
                Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
                Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
                Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)     

                pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j        
                pixel_coord_camj = np.matmul(K, pts_coord_camj)          
                valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0, 
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]>=0, 
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]<=W-1, 
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]>=0, 
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]<=H-1)))[0]
                if len(valid_idxj) == 0:         
                    continue
                 
                pts_depthsj = pixel_coord_camj[-1:, valid_idxj]
                pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]    
                round_coord_camj = np.round(pixel_coord_camj).astype(np.int32)

                x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
                grid = np.stack((x,y), axis=-1).reshape(-1,2)
                imagej = interp_grid(pixel_coord_camj.transpose(1,0), pts_colors[valid_idxj], grid, method='linear', fill_value=0).reshape(H,W,3)
                imagej = edgemask[...,None]*imagej + (1-edgemask[...,None])*np.pad(imagej[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

                depthj = interp_grid(pixel_coord_camj.transpose(1,0), pts_depthsj.T, grid, method='linear', fill_value=0).reshape(H,W)
                depthj = edgemask*depthj + (1-edgemask)*np.pad(depthj[1:-1,1:-1], ((1,1),(1,1)), mode='edge')

                maskj = np.zeros((H,W), dtype=np.float32)
                maskj[round_coord_camj[1], round_coord_camj[0]] = 1
                maskj = maximum_filter(maskj, size=(9,9), axes=(0,1))
                imagej = maskj[...,None]*imagej + (1-maskj[...,None])*(-1)
                maskj = minimum_filter((imagej.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))
                imagej = maskj[...,None]*imagej + (1-maskj[...,None])*0

                depth_pred = self.d(Image.fromarray(np.round(imagej*255.).astype(np.uint8)))
                traindata['frames'].append({         
                    'image': Image.fromarray(np.round(imagej*255.).astype(np.uint8)), 
                    'depth': depth_pred,
                    'transform_matrix': Pc2w.tolist(),
                })
        return traindata