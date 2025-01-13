import numpy as np


class GSParams: 
    def __init__(self):
        self.sh_degree = 3   
        self.feat_dim = 32           
        self.n_offsets = 10          
        self.voxel_size =  0.001     
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = True

        self.iterations = 2990  # 3_000 2990
        self.position_lr_init = 0.0016         # BloomScene: 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 2990  # 3_000

        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 2990
        
        self.mask_lr_init = 0.01
        self.mask_lr_final = 0.0001
        self.mask_lr_delay_mult = 0.01
        self.mask_lr_max_steps = 2990

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05   
        self.scaling_lr = 0.005   
        self.rotation_lr = 0.001   

        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 2990

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 2990
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 2990
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 2990

        self.encoding_xyz_lr_init = 0.005
        self.encoding_xyz_lr_final = 0.00001
        self.encoding_xyz_lr_delay_mult = 0.33
        self.encoding_xyz_lr_max_steps = 2990

        self.mlp_grid_lr_init = 0.005
        self.mlp_grid_lr_final = 0.00001
        self.mlp_grid_lr_delay_mult = 0.01
        self.mlp_grid_lr_max_steps = 2990

        self.mlp_deform_lr_init = 0.005
        self.mlp_deform_lr_final = 0.0005
        self.mlp_deform_lr_delay_mult = 0.01
        self.mlp_deform_lr_max_steps = 2990

        # for anchor densification
        self.start_stat = 200
        self.update_from = 500
        self.update_interval = 100
        self.update_until = 2000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2   
        self.densification_interval = 100
        self.opacity_reset_interval = 2990
        self.densify_from_iter = 500
        self.densify_until_iter = 2990
        self.densify_grad_threshold = 0.0002

        self.min_opacity = 0.005  # 0.2
        self.success_threshold = 0.8

        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


  
class CameraParams:
    def __init__(self, H: int = 512, W: int = 512):
        self.H = H
        self.W = W
        self.focal = (5.8269e+02, 5.8269e+02)   
        self.fov = (2*np.arctan(self.W / (2*self.focal[0])), 2*np.arctan(self.H / (2*self.focal[1])))     
        self.K = np.array([
            [self.focal[0], 0., self.W/2],   
            [0., self.focal[1], self.H/2],   
            [0.,            0.,       1.],
        ]).astype(np.float32)