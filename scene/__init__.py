import random

from arguments import GSParams
from scene.dataset_readers import readDataInfo
from scene.gaussian_model import GaussianModel
from utils.pose_noise_util import apply_noise_bloomscene


class Scene:
    gaussians: GaussianModel

    def __init__(self, traindata, gaussians: GaussianModel, opt: GSParams):
        self.traindata = traindata
        self.gaussians = gaussians
        
        info = readDataInfo(traindata, opt.white_background, opt.eval)
        random.shuffle(info.train_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = info.nerf_normalization["radius"]

        print("Loading Training Cameras ...")
        self.train_cameras = info.train_cameras        
        print("Loading Preset Cameras ...")
        self.preset_cameras = {}
        print("Loading Eval Cameras ...")
        self.eval_cameras = apply_noise_bloomscene(self.train_cameras)

        for campath in info.preset_cameras.keys():
            self.preset_cameras[campath] = info.preset_cameras[campath]

        self.gaussians.create_from_pcd(info.point_cloud, self.cameras_extent)
        self.gaussians.training_setup(opt)

    def getTrainCameras(self):
        return self.train_cameras
    
    def getPresetCameras(self, preset):
        assert preset in self.preset_cameras
        return self.preset_cameras[preset]
    
    def getEvalCameras(self):
        return self.eval_cameras