# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
#
# Copyright 2023 LucidDreamer Authors
#
# Computer Vision Lab, SNU, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from the Computer Vision Lab, SNU or
# its affiliates is strictly prohibited.
#
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
import numpy as np


def generate_seed_360(viewangle, n_views):   
    N = n_views
    render_poses = np.zeros((N, 3, 4))
    for i in range(N):
        th = (viewangle/N)*i/180*np.pi
        render_poses[i,:3,:3] = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        render_poses[i,:3,3:4] = np.random.randn(3,1)*0.0 # Transition vector

    return render_poses

def my_generate_seed_360(viewangle, n_views):   
    N = n_views
    render_poses = np.zeros((N, 3, 4))
    th_list = [0, 1, 9, 2, 8, 3, 7, 4, 6, 5]
    for i in range(N):
        th = (viewangle/N)*th_list[i]/180*np.pi
        render_poses[i,:3,:3] = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        render_poses[i,:3,3:4] = np.random.randn(3,1)*0.0 # Transition vector

    return render_poses


def generate_seed_360_half(viewangle, n_views):
    N = n_views // 2
    halfangle = viewangle / 2
    render_poses = np.zeros((N*2, 3, 4))
    for i in range(N): 
        th = (halfangle/N)*i/180*np.pi
        render_poses[i,:3,:3] = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        render_poses[i,:3,3:4] = np.random.randn(3,1)*0.0 # Transition vector
    for i in range(N):
        th = -(halfangle/N)*i/180*np.pi
        render_poses[i+N,:3,:3] = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        render_poses[i+N,:3,3:4] = np.random.randn(3,1)*0.0 # Transition vector
    return render_poses


def generate_seed_hemisphere(center_depth, degree=5):
    degree = 5                                                   
    thlist = np.array([degree, 0, 0, 0, -degree])                
    philist = np.array([0, -degree, 0, degree, 0])
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        d = center_depth 
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))   
        render_poses[i,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), d-d*np.cos(phi/180*np.pi)]).reshape(3,1)       

    return render_poses


def my_generate_seed_hemisphere(center_depth, degree=5):
    degree = 5                                                   
    thlist = np.array([degree, 0, 0, 0, -degree])                
    philist = np.array([0, -degree, 0, degree, 0])
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist) * len(center_depth), 3, 4))
    for j in range(len(center_depth)):
        per_center_depth = center_depth[j]
        for i in range(len(thlist)):
            th = thlist[i]
            phi = philist[i]
            d = per_center_depth 
            
            idx = j * len(thlist) + i
            render_poses[idx,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))   
            render_poses[idx,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), d-d*np.cos(phi/180*np.pi)]).reshape(3,1)       

    return render_poses


def get_pcdGenPoses(pcdgenpath, argdict={}):
    if pcdgenpath == 'rotate360':
        render_poses = my_generate_seed_360(360, 10)        
    elif pcdgenpath == 'hemisphere':
        render_poses = my_generate_seed_hemisphere(argdict['center_depth'])         
    else:
        raise("Invalid pcdgenpath")
    return render_poses


def get_camerapaths():
    preset_json = {}
    for cam_path in ['rotate360']:
        if cam_path == 'rotate360':
            render_poses = generate_seed_360(360, 180)
        else:
            raise("Unknown pass")
            
        yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
        blender_train_json = {"frames": []}
        for render_pose in render_poses:
            curr_frame = {}
            Rw2i = render_pose[:3,:3]
            Tw2i = render_pose[:3,3:4]
            Ri2w = np.matmul(yz_reverse, Rw2i).T
            Ti2w = -np.matmul(Ri2w, np.matmul(yz_reverse, Tw2i))
            Pc2w = np.concatenate((Ri2w, Ti2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([0,0,0,1]).reshape((1,4))), axis=0)

            curr_frame["transform_matrix"] = Pc2w.tolist()
            blender_train_json["frames"].append(curr_frame)

        preset_json[cam_path] = blender_train_json

    return preset_json