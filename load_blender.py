import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def show_poses(poses, frustum_size=0.1, axis_size=0.2):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pose in poses:
        extrinsic = np.array(pose, dtype=np.float64)
        frustum = o3d.geometry.LineSet()
        points = [
            [0, 0, 0],  # 相机位置
            [frustum_size, frustum_size, frustum_size * 2], [-frustum_size, frustum_size, frustum_size * 2], 
            [-frustum_size, -frustum_size, frustum_size * 2], [frustum_size, -frustum_size, frustum_size * 2],  # 前视面

            [frustum_size * 2, frustum_size * 2, frustum_size * 4], [-frustum_size * 2, frustum_size * 2, frustum_size * 4], 
            [-frustum_size * 2, -frustum_size * 2, frustum_size * 4], [frustum_size * 2, -frustum_size * 2, frustum_size * 4]   # 后视面
        ]
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # 相机位置到前视面
            [1, 2], [2, 3], [3, 4], [4, 1],  # 前视面
            [5, 6], [6, 7], [7, 8], [8, 5],  # 后视面
            [1, 5], [2, 6], [3, 7], [4, 8]   # 前视面到后视面
        ]
        frustum.points = o3d.utility.Vector3dVector(points)
        frustum.lines = o3d.utility.Vector2iVector(lines)
        frustum.paint_uniform_color([1.0, 0.0, 0.0])  # 设置颜色为红色
        frustum.transform(extrinsic)
        vis.add_geometry(frustum)
        # 添加方向箭头
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        axis.transform(extrinsic)
        vis.add_geometry(axis)
    vis.run()
    vis.destroy_window()


def show_imgs(imgs):
    num_imgs = len(imgs)
    cols = 5
    rows = (num_imgs // cols) + (1 if num_imgs % cols != 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()
    
    for i, img in enumerate(imgs):
        axes[i].imshow(img[:, :, :3])
        axes[i].axis('off')
    
    for i in range(num_imgs, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def load_blender_data(basedir, half_res=False, testskip=1, visualize_poses=False, visualize_imgs=False):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    # Add this line to visualize the poses
    if visualize_poses:
        show_poses(poses) 
    # Add this line to visualize the images
    if visualize_imgs:
        show_imgs(imgs)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


