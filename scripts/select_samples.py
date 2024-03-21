
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

import argparse
import json
import tqdm

def read_intrinsic(camera_path):

    with open(camera_path, 'r') as f:
        lines = f.readlines()

    content = lines[3]
    elements = content.strip().split()
    print(*elements)

    type_cam = elements[1]
    print(f"CAM_TYPE: {type_cam}")

    if type_cam == "PINHOLE":
        fx, fy, cx, cy = map(float, elements[-4:] )
    elif type_cam == "SIMPLE_PINHOLE":
        fx, cx, cy = map(float, elements[-3:] )
        fy = fx
    else:
        raise NotImplementedError
    
    return fx, fy, cx, cy
def read_poses(images_path):
    """
    images_path: path where images.txt is located.
    """
    if not os.path.exists(images_path):
        raise Exception(f"No such file : {images_path}")

    with open(images_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise Exception(f"Invalid cameras.txt file : {images_path}")

    comments = lines[:4]
    contents = lines[4:]

    poses = []
    idx = [0 for _ in contents[::2]]
    for i, content in enumerate(contents[::2]):
        content_items = content.split(' ')
        q_xyzw = np.array(content_items[2:5] + content_items[1:2], dtype=np.float32) # colmap uses wxyz
        t_xyz = np.array(content_items[5:8], dtype=np.float32)
        img_name = content_items[9]

        R = Rot.from_quat(q_xyzw).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = t_xyz

        img_num = int((img_name.strip().split("."))[0]) # "00056.jpg\n" -> 56
        poses.append(T)
        idx[img_num] = i

    poses = np.stack(poses)
    poses = poses[idx]
    return poses

def to_c2w(w2c):
    """
    w2c[4x4]->c2w[4x4]
    numerically stable than np.linalg.inv
    """
    rot = w2c[:3,:3].T
    trans = -w2c[:3,3] @ w2c[:3,:3]

    c2w = np.eye(4)
    c2w[:3,:3] = rot
    c2w[:3,3] = trans

    return c2w


def write_pose(poses, out_path):
    """
    poses: [n,3,4] or [n,4,4] cam2world matrix
    stride: # of camera group per saving (just pass large number like 1000000 to make it one file)
    """
    m_cam = None

    for j,pose in enumerate(poses):
        m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        m.transform(pose)
        if m_cam is None:
            m_cam = m
        else:
            m_cam += m

    o3d.io.write_triangle_mesh(out_path, m_cam)


def build_K(fx, fy, cx, cy):
    K = np.array([[fx, 0., cx,],
                  [0., fy, cy,],
                  [0., 0., 1.,]]).astype(np.float64)
    return K

def get_normalized_coords(pixel_coords, K):
    
    pixel_coords_homog = np.pad(pixel_coords, ((0,0),(0,1)), 'constant', constant_values=1.0)    
    
    K_inv = np.linalg.inv(K)
    norm_coords = K_inv @ pixel_coords_homog.T

    return (norm_coords.T)[...,:2] 


def normalized_coords_to_cam_coords(normalized_coords, depth):
    normalized_coords_homog = np.pad(normalized_coords, ((0,0),(0,1)), 'constant', constant_values=1.0)  # (hw, 3)
    return normalized_coords_homog * depth.reshape(-1,1)

def cam_to_world_coords(cam_coords, c2w):
    """
    cam_coords: (hw, 3)
    c2w: (4,4)
    """
    cam_coords_homog = np.pad(cam_coords, ((0,0),(0,1)), 'constant', constant_values=1.0)  # (hw,4)
    world_coords = c2w @ cam_coords_homog.T

    return (world_coords.T)[...,:3]
def pcd_spread(c2ws, K, near=0.0, far=8.0, num_pcd=100_000):
    """
    generate pcd along to camera frustrum
    c2ws: cam pose in cam2world format
    K: intrinsic matrix(3,3)
    """
    all_xyz = []
    d = 50 # num of points per ray
    num_pcd_per_cam = num_pcd // (max(len(c2ws)-5,1))
    h = int(K[1,2]*2)
    w = int(K[0,2]*2)

    for c2w in tqdm.tqdm(c2ws, desc="spreading pcds..."):

        stride_coeff = num_pcd_per_cam**(-1/3)
        stride_h = int(h*stride_coeff)
        stride_w = int(w*stride_coeff)
        # stride_d = int(d*stride_coeff)

        pixel_coords = np.stack( np.meshgrid(np.linspace(0,w-1,w), np.linspace(0,h-1,h), indexing="xy"),axis=-1)
        pixel_coords = pixel_coords[::stride_h, ::stride_w]
        pixel_coords = pixel_coords.reshape((-1,2))
        
        norm_coords = get_normalized_coords(pixel_coords, K)
        
        norm_coords = np.tile(norm_coords, (d*2,1))
        depth = np.random.random((norm_coords.shape[0]))*(far-near)+near
        # import pdb; pdb.set_trace()
        cam_coords = normalized_coords_to_cam_coords(norm_coords, depth)[:num_pcd_per_cam]
        
        xyz_world = cam_to_world_coords(cam_coords, c2w)
        all_xyz.append(xyz_world)

    return np.concatenate(all_xyz,axis=0)[:num_pcd]


def get_majorly_visible_pcds(c2ws,K, pcds, threshold, gamma=0.999):

    N = pcds.shape[0]
    pcds_homog = np.pad(pcds, ((0,0),(0,1)), 'constant', constant_values=1.0) # (N,4)
    h = int(K[1,2]*2)
    w = int(K[0,2]*2)
    
    score = np.zeros((N,)).astype(int)

    for c2w in tqdm.tqdm(c2ws, desc="sorting pcds..."):
        
        w2c = np.linalg.inv(c2w)
        cam_xyz = (w2c @ pcds_homog.T)[:3] # (3,N)
        norm_xy = cam_xyz[:2] / cam_xyz[2:3]
        norm_xy_homog = np.pad(norm_xy, ((0,1),(0,0)) , 'constant', constant_values=1.0) # (3,N)
        pixel_xy_homog = K@norm_xy_homog # (3,N)
        pixel_xy = pixel_xy_homog[:2] / pixel_xy_homog[2:3] # (2,N)
        pixel_xy = pixel_xy.T # (N,2)

        cam_xyz = cam_xyz.T

        score = score + ( (pixel_xy[:,0]>=0) * (pixel_xy[:,0]<=w-1) * (pixel_xy[:,1] >= 0) * (pixel_xy[:,1] <= h-1) * (cam_xyz[:,2] >= 0) ).astype(int)
    score = score.astype(float) / c2ws.shape[0]
    while (score>threshold).sum() < 100:
        threshold *= gamma
    print(f"Major points: visible by {threshold*100:.2f}%")

    return pcds[score>threshold]


def write_pcd(vis_path, xyz, rgb=None):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    path = os.path.join(vis_path, "pcd.ply")
    o3d.io.write_point_cloud(path, pcd)


        
if __name__ == "__main__":
    
    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--dset", type=str, choices=['nerfllff','dtu','mipnerf360', 'nerfsynthetic'])
    parser.add_argument("--visible_threshold", type=float, default=0.99 )
    parser.add_argument("--hull_threshold", type=float, default=5e-1 )
    
    args = parser.parse_args()

    ## Read poses
    w2cs = read_poses(os.path.join(args.path, "sparse_txt", "images.txt"))
    c2ws = np.stack([to_c2w(w2c) for w2c in w2cs]) 
    N = c2ws.shape[0]
    # (Assuming z is front and all poses are looking at front,) keep only xy's of cameras.
    
    ## Read intrinsic
    fx, fy, cx, cy = read_intrinsic(os.path.join(args.path, "sparse_txt", "cameras.txt"))
    K = build_K(fx, fy, cx, cy)

    ## spread points along the camera frustrums
    pcd = pcd_spread(c2ws,K, near=2.0, far=12.0, num_pcd=500_000)
    pcd = get_majorly_visible_pcds(c2ws, K, pcd, threshold=args.visible_threshold)
    
    ## visualize folder
    visualize_path = os.path.join(args.path, "split_visualization")
    os.makedirs(visualize_path, exist_ok=True)
    
    ## train/test split
    if args.dset=='nerfllff' or args.dset=='dtu': ## forward-facing
        cam_xys = c2ws[:,:2,3]
        
        # Convex Hull algorithm
        def ccw(x1, y1, x2, y2, x3, y3):
            """
            True if vector (x1,y1)->(x3,x3) is at "counter-clockwise(ccw) w.r.t vector (x1,y1)->(x2,y2)
            """
            return (y2-y1)*(x3-x1) - (x2-x1) * (y3-y1) < 0
        
        # Leftmost point is always part of convex hull.
        first_i = np.argmin(cam_xys[:,0])
        
        pivot_i = first_i
        candidate_i = None
        hull_list = []

        while candidate_i != first_i:
            print(pivot_i)    
            hull_list.append(pivot_i)
            candidate_i = (pivot_i+1) % N
            
            for j in range(N):
                if j == pivot_i: continue

                if ccw(cam_xys[pivot_i,0], cam_xys[pivot_i,1],
                    cam_xys[candidate_i,0], cam_xys[candidate_i,1],
                    cam_xys[j,0], cam_xys[j,1]):
                    candidate_i = j
            
            pivot_i = candidate_i
        
        # Split Train/Test

        train_list = hull_list
        test_list = [i for i in range(N) if i not in train_list]

        # For test set, if too close to the hull, exceptionally add to train set.
        
        additional_set = []
        for idx,(x,y) in enumerate(cam_xys[test_list]):
            for i, (x1,y1) in enumerate(cam_xys[train_list]):
                x2, y2 = cam_xys[train_list[(i+1)%len(train_list)]]

                # line equation
                a = y2-y1
                b = -(x2-x1)
                c = y1*(x2-x1) - x1*(y2-y1)
                try:          
                    # distance
                    dist = abs(a*x + b*y + c) / (a**2 + b**2)**0.5
                except:
                    import pdb; pdb.set_trace()
                if dist < args.hull_threshold:
                    additional_set.append(test_list[idx])
                    break
        
        train_list.extend(additional_set)
        test_list = [i for i in range(N) if i not in train_list]
        
    elif args.dset=='mipnerf360': ## 360
        train_list = [i for i in range(len(c2ws)) if np.random.rand()<0.5]
        test_list = [i for i in range(len(c2ws)) if i not in train_list]
        write_pcd(visualize_path, pcd)
        np.save(os.path.join(args.path,"center.npy"), pcd.mean(axis=0))
        
    elif args.dset=='nerfsynthetic':
        train_list = np.arange(100).tolist()
        test_list = np.arange(100,300).tolist()
        write_pcd(visualize_path, pcd)
        np.save(os.path.join(args.path,"center.npy"), pcd.mean(axis=0))
        
    else:
        raise NotImplementedError

    # Visualize
    c2w_train = c2ws[train_list]
    c2w_test = c2ws[test_list]
    write_pose(c2w_train, os.path.join(visualize_path, "train.ply"))
    write_pose(c2w_test, os.path.join(visualize_path, "test.ply"))

    # Save as split idx
    train_list.sort()
    test_list.sort()
    print(train_list)
    dic = dict(train=[int(e) for e in train_list], test=[int(e) for e in test_list])

    with open(os.path.join(args.path, "split_index.json"),"w") as fp:
        json.dump(dic, fp, indent=4)