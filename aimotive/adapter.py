# Adapter 
"""The code to translate aimotive dataset to KITTI dataset format"""


# from pyquaternion import Quaternion # pip install pyquaternion
from scipy.spatial.transform import Rotation as R

import os
# from shutil import copyfile
import json
import numpy as np
import math
from typing import Union
import numpy as np
from time import sleep
from src.aimotive_dataset import AiMotiveDataset
import laspy
from PIL import Image # pip install pillow


"""
Your original file directory is:
aimotive
└── data <----------------------------root_dir
    └── train <-------------------------------------data_dir
        └── nighttime
            └── 20210901-194123-00.37.12-00.37.27@Yoda
                └── dynamic
                    └── box/3d_body <-----------------------------detected_3D_objects/ used
                        └── frame_0033643.json
                        └── frame_0033644.json
                        └── ...
                    └── raw-revolution <-----------------------------lidar_point_cloud / used
                        └── frame_0033643.laz
                        └── frame_0033644.laz
                        └── ...
                └── sensor
                    └── camera <-----------------------------only front camera (F_MIDLONDRANGECAM_CL)
                        └── B_MIDRANGECAM_C
                            └── B_MIDRANGECAM_C_0033643.jpg
                            └── B_MIDRANGECAM_C_0033644.jpg
                            └── ...
                        └── F_MIDLONDRANGECAM_CL <-----------------------------front camera (F_MIDLONDRANGECAM_CL)/ used
                            └── F_MIDLONDRANGECAM_CL_0033643.jpg
                            └── F_MIDLONDRANGECAM_CL_0033644.jpg
                            └── ...
                        └── M_FISHEYE_L
                            └── M_FISHEYE_L_0033643.jpg
                            └── M_FISHEYE_L_0033644.jpg
                            └── ...
                        └── M_FISHEYE_R
                            └── M_FISHEYE_R_0033643.jpg
                            └── M_FISHEYE_R_0033644.jpg
                            └── ...
                    └── calibration
                        └── calibration.json
                        └── extrinsic_matrices.json
                    └── gnssins
                    └── radar
    └── test
        └──
        └──...
"""
"""
Your converted file directory is:
aimotive-kitti_format
└── data <----------------------------root_dir
    └── training <-------------------------------------data_dir
        └── calib
            └── 0033643.txt
            └── 0033644.txt
            └── ...
        └── image_2
            └── 0033643.png
            └── 0033644.png
            └── ...
        └── label_2
            └── 0033643.txt
            └── 0033644.txt
            └── ...
        └── velodyne
            └── 0033643.bin
            └── 0033644.bin
            └── ...
    └── testing
        └──
        └──...
"""
MEAN_SIZE = {}
def get_frame_id(path: str) -> str:
    """
    Parses the frame id form a given path.

    Args:
        path: a path to a frame.

    Returns:
        frame_id: the parsed frame id
    """
    frame_id = os.path.normpath(path).split(os.path.sep)[-1]
    frame_id = os.path.splitext(frame_id)[0]
    frame_id = frame_id.split('_')[1]

    return frame_id

def cart2hom(pts_3d: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to Homogeneous.

    Args:
        pts_3d: nx3 points in Cartesian

    Returns:
        nx4 points in Homogeneous by appending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def project_cam_to_image(pts_3d_rect: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Project camera coordinate to image.

    Args:
        pts_3d_ego: nx3 points in camera coord
        intrinsic: 3x4 intrinsic matrix

    Returns:
        nx3 points in image coord + depth
    """
    uv_cam = cart2hom(pts_3d_rect).T
    uv = intrinsic.dot(uv_cam)
    uv[0:2, :] /= uv[2, :]
    return uv.transpose()

def project_ego_to_cam(pts_3d_ego: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Project egovehicle point onto camera frame.

    Args:
        pts_3d_ego: nx3 points in egovehicle coord.
        extrinsic: 4x4 extrinsic matrix.

    Returns:
        nx3 points in camera coord.
    """

    uv_cam = extrinsic.dot(cart2hom(pts_3d_ego).transpose())

    return uv_cam.transpose()[:, 0:3] # remove the 1 in the last column and size is nx3

def project_ego_to_image(pts_3d_ego: np.ndarray, intrinsic: np.ndarray, extrinsic:np.ndarray) -> np.ndarray:
    """Project egovehicle coordinate to image.

    Args:
        pts_3d_ego: nx3 points in egovehicle coord
        intrinsic: 3x4 intrinsic matrix
        extrinsic: 4x4 extrinsic matrix

    Returns:
        nx3 points in image coord + depth
    """

    uv_cam = project_ego_to_cam(pts_3d_ego, extrinsic)
    return project_cam_to_image(uv_cam, intrinsic)

root_dir = 'aimotive'
data_dir = root_dir + '/data'
target_data_dir = 'data/aimotive-kitti_format'
target_mode = 'training'
mode = 'train'
# calib_path = 'data/'+ mode + intermediate_path + 'sensor/calibration/calibration.json'
# with open(calib_path, 'r') as stream:
#         try:
#             calib_json = json.load(stream)
#         except json.decoder.JSONDecodeError:
#             print(f"Failed to load: {calib_path}")

# Create target directory & all intermediate directories if don't exists
if not os.path.exists(target_data_dir):
    os.makedirs(target_data_dir)
    print("Directory ", target_data_dir, " Created ")
else:
    print("Directory ", target_data_dir, " already exists")
if not os.path.exists(target_data_dir + '/' + target_mode):
    os.makedirs(target_data_dir + '/' + target_mode)
    os.makedirs(target_data_dir + '/' + target_mode + '/calib')
    os.makedirs(target_data_dir + '/' + target_mode + '/image_2')
    os.makedirs(target_data_dir + '/' + target_mode + '/label_2')
    os.makedirs(target_data_dir + '/' + target_mode + '/velodyne')
else:
    print("Directory ", target_data_dir + '/' + target_mode, " already exists")

    

dataset = AiMotiveDataset('aimotive/data', mode)
data_num = len(dataset)

# Px is the projection matrix from the rectified camera coordinate system to the image plane of camera x.

L1='P0: 0 0 0 0 0 0 0 0 0 0 0 0'
L2='P1: 0 0 0 0 0 0 0 0 0 0 0 0'
L4='P3: 0 0 0 0 0 0 0 0 0 0 0 0'
L7='Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0'

for idx in range(data_num):
    intermediate_path = '/' + dataset.dataset_index[idx].split('/')[3] + '/' + dataset.dataset_index[idx].split('/')[4] + '/'
    data_item = dataset.data_loader[dataset.dataset_index[idx]]
    frame_id = get_frame_id(data_item.annotations.path)
    print('frame_id', frame_id)
    target_id = str(idx).zfill(6)

    ############# convert calibration.json to KITTI format #############
    # calculate P2
    intrinsic = data_item.camera_data.front_camera.camera_params.intrinsic # 3x4
    extrinsic = data_item.camera_data.front_camera.camera_params.extrinsic # 4x4 # in calibration.json RT_sensor_from_body (camera)
    L3='P2: ' + ' '.join(map(str, intrinsic.reshape(12).tolist())) #K=intrinsic_matrix

    # calculate R0_rect
    # if front, R0_rect = I
    L5='R0_rect: 1 0 0 0 1 0 0 0 1'

    # calculate Tr_velo_to_cam 
    ext_rot= R.from_matrix(extrinsic[0:3,0:3].T) # rotation matrix (3x3)

    # point cloud coordinate to reference coordinate (P_cam = tr_velo_to_cam * P_lidar)
    # yet, using the extrinsic matrix and pc data without any change worked, which were verified by the visualization
    tr_velo_to_cam = np.hstack((extrinsic[0:3,0:3],extrinsic[0:3,3].reshape(3,1))) # 3x4
    L6= 'Tr_velo_to_cam: ' 
    for k in tr_velo_to_cam.reshape(1,12)[0][0:12]:
        L6= L6+ str(k)+ ' '
    L6=L6[:-1]

    # write to file
    target_calib_file = target_data_dir + '/' + target_mode + '/calib/' + target_id + '.txt'
    with open(target_calib_file, 'w') as f:
        f.write(L1+'\n')
        f.write(L2+'\n')
        f.write(L3+'\n')
        f.write(L4+'\n')
        f.write(L5+'\n')
        f.write(L6+'\n')
        f.write(L7+'\n')

    ############# convert camera image to KITTI format #############

    image_file = 'aimotive/data/' + mode + intermediate_path + 'sensor/camera/F_MIDLONGRANGECAM_CL/F_MIDLONGRANGECAM_CL_' + frame_id + '.jpg'
    target_image_file = target_data_dir + '/' + target_mode + '/image_2/' + target_id + '.png'
    im = Image.open(image_file)
    im.save(target_image_file)
    im_width, im_height = im.size # im_width 1280, im_height 704

    ############# convert label.json to KITTI format #############
    label_file =  'aimotive/data/'+ mode + intermediate_path + 'dynamic/box/3d_body/frame_' + frame_id + '.json'
    target_label_file = target_data_dir + '/' + target_mode + '/label_2/' + target_id + '.txt'

    # read label.json
    with open(label_file, 'r') as stream:
        try:
            label_json = json.load(stream)
        except json.decoder.JSONDecodeError:
            print(f"Failed to load: {label_file}")

    objects = label_json['CapturedObjects']
    file=open(target_label_file,'w+')

    # write label.txt
    for obj in objects:
        # get the quaternion
        quat= Quar = R.from_quat([obj['BoundingBox3D Orientation Quat X'],obj['BoundingBox3D Orientation Quat Y'],obj['BoundingBox3D Orientation Quat Z'],obj['BoundingBox3D Orientation Quat W']])
        classes= obj['ObjectType']
        truncated= obj['Truncated']
        occulusion= obj['Occluded']
        height= obj['BoundingBox3D Extent Z']
        width= obj['BoundingBox3D Extent Y']
        length= obj['BoundingBox3D Extent X']

        # in ego frame (body coordinate system, whose origin is the projected ground plane point under the center of the vehicle’s rear axis)
        center= np.array([obj['BoundingBox3D Origin X'],obj['BoundingBox3D Origin Y'],obj['BoundingBox3D Origin Z']])
        if center[0] < 0:
            continue
        # all eight points in ego frame 
        corners_ego_frame = np.array([[center[0]-length/2,center[1]-width/2,center[2]-height/2],
                                    [center[0]+length/2,center[1]-width/2,center[2]-height/2],
                                    [center[0]+length/2,center[1]+width/2,center[2]-height/2],
                                    [center[0]-length/2,center[1]+width/2,center[2]-height/2],
                                    [center[0]-length/2,center[1]-width/2,center[2]+height/2],
                                    [center[0]+length/2,center[1]-width/2,center[2]+height/2],
                                    [center[0]+length/2,center[1]+width/2,center[2]+height/2],
                                    [center[0]-length/2,center[1]+width/2,center[2]+height/2]])

        
        corners_cam_frame= project_ego_to_cam(corners_ego_frame,extrinsic) # all eight points in the camera frame # 8x3
        image_corners= project_ego_to_image(corners_ego_frame, intrinsic, extrinsic)

        image_bbox= [min(image_corners[:,0]), min(image_corners[:,1]),max(image_corners[:,0]),max(image_corners[:,1])]
        # the four coordinates we need for KITTI
        image_bbox=[round(x) for x in image_bbox]    

        # if the object is out of the image, skip it
        if image_bbox[0] < 0 or image_bbox[1] < 0 or image_bbox[2] > im_width or image_bbox[3] > im_height:
            print('out of image')
            continue

        if image_bbox[0] >= image_bbox[2] or image_bbox[1] >= image_bbox[3]:
            print('invalid bbox')
            continue

        # the center coordinates in cam frame
        center_cam_frame= project_ego_to_cam(np.array([center]),extrinsic) # 1x3

        
        # rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        # rotation_y= -np.arctan2(corners_cam_frame[0][2]-corners_cam_frame[1][2],corners_cam_frame[0][0]-corners_cam_frame[1][0])
        # print('rotation_y', rotation_y)


        angle= ext_rot * quat # quaternion to rotation matrix
        angle=angle.as_euler('zyx')[1]
        # print('angle', angle) # angle ~= rotation_y



        angle = (angle + np.pi) % (2 * np.pi) - np.pi 
        beta= math.atan2(center_cam_frame[0][2],center_cam_frame[0][0])
        alpha= (angle-beta + np.pi) % (2 * np.pi) - np.pi 
        line=classes+ ' {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n'.format(round(truncated,2),occulusion,round(alpha,2),round(image_bbox[0],2),round(image_bbox[1],2),round(image_bbox[2],2),round(image_bbox[3],2),round(height,2), round(width,2),round(length,2), round(center_cam_frame[0][0],2),round(center_cam_frame[0][1],2)+0.5*round(height,2),round(center_cam_frame[0][2],2),round(angle,2))                

        file.write(line)

        if classes not in MEAN_SIZE.keys():
            MEAN_SIZE[classes] = {'sum': np.array([length, width, height]), 'count': 1}
        else:
            MEAN_SIZE[classes]['sum'] += np.array([length, width, height])
            MEAN_SIZE[classes]['count'] += 1

    file.close()

    ############# convert lidar.laz to KITTI format #############

    lidar_file = 'aimotive/data/' + mode + intermediate_path + 'dynamic/raw-revolutions/frame_' + frame_id + '.laz'
    target_lidar_file = target_data_dir + '/' + target_mode + '/velodyne/' + target_id + '.bin'
    # get extrinsic matrix
    # lidar_extrinsic = np.array(calib_json['M_LIDAR_M']['RT_sensor_from_body'])
    
    las = laspy.read(lidar_file)
    # shape of las is (n_points, 4) in ego frame
    points_in_ego = np.vstack((las.x, las.y, las.z)).T
    # transform to lidar coordinate system
    # points_in_cam = project_ego_to_cam(points_in_ego, lidar_extrinsic) # did not work
    las_points = np.hstack((points_in_ego, las.intensity.reshape(-1,1)))
    # print('las_points.shape', las_points.shape) # (n_points, 4)
    las_points = las_points.astype(np.float32)
    las_points.tofile(target_lidar_file)

for key in MEAN_SIZE.keys():
    MEAN_SIZE[key] = MEAN_SIZE[key]['sum'] / MEAN_SIZE[key]['count']
print(MEAN_SIZE)