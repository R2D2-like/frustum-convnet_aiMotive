import numpy as np

class KITTICategory(object):

    CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    CLASS_MEAN_SIZE = {
        'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
        'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
        'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
    }

    NUM_SIZE_CLUSTER = len(CLASSES)

    MEAN_SIZE_ARRAY = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clusters
    for i in range(NUM_SIZE_CLUSTER):
        MEAN_SIZE_ARRAY[i, :] = CLASS_MEAN_SIZE[CLASSES[i]]


class AIMOTIVECategory(object):

    # CLASSES = ['CAR', 'TRUCK', 'PEDESTRIAN', 'BUS', 'VAN', 'MOTORCYCLE', 'PICKUP', 'RIDER', 'BICYCLE', 'TRAILER', 'OTHER-OBJECT', 'OTHER-RIDEABLE', 'TRAIN', 'SHOPPING-CART']
    CLASSES = ['CAR', 'TRUCK', 'BUS', 'RIDER', 'MOTORCYCLE', 'PEDESTRIAN', 'BICYCLE','TRAIN'] 

    CLASS_MEAN_SIZE = {
        'CAR': np.array([4.12934884, 1.94110326, 1.47315377]),
        'TRUCK': np.array([6.88932462, 2.69689112, 2.69581303]),
        'BUS': np.array([7.86844541, 2.43340183, 2.35588492]),
        'RIDER': np.array([2.40401426, 1.04555928, 1.67159249]),
        'MOTORCYCLE': np.array([1.90619526, 0.93973676, 1.43790165]),
        'PEDESTRIAN': np.array([0.89820091, 0.90657051, 1.66919671]),
        'BICYCLE': np.array([1.96547235, 1.02996674, 1.3922288 ]),
        'TRAIN': np.array([4.05319149, 0.5341259 , 1.62111028])
    }

    NUM_SIZE_CLUSTER = len(CLASSES)

    MEAN_SIZE_ARRAY = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clusters
    for i in range(NUM_SIZE_CLUSTER):
        MEAN_SIZE_ARRAY[i, :] = CLASS_MEAN_SIZE[CLASSES[i]]


class SUNRGBDCategory(object):

    CLASSES = ['bathtub', 'bed', 'bookshelf', 'chair', 'desk', 'dresser', 'night_stand', 'sofa', 'table', 'toilet']
    CLASS_MEAN_SIZE = {
        'bathtub': np.array([0.765840, 1.398258, 0.472728]),
        'bed': np.array([2.114256, 1.620300, 0.927272]),
        'bookshelf': np.array([0.404671, 1.071108, 1.688889]),
        'chair': np.array([0.591958, 0.552978, 0.827272]),
        'desk': np.array([0.695190, 1.346299, 0.736364]),
        'dresser': np.array([0.528526, 1.002642, 1.172878]),
        'night_stand': np.array([0.500618, 0.632163, 0.683424]),
        'sofa': np.array([0.923508, 1.867419, 0.845495]),
        'table': np.array([0.791118, 1.279516, 0.718182]),
        'toilet': np.array([0.699104, 0.454178, 0.756250])
    }

    NUM_SIZE_CLUSTER = len(CLASSES)

    MEAN_SIZE_ARRAY = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clusters
    for i in range(NUM_SIZE_CLUSTER):
        MEAN_SIZE_ARRAY[i, :] = CLASS_MEAN_SIZE[CLASSES[i]]


DATASET_INFO = {
    "AIMOTIVE": AIMOTIVECategory,
    "SUNRGBD" : SUNRGBDCategory,
    "KITTI" : KITTICategory
}


