# split kitti dataset into train and test set and save to txt file
# select index randomly
# example: 000000, 000001, 000002
import numpy as np
import os
def kitti_split(rate:float = 0.8):
    # read txt file number
    DIR = 'data/aimotive-kitti_format/training/label_2'
    data_num = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

    # split train and test set
    train_num = int(data_num * rate)
    test_num = data_num - train_num
    train_index = np.random.choice(data_num, train_num, replace=False)
    test_index = np.setdiff1d(np.arange(data_num), train_index)

    train_index = [str(i).zfill(6) for i in list(np.sort(train_index))]
    test_index = [str(i).zfill(6) for i in list(np.sort(test_index))]

    # save 
    save_path = 'kitti/image_sets/train.txt'
    with open(save_path, 'w') as f:
        for index in train_index:
            f.write(index + '\n')
    save_path = 'kitti/image_sets/val.txt'
    with open(save_path, 'w') as f:
        for index in test_index:
            f.write(index + '\n')
    

    # make file for rgb detection (only val)
            
    save_path = 'kitti/rgb_detections/rgb_detection_val.txt'
    # det_str2id = {'CAR': 1, 'VAN': 2, 'BUS': 3, 'RIDER': 4, 'MOTORCYCLE': 5, 'PEDESTRIAN': 6, 'BICYCLE': 7, 'TRAIN': 8}
    det_str2id = {'CAR': 1, 'TRUCK': 2, 'BUS': 3, 'RIDER': 4, 'MOTORCYCLE': 5, 'PEDESTRIAN': 6, 'BICYCLE': 7, 'TRAIN': 8}

    PHOTO_DIR = 'data/aimotive-kitti_format/training/image_2/'
    with open(save_path, 'w') as sf:
        for index in test_index:
            file_path = os.path.join(DIR, index + '.txt')
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split()
                    save_info = ''
                    save_info += PHOTO_DIR + index + '.png' + ' '
                    save_info += str(det_str2id[line[0]]) + ' '
                    save_info += str(1.0) + ' ' # confidence
                    save_info += line[4] + ' ' + line[5] + ' ' + line[6] + ' ' + line[7] + '\n'
                    sf.write(save_info)

    
if __name__ == '__main__':
    kitti_split()   