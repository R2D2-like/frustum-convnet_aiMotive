# Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection

This repository is the code for Frustum ConvNet with aiMotive dataset.
<div><video controls src="frustum-convnet-viz.mp4"  muted autoplay loop playsinline preload="auto"></video></div>





## Citation

The citation for the original paper (IROS 2019 paper [[arXiv]](https://arxiv.org/abs/1903.01864),[[IEEEXplore]](https://ieeexplore.ieee.org/document/8968513)) is as follows:

```BibTeX
@inproceedings{wang2019frustum,
    title={Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection},
    author={Wang, Zhixin and Jia, Kui},
    booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    pages={1742--1749},
    year={2019},
    organization={IEEE}
}
```

## Installation

You can create the envirionment with docker.

We tested it on
- Ubuntu 22.04
- RTX 3060

### Clone the repository and build docker 

```shell
git clone https://github.com/tier4/frustum-convnet.git
cd frustum-convnet
make build_docker
make run_docker
```



### Compile extension

```shell
cd ops
bash clean.sh
bash make.sh
```

## Download data

Download the aiMotive dataset from [here](https://github.com/aimotive/aimotive_dataset.git) and organize them as follows.

```text
aimotive/data/train
├── highway
├── night
├── rain
├── urban
```

Convert aimotive to kitti format
```
cd frustum-convnet
python3 aimotive/adapter.py
python3 kitti/spliter.py
```

## Training and visualization

### Training

```
python3 kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection --gen_vis_rgb_detection

python3 train/train_net_det.py --cfg cfgs/det_aimotive.yaml OUTPUT_DIR output/aimotive_train
(nohup python3 train/train_net_det.py --cfg cfgs/det_aimotive.yaml OUTPUT_DIR output/aimotive_train > nohup-train.out & #remote)
```

### Visualization

```
python3 visualization/make_dataset.py #for full length visualization

python visualization/visualization.py
```



## Acknowledgements

The official Frustum ConvNet code is available [here](https://github.com/Gorilla-Lab-SCUT/frustum-convnet.git).

Part of the code was adapted from [F-PointNets](https://github.com/charlesq34/frustum-pointnets).

## License

The official Frustum ConvNet is released under [MIT license](LICENSE).
