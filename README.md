#  awesome-deep-point-cloud-compression [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)


## Keywords

__`lid.`__: LiDAR &emsp; 
__`vcon.`__: Voxel CNN &emsp; 
__`oct.`__: Octree &emsp; 
__`pnet.`__: PointNet &emsp; 
__`dyn.`__: Dynamic 3D point cloud &emsp; 

## Papers

### 2016

- [[MM](https://ieeexplore.ieee.org/document/7405340)] [__`dyn.`__] Graph-based compression of dynamic 3D point cloud sequences

### 2018

- [[MM](https://dl.acm.org/doi/10.1145/3240508.3240696)] Hybrid Point Cloud Attribute Compression Using Slice-based Layered Structure and Block-based Intra Prediction.

### 2019

- [[ICIP](https://ieeexplore.ieee.org/document/8803413)] [__`vcon.`__] Learning Convolutional Transforms for Lossy Point Cloud Geometry Compression. [[Tensorflow](https://github.com/mauriceqch/pcc_geo_cnn)]

- [[ICRA](https://ieeexplore.ieee.org/document/8794264)] Point Cloud Compression for 3D LiDAR Sensor using Recurrent Neural Network with Residual Blocks. [[PyTorch](https://github.com/ChenxiTU/Point-cloud-compression-by-RNN)]

- [[PCS](https://ieeexplore.ieee.org/document/8954537)] Point cloud coding: Adopting a deep learning-based approach. 

- [[arXiv](https://arxiv.org/abs/1909.12037)] [__`vcon.`__] Learned point cloud geometry compression.

- [[arXiv](https://arxiv.org/abs/1905.03691)] [__`pnet.`__] Deep autoencoder-based lossy geometry compression for point clouds.  [[Tensorflow](https://github.com/YanWei123/Deep-AutoEncoder-based-Lossy-Geometry-Compression-for-Point-Clouds)]

- [[CMM](https://dl.acm.org/doi/10.1145/3343031.3351061)] [__`pnet.`__] 3d point cloud geometry compression on deep learning.

- [[TIP](https://ieeexplore.ieee.org/document/8676054)] A Volumetric Approach to Point Cloud Compression—Part I: Attribute Compression.

- [[TIP](https://ieeexplore.ieee.org/document/8931233)] A Volumetric Approach to Point Cloud Compression–Part II: Geometry Compression.

### 2020

- [[ICME](https://ieeexplore.ieee.org/document/9102866)] [__`oct.`__] Lossy Geometry Compression Of 3d Point Cloud Data Via An Adaptive Octree-Guided Network. [[Tensorflow](https://github.com/wxz1996/pc_compress)]

- [[MMSP](https://ieeexplore.ieee.org/document/9287077)] [__`vcon.`__] Improved Deep Point Cloud Geometry Compression. [[Tensorflow](https://github.com/mauriceqch/pcc_geo_cnn_v2)]

- [[CVPR](https://ieeexplore.ieee.org/document/9157381)] [__`oct.`__] OctSqueeze: Octree-Structured Entropy Model for LiDAR Compression.

- [[NIPS](https://arxiv.org/abs/2011.07590)] [__`oct.`__] MuSCLE: Multi Sweep Compression of LiDAR using Deep Entropy Models.

- [[ICIP](https://ieeexplore.ieee.org/document/9191180)] Folding-Based Compression Of Point Cloud Attributes. [[Tensorflow](https://github.com/mauriceqch/pcc_attr_folding)]

### 2021

- [[TCSVT](https://ieeexplore.ieee.org/document/9287077)] Lossy Point Cloud Geometry Compression via End-to-End Learning.

- [[DCC](https://ieeexplore.ieee.org/document/9418789)] [__`vcon.`__] Multiscale Point Cloud Geometry Compression. [[Pytorch](https://github.com/NJUVISION/PCGCv2)] [[Presentation](https://sigport.org/documents/multiscale-point-cloud-geometry-compression)] 

- [[DCC](https://ieeexplore.ieee.org/document/9418793)] Point AE-DCGAN: A deep learning model for 3D point cloud lossy geometry compression. [[Presentation](https://sigport.org/documents/point-ae-dcgan-deep-learning-model-3d-point-cloud-lossy-geometry-compression)]

- [[CVPR](https://arxiv.org/abs/2105.02158)] [__`oct.`__] [__`lid.`__] VoxelContext-Net: An Octree based Framework for Point Cloud Compression. 

- [[ICASPP](https://ieeexplore.ieee.org/document/9414763)] Learning-Based Lossless Compression of 3D Point Cloud Geometry. [[Tensorflow](https://github.com/Weafre/VoxelDNN)]

- [[RAL-ICRA](https://ieeexplore.ieee.org/document/9354895)] Deep Compression for Dense Point Cloud Maps. [[Pytorch](https://github.com/PRBonn/deep-point-map-compression)]

- [[arXiv](https://arxiv.org/abs/2104.09859)] Multiscale deep context modeling for lossless point cloud geometry compression. [[Pytorch](https://github.com/Weafre/MSVoxelDNN)]

## Non-Deep Learning Methods and Library

- [[Draco](https://github.com/google/draco)] Draco is a library for compressing and decompressing 3D geometric meshes and point clouds. It is intended to improve the storage and transmission of 3D graphics.

- [[MPEG V-PCC](https://github.com/MPEGGroup/mpeg-pcc-tmc2)] MPEG Video codec based point cloud compression (V-PCC) test model.

- [[MPEG G-PCC](https://github.com/MPEGGroup/mpeg-pcc-tmc13)] MPEG Geometry based point cloud compression (G-PCC) test model (tmc13).

- [[CAS '18](https://ieeexplore.ieee.org/document/8571288)] Emerging MPEG Standards for Point Cloud Compression.

- [[EG '06](https://dl.acm.org/doi/10.5555/2386388.2386404)] Octree-based point-cloud compression.

- [[ICRA '12](https://ieeexplore.ieee.org/document/6224647)] Real-time compression of point cloud streams.

### 2020

- [[IROS](https://ieeexplore.ieee.org/document/9341071)] [__`lid.`__] Real-Time Spatio-Temporal LiDAR Point Cloud Compression. [[C++ '1](https://github.com/yaoli1992/LiDAR-Point-Cloud-Compression)] [[C++ '2](https://github.com/horizon-research/Real-Time-Spatio-Temporal-LiDAR-Point-Cloud-Compression)]

## Datasets

- [[KITTI](http://www.cvlibs.net/datasets/kitti/)] The KITTI Vision Benchmark Suite.

- [[ShapeNet](https://shapenet.org/)] A collaborative dataset between researchers at Princeton, Stanford and TTIC.

- [[ModelNet](https://modelnet.cs.princeton.edu/)] ModelNet Database.

- [[JPEG Pleno](http://plenodb.jpeg.org/)] JPEG Pleno Database.

- [[MVUB](http://plenodb.jpeg.org/pc/microsoft/)] Microsoft Voxelized Upper Bodies dataset.
