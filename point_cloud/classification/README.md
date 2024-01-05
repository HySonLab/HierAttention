3D Object Classification
============================

## Contributors:
- Trang Nguyen Anh Thuan

- Hugo Sonnery

- Hy Truong Son (Correspondent / PI)

- Siamak Ravanbakhsh

### Requirements
PyTorch=1.7.1, Python=3.7, (I use CUDA=11.1 since I train on RTX3090), Torch Geometric, Wandb

### Dataset

**ModelNet40:**
Please use the data preparation of this repo https://github.com/yanx27/Pointnet_Pointnet2_pytorch and put modelnet40_normal_resampled in dataset/modelnet40_normal_resampled

## Run Code
`sh scripts/train.sh`

`sh scripts/test.sh`

Add `--wandb` in the script to use wandb

## Checkpoint 

Download here: https://drive.google.com/file/d/1dsShZR0v9EuVQmTTo81xI1xvXNvTen8L/view?usp=share_link

## References
```bibtex
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```
```bibtex
@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16259--16268},
  year={2021}
}
```



