# NDNet
> This is the  official implementation of "NDNet: Space-wise Multiscale Representation Learning via Neighbor Decoupling for Real-time Driving Scene Parsing" (PyTorch)

## NOTICE: This repo is still under construction
- [x] Implementation of paper models
- [x] Training code
- [x] Speed measurement 
- [ ] Pretrained weights on ImageNet and Cityscapes
- [ ] TensorRT implementation

## Install

You can install all these by:
```
pip install -r requirement.txt
```

## Speed Measurement (Pytorch not TRT)
Run the following commond and you'll see the model statistics and the inference speed on your mechine:
```
python tools/speedMeasure.py --cfg experiments\cityscapes\ndnet_res18.yaml
```

## Training
For multi-GPU training, take NDNet-Res18 as an example, type:
```
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/cityscapes/ndnet_res18.yaml
```

## Acknowledgement
HRNet-Semantic-Segmentation <https://github.com/HRNet/HRNet-Semantic-Segmentation>
