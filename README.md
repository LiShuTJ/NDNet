# NDNet
> This is the  official implementation of "NDNet: Space-wise Multiscale Representation Learning via Neighbor Decoupling for Real-time Driving Scene Parsing" (PyTorch)

## NOTICE: This repo is still under construction
- [x] Implementation of paper models
- [x] Training code
- [x] Speed measurement 
- [ ] Pretrained weights on ImageNet and Cityscapes
- [x] TensorRT implementation

## Install

You can install all dependencies (without TensorRT) by:
```
pip install -r requirement.txt
```

For TensorRT installation, please refer to [official installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

## Speed Measurement (Pytorch not TRT)
Run the following commond and you'll see the model statistics and the inference speed on your mechine:
```
python tools/speedMeasure.py --cfg experiments\cityscapes\ndnet_res18.yaml
```

## TensorRT Support
There are two ways to convert a trained model into trt engine.

1. Export to ONNX, and run `trtexec`:
   ```
   1. python tools/exportONNX.py --cfg experiments\cityscapes\ndnet_res18.yaml
   2. trtexec --onnx=NDNet_Res18.onnx --saveEngine=NDNet_Res18.engine
   ```
   > Tips: Running the above may reproduce the speed result in out paper. If you want faster speed (lower precision), you may enable the `--fp16`, `--int8` or `--best` flag in the `trtexec` command.
2. Using [PyTorch-TensorRT](https://www.runoob.com) 
   ```
   # TODO
   ```

## Training
For multi-GPU training, taking NDNet-Res18 as an example, type:
```
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/cityscapes/ndnet_res18.yaml
```

## Acknowledgement
HRNet-Semantic-Segmentation <https://github.com/HRNet/HRNet-Semantic-Segmentation>
