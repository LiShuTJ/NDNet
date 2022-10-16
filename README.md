# NDNet
> This is the  official implementation of "NDNet: Space-wise Multiscale Representation Learning via Neighbor Decoupling for Real-time Driving Scene Parsing" (PyTorch)

## NOTICE: This repo is still under construction
- [x] Implementation of paper models
- [x] Training code
- [x] Speed measurement 
- [ ] Pretrained weights on ImageNet and Cityscapes
    - [x] Pretrained weights on ImageNet
    - [ ] Pretrained weights on Cityscapes
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
python tools/speedMeasure.py --cfg experiments/cityscapes/ndnet_res18.yaml
```

## TensorRT Support
There are two ways to convert a trained model into trt engine.

1. Export to ONNX, and run `trtexec`:
   ```
   1. python tools/exportONNX.py --cfg experiments/cityscapes/ndnet_res18.yaml
   2. trtexec --onnx=NDNet_Res18.onnx --saveEngine=NDNet_Res18.engine
   ```
   > Tips: Running the above may reproduce the speed result in out paper. If you want faster speed (lower precision), you may enable the `--fp16`, `--int8` or `--best` flag in the `trtexec` command.
2. Using [PyTorch-TensorRT](https://www.runoob.com) 
   ```
   # TODO
   ```

## Training
1. Download the pretrained weights, and put it under `pretrained_models` (or modify the `MODEL.PRETRAINED` path in the cfg file).
   
   | Model | Top 1 Acc. | Link |
   | :----: | :----:  | :----: |
   | NDNet-DF1   | 70.86 | [model (code: l27o)](https://pan.baidu.com/s/1vvjtUmz5QcS61onunO8gqw) |
   | NDNet-DF2   | 75.56 | [model (code: nl59)](https://pan.baidu.com/s/1hbDVb2leNrNc7W5Jtl2edQ) |
   | NDNet-Res18 | 72.16 | [model (code: uel1)](https://pan.baidu.com/s/1DbPaxKED_S_0QnwYEec2ZA) |
   | NDNet-Res34 | 76.97 | [model (code: 0pss)](https://pan.baidu.com/s/1h44wjl9-_oJ-9ZzHnUdMnQ ) |


2. For multi-GPU training, taking NDNet-DF1 as an example, type:
   ```
   python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/cityscapes/ndnet_df1.yaml
   ```

## Validation/Testing
1. Train the model, or download the Cityscapes pretrained weights.
   | Model | Test MIoU | Link |
   | :----: | :----:  | :----: |
   | NDNet-DF1   | 75.5 | [model (code: h8nx)](https://pan.baidu.com/s/1ihWD4l9FOXzKzrVn3DFiyg) |
   | NDNet-DF2   | - | TODO |
   | NDNet-Res18 | 76.5 | [model (code: f9je)](https://pan.baidu.com/s/1O-7wWbQ_4O1ZROdrULeR2A) |
   | NDNet-Res34 | - | TODO |

2. Using the `test.py` script (check the DATASET.TEST_SET path and TEST.MODEL_FILE path carefully):
   ```
   python tools/test.py --cfg experiments/cityscapes/ndnet_df1.yaml
   ```

## Acknowledgement
HRNet-Semantic-Segmentation <https://github.com/HRNet/HRNet-Semantic-Segmentation>
