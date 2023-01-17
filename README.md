# NDNet
> This is the  official implementation of "NDNet: Space-wise Multiscale Representation Learning via Neighbor Decoupling for Real-time Driving Scene Parsing" (PyTorch)
> IEEE Transactions on Neural Networks and Learning Systems

## NOTICE: This repo is still under construction
- [x] Implementation of paper models
- [x] Training code
- [x] Speed measurement 
- [x] Pretrained weights on ImageNet and Cityscapes
    - [x] Pretrained weights on ImageNet
    - [x] Pretrained weights on Cityscapes
- [x] TensorRT implementation
- [x] Prediction code

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
   | NDNet-DF2   | 77.0 | [model (code: 002a)](https://pan.baidu.com/s/1hOQecVXspbSvZIXO273SKw) |
   | NDNet-Res18 | 76.5 | [model (code: f9je)](https://pan.baidu.com/s/1O-7wWbQ_4O1ZROdrULeR2A) |
   | NDNet-Res34 | 78.8 | [model (code: 1tsc)](https://pan.baidu.com/s/1D34hLWqJlYwemRQOfqmm6Q) |

2. Using the `test.py` script (check the DATASET.TEST_SET path and TEST.MODEL_FILE path carefully):
   ```
   python tools/test.py --cfg experiments/cityscapes/ndnet_df1_test.yaml
   ```

## Predict an Image
1. Download the Cityscapes pretrained weights.

2. Using the `tools/predict.py` script (check the TEST.MODEL_FILE path carefully):
   ```
   python tools/predict.py --cfg experiments/cityscapes/ndnet_df1_test.yaml \
                           --img_path images/1.png \
                           --save_path predict.png \
                           TEST.MODEL_FILE pretrained_models/NDNet_DF1_Cityscapes.pth
   ```

## Citation
If you find our work useful, please consider citing our paper:

```
@ARTICLE{li2022ndnet,
  author={Li, Shu and Yan, Qingqing and Zhou, Xun and Wang, Deming and Liu, Chengju and Chen, Qijun},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={NDNet: Spacewise Multiscale Representation Learning via Neighbor Decoupling for Real-Time Driving Scene Parsing}, 
  year={2022}
  doi={10.1109/TNNLS.2022.3221745}}
```

## Acknowledgement
HRNet-Semantic-Segmentation <https://github.com/HRNet/HRNet-Semantic-Segmentation>
