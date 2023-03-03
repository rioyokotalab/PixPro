# Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning

By [Zhenda Xie](https://scholar.google.com/citations?user=0C4cDloAAAAJ)\*, [Yutong Lin](https://scholar.google.com/citations?user=mjUgH44AAAAJ)\*, [Zheng Zhang](https://www.microsoft.com/en-us/research/people/zhez/), [Yue Cao](http://yue-cao.me), [Stephen Lin](https://www.microsoft.com/en-us/research/people/stevelin/) and [Han Hu](https://ancientmooner.github.io/).

This repo is an official implementation of ["Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning"](https://arxiv.org/abs/2011.10043) on PyTorch.


[Jump to Instruction added by this fork repository](#instruction-added-by-this-forked-repository) 

## Introduction

`PixPro` (pixel-to-propagation) is an unsupervised visual feature learning approach by leveraging pixel-level pretext tasks. The learnt feature can be well transferred to downstream dense prediction tasks such as object detection and semantic segmentation. `PixPro` achieves the best transferring performance on Pascal VOC object detection (`60.2 AP` using C4) and COCO object detection (`41.4 / 40.5 mAP` using FPN / C4) with a ResNet-50 backbone.

<div align="center">
    <img src="demo/github_teaser.png" height="300px" />
    <p>An illustration of the proposed <b><em>PixPro</em></b> method.</p>
</div>
<div align="center">
    <img src="demo/github_pixpro_pipeline.png" height="160px" />
    <p>Architecture of the <b><em>PixContrast</em></b> and <b><em>PixPro</em></b> methods.</p>
</div>


## Citation

```
@article{xie2020propagate,
  title={Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning},
  author={Xie, Zhenda and Lin, Yutong and Zhang, Zheng and Cao, Yue and Lin, Stephen and Hu, Han},
  conference={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

## Main Results

### _PixPro pre-trained models_

|Epochs|Arch|Instance Branch|Download|
|:---:|:---:|:---:|:---:|
|100|ResNet-50||[script](tools/pixpro_base_r50_100ep.sh) \| [model](https://drive.google.com/file/d/1oZwYSLSYSOzTLtOFdi4jXW6T5cGkQTgD/view?usp=sharing) |
|400|ResNet-50||[script](tools/pixpro_base_r50_400ep.sh) \| [model](https://drive.google.com/file/d/1Ox2RoFbTrrllbwvITdZvwkNnKUQSUPmV/view?usp=sharing) |
|100|ResNet-50|:heavy_check_mark:|-|
|400|ResNet-50|:heavy_check_mark:|-|

### _Pascal VOC object detection_

#### Faster-RCNN with C4

|Method|Epochs|Arch|AP|AP<sub>50</sub>|AP<sub>75</sub>|Download|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Scratch|-|ResNet-50|33.8|60.2|33.1|-|
|Supervised|100|ResNet-50|53.5|81.3|58.8|-|
|MoCo|200|ResNet-50|55.9|81.5|62.6|-|
|SimCLR|1000|ResNet-50|56.3|81.9|62.5|-|
|MoCo v2|800|ResNet-50|57.6|82.7|64.4|-|
|InfoMin|200|ResNet-50|57.6|82.7|64.6|-|
|InfoMin|800|ResNet-50|57.5|82.5|64.0|-|
|[PixPro (ours)](tools/pixpro_base_r50_100ep.sh)|100|ResNet-50|58.8|83.0|66.5|[config](transfer/detection/configs/Pascal_VOC_R_50_C4_24k_PixPro.yaml) \| [model](https://drive.google.com/file/d/1yk-B5qo_jYqrMC_NcnlY5Z7OqWlJj2Nr/view?usp=sharing)|
|[PixPro (ours)](tools/pixpro_base_r50_400ep.sh)|400|ResNet-50|60.2|83.8|67.7|[config](transfer/detection/configs/Pascal_VOC_R_50_C4_24k_PixPro.yaml) \| [model](https://drive.google.com/file/d/1qoiKhAKI-KaWDj1MGHaPrgsQ4dr0RDjh/view?usp=sharing)|

### _COCO object detection_

#### Mask-RCNN with FPN

|Method|Epochs|Arch|Schedule|bbox AP|mask AP|Download|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Scratch|-|ResNet-50|1x|32.8|29.9|-|
|Supervised|100|ResNet-50|1x|39.7|35.9|-|
|MoCo|200|ResNet-50|1x|39.4|35.6|-|
|SimCLR|1000|ResNet-50|1x|39.8|35.9|-|
|MoCo v2|800|ResNet-50|1x|40.4|36.4|-|
|InfoMin|200|ResNet-50|1x|40.6|36.7|-|
|InfoMin|800|ResNet-50|1x|40.4|36.6|-|
|[PixPro (ours)](tools/pixpro_base_r50_100ep.sh)|100|ResNet-50|1x|40.8|36.8|[config](transfer/detection/configs/COCO_R_50_FPN_1x.yaml) \| [model](https://drive.google.com/file/d/1v5gYT-jjY9n-rkvbocQNDuv0UGxD3c7z/view?usp=sharing)|
|PixPro (ours)|100*|ResNet-50|1x|41.3|37.1|-|
|PixPro (ours)|400*|ResNet-50|1x|41.4|37.4|-|

\* Indicates methods with instance branch.

#### Mask-RCNN with C4

|Method|Epochs|Arch|Schedule|bbox AP|mask AP|Download|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Scratch|-|ResNet-50|1x|26.4|29.3|-|
|Supervised|100|ResNet-50|1x|38.2|33.3|-|
|MoCo|200|ResNet-50|1x|38.5|33.6|-|
|SimCLR|1000|ResNet-50|1x|38.4|33.6|-|
|MoCo v2|800|ResNet-50|1x|39.5|34.5|-|
|InfoMin|200|ResNet-50|1x|39.0|34.1|-|
|InfoMin|800|ResNet-50|1x|38.8|33.8|-|
|[PixPro (ours)](tools/pixpro_base_r50_100ep.sh)|100|ResNet-50|1x|40.0|34.8|[config](transfer/detection/configs/COCO_R_50_C4_1x.yaml) \| [model](https://drive.google.com/file/d/1V_IUmaoxGYqq6Dty7AadoYHQgSruWixP/view?usp=sharing)|
|[PixPro (ours)](tools/pixpro_base_r50_400ep.sh)|400|ResNet-50|1x|40.5|35.3|[config](transfer/detection/configs/COCO_R_50_C4_1x.yaml) \| [model](https://drive.google.com/file/d/18zjhg7e_QZHI2JgNWjhrR90DFPJcpQzi/view?usp=sharing)|

## Getting started

### _Requirements_

At present, we have not checked the compatibility of the code with other versions of the packages, so we only recommend the following configuration.

* Python 3.7
* PyTorch == 1.4.0
* Torchvision == 0.5.0
* CUDA == 10.1
* Other dependencies

### _Installation_

We recommand using conda env to setup the experimental environments.
```shell script
# Create environment
conda create -n PixPro python=3.7 -y
conda activate PixPro

# Install PyTorch & Torchvision
conda install pytorch=1.4.0 cudatoolkit=10.1 torchvision -c pytorch -y

# Install apex
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 8a7a332539809adcad88546f492945c4e752ff49
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# Clone repo
git clone https://github.com/zdaxie/PixPro ./PixPro
cd ./PixPro

# Create soft link for data
mkdir data
ln -s ${ImageNet-Path} ./data/imagenet

# Install other requirements
pip install -r requirements.txt
```

### _Pretrain with PixPro_

```shell script
# Train with PixPro base for 100 epochs.
./tools/pixpro_base_r50_100ep.sh
```

### _Transfer to Pascal VOC or COCO object detection_

```shell script
# Convert a pre-trained PixPro model to detectron2's format
cd transfer/detection
python convert_pretrain_to_d2.py ${Input-Checkpoint(.pth)} ./output.pkl  

# Install Detectron2
python -m pip install detectron2==0.2.1 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.4/index.html

# Create soft link for data
mkdir datasets
ln -s ${Pascal-VOC-Path}/VOC2007 ./datasets/VOC2007
ln -s ${Pascal-VOC-Path}/VOC2012 ./datasets/VOC2012
ln -s ${COCO-Path} ./datasets/coco

# Train detector with pre-trained PixPro model
# 1. Train Faster-RCNN with Pascal-VOC
python train_net.py --config-file configs/Pascal_VOC_R_50_C4_24k_PixPro.yaml --num-gpus 8 MODEL.WEIGHTS ./output.pkl
# 2. Train Mask-RCNN-FPN with COCO
python train_net.py --config-file configs/COCO_R_50_FPN_1x_PixPro.yaml --num-gpus 8 MODEL.WEIGHTS ./output.pkl
# 3. Train Mask-RCNN-C4 with COCO
python train_net.py --config-file configs/COCO_R_50_C4_1x_PixPro.yaml --num-gpus 8 MODEL.WEIGHTS ./output.pkl

# Test detector with provided fine-tuned model
python train_net.py --config-file configs/Pascal_VOC_R_50_C4_24k_PixPro.yaml --num-gpus 8 --eval-only \
  MODEL.WEIGHTS ./pixpro_base_r50_100ep_voc_md5_ec2dfa63.pth
```

More models and logs will be released!

## Acknowledgement

Our testbed builds upon several existing publicly available codes. Specifically, we have modified and integrated the following code into this project:

* https://github.com/facebookresearch/moco
* https://github.com/HobbitLong/PyContrast

## Contributing to the project

Any pull requests or issues are welcomed.


# Instruction added by this forked repository

## Results

### [_PixPro with Optical Flow pre-trained models_](https://drive.google.com/drive/folders/1kRdJwMKecPZvPeNab12IB74J1Gf0cb_e?usp=share_link)

|Epochs|Arch|Frames|Optical Flow|Download|
|:---:|:---:|:---:|:---:|:---:|
|2000|ResNet-50|1||[script](tools/pretrain_bdd100k_job_2000ep.sh) \| [model](https://drive.google.com/file/d/1cC-EncqJA55UPeOSJmwl50QdLMmlyCAZ/view?usp=share_link) |
|2000|ResNet-50|2|:heavy_check_mark:|[script](tools/pretrain_bdd100k_job_2000ep_nframe2_gpu16.sh) \| [model](https://drive.google.com/file/d/1ZdtonuVsUioOpPk3pKoPlg4vWzpBUPbE/view?usp=share_link) |
|2000|ResNet-50|6|:heavy_check_mark:|[script](tools/pretrain_bdd100k_job_2000ep_nframe6_gpu16.sh) \| [model](https://drive.google.com/file/d/1sUTKAjCO4PkC3zt0tHaH8BippJ1StbAP/view?usp=share_link) |
<!-- ## [_PixPro with Optical Flow pre-trained models_](https://drive.google.com/drive/folders/1Y-PpTfjofD_OrySYpeqLMHe9AEIqbriL?usp=share_link)
|2000|ResNet-50|1||[script](tools/pretrain_bdd100k_job_2000ep.sh) \| [model](https://drive.google.com/file/d/11rKnPR8zUu1Ontq4OXMzuwBdhw3I3zPu/view?usp=share_link) |
|2000|ResNet-50|2|:heavy_check_mark:|[script](tools/pretrain_bdd100k_job_2000ep_nframe2_gpu16.sh) \| [model](https://drive.google.com/file/d/1HQ9IHHzA4z33rP0CDmRhex77uR29-2XC/view?usp=share_link) |
|2000|ResNet-50|6|:heavy_check_mark:|[script](tools/pretrain_bdd100k_job_2000ep_nframe6_gpu16.sh) \| [model](https://drive.google.com/file/d/1XHwYi0KX9Ui380KcBqTCswOiutiqUeRo/view?usp=share_link) | -->

### _CityScapes Semantic Segmentation_

[config](https://github.com/rioyokotalab/detectron2/blob/dev-v0.6/projects/DeepLab/configs/Cityscapes-SemanticSegmentation/deeplab_v3_R_50_myencoder_mg124_poly_40k_bs8.yaml)

|Method|Epochs|Arch|PreDataset|Frames|mIOU|Download|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BYOL|1000|ResNet-50|ImageNet|-|60.0|-|
|[PixPro](tools/pixpro_base_r50_100ep.sh)|100|ResNet-50|ImageNet|-|58.4|-|
|[PixPro](tools/pretrain_bdd100k_job_2000ep.sh)|2000|ResNet-50|BDD100k|1|53.0|-|
|[PixPro with OF (Ours)](tools/tools/pretrain_bdd100k_job_2000ep_nframe6_gpu16.sh)|2000|ResNet-50|BDD100k|6|53.4|-|

## Getting started

### _Requirements_

At present, we have not checked the compatibility of the code with other versions of the packages, so we only recommend the following configuration.

* Python 3.8.6
* PyTorch == 1.8.2
* Torchvision == 0.9.2
* CUDA == 10.2
* Other dependencies

### _Installation_

We recommand using pyenv virtual env to setup the experimental environments.
```shell script
# Create environment
pyenv virtualenv 3.8.6 pixpro-wt-of-cu102-wandb
pyenv local pixpro-wt-of-cu102-wandb

# Install PyTorch & Torchvision
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102

# Install apex
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 8a7a332539809adcad88546f492945c4e752ff49
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# Clone repo
git clone https://github.com/zdaxie/PixPro
cd ./PixPro

# Create soft link for data
mkdir data
ln -s ${ImageNet-Path} ./data/imagenet
ln -s ${BDD100k-Path} ./data/bdd100k

# Install other requirements
pip install -r requirements.txt
```

### _Pretrain with PixPro_

```shell script
# Train with PixPro base for 2000 epochs.
bash ./tools/pretrain_bdd100k_job_2000ep_nframe6_gpu16.sh
```

### _Transfer to CityScapes Semantic Segmentaion_

```bash
# Convert a pre-trained PixPro model to detectron2's format
cd transfer/detection
python convert_pretrain_to_d2.py ${Input-Checkpoint(.pth)} ./output.pkl  

# Install Detectron2
cd ../../../
git clone https://github.com/rioyokotalab/detectron2
cd detectron2
git checkout dev-v0.6
pip install -e .

# Train detector with pre-trained PixPro model without finetune
python train_net.py --config-file Cityscapes-SemanticSegmentation/deeplab_v3_R_50_myencoder_mg124_poly_40k_bs8.yaml --output ./output 
    --model_path ./output.pkl --no_finetune --num-gpus 4
```

## Evaluation code using detectron2
supported for cityscapes semantic segmentaion, etc..
- https://github.com/rioyokotalab/detectron2

