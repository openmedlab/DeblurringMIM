
### Deblurring Masked Image Modelling is Better Recipe for Ultrasound Image Analysis

### Introduction
Our initial paper is presented in [Deblurring Masked Autoencoder is Better Recipe for Ultrasound Image Recognition]().

Compared with the initial version which propose a novel deblurring MAE for ultrasound image recognition, this repository is an extension which has the following enhancements:
- We extended the deblurring idea from only MAE to two MIM-based approaches ([MAE](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) and [ConvMAE](https://arxiv.org/pdf/2205.03892.pdf) ).
- We increased the number of thyroid ultrasound images for pretraining from 10,675 to 280,000.
- We extended the downstram task from only classification to classification and segmentation.

*The corresponding extended paper is still working in progress, we will release it soon.*


### Method
![tenser](figure/main.drawio.png)

### Pre-trained checkpoints
The pre-trained models including our proposed deblurring (Deblurring MAE and Deblurring ConvMAE), as well as the vanilla MAE and ConvMAE. All models are pretrained with 280,000 thyroid ultrasound images.
The following table provides the pre-trained checkpoints:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">MAE(Base)</th>
<th valign="bottom">ConvMAE(Base)</th>

<!-- TABLE BODY -->
<tr><td align="left">Vanilla</td>
<td align="center"><a href="https://drive.google.com/file/d/1bXcf6A0kQccnpe3Tk2l0-xWl5HSe4fJ2/view?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1Tv9yjWM9sqxmd60GEIT3WwnT3CHt0kjg/view?usp=drive_link">download</a></td>
</tr>
<tr><td align="left">Deblurring</td>
<td align="center"><a href="https://drive.google.com/file/d/1cOmi7OaqPSMrds7gVZfzTrv8ftEWIH_u/view?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/19PdQpkOOEAi-4aJvVnwB2JTyP-KKa5pp/view?usp=drive_link">download</a></td>
</tr>
</tbody></table>

### Fine-tuning Results
We provide the fine-tuning segmentation results on publicly available [TN3K dataset](https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view?usp=sharing):

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Architecture</th>
<th valign="bottom">Pretraining</th>
<th valign="bottom">IoU (%)</th>
<tr>
<td><a href="https://www.sciencedirect.com/science/article/pii/S0010482522010976?casa_token=_EBAHS8oNlEAAAAA:WmsXBHQAPKT9-ou7Exth9mw3vJvH5HfVe6PP0UC4V5l-5QF_gU37qEsBOKj73wap1F2v3PL5uQ">TRFE+</a></td>
<td>-</td>
<td>-</td>
<td>71.38</td>
</tr>
<tr>
<td>ConvMAE</td>
<td>UPerNet(ConViT-B)</td>
<td>ImageNet</td>
<td>72.98</td>
</tr>
<tr>
<td>ConvMAE</td>
<td>UPerNet(ConViT-B)</td>
<td>US-280K</td>
<td>75.18</td>
</tr>
<tr>
<td>Deblurring ConvMAE</td>
<td>UPerNet(ConViT-B)</td>
<td>US-280K</td>
<td>77.31</td>
</tr>
</tbody></table>

### installation
- Clone this repo:

```bash
git clone https://github.com/MembrAI/DeblurringMAE
cd DeblurringMAE
```

- Create a conda environment and activate it:
```bash
conda create -n deblurringmae python=3.7
conda activate deblurringmae
```

- Install `Pytorch==1.8.0` and `torchvision==0.9.0` with `CUDA==11.1`

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- Install `timm==0.3.2`

```bash
pip install timm==0.3.2
```

### Pretraining

#### Data preparation
Prepareing the original dataset follow this format:
```tree data
dataset_orig
  ├── train
      ├── class1
      │   ├── img1.png
      │   ├── img2.png
      │   └── ...
      ├── class2
      │   ├── img3.png
      │   └── ...
      └── ...
  ├── val
      ├── class1
      │   ├── img4.png
      │   ├── img5.png
      │   └── ...
      ├── class2
      │   ├── img6.png
      │   └── ...
      └── ...
```
For deblurring pretraining, you also need to apply image blurring operation on the original dataset to prepare blurred dataset:
```bash 
python blurred_images.py --src_dir /path/to/dataset_orig/ --dst_dir /path/to/dataset_blurred/ \
     --method gaussian --sigma 1.1
```

#### Running Pretraining Scripts
To pretrain the deblurring MAE, run the following on 1 nodes with 8 GPUs each:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 main_pretrain.py \
--model dmae_vit_base_patch16 --output /path/to/saved/weights/ \
--data_path_blurred /path/to/dataset_blurred/ \
--data_path_orig /path/to/dataset_orig/ --batch_size 32
```

To pretrain the deblurring ConvMAE, run the following on 1 nodes with 8 GPUs each:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 main_pretrain.py \
--model dconvmae_convvit_base_patch16 --output /path/to/saved/weights/ \
--data_path_blurred /path/to/dataset_blurred/ \
--data_path_orig /path/to/dataset_orig/ --batch_size 32
```

To pretrain the vanilla MAE, run the following on 1 nodes with 8 GPUs each:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 main_pretrain.py \
--model mae_vit_base_patch16 --output /path/to/saved/weights/ \
--data_path_blurred /path/to/dataset_orig/ \
--data_path_orig /path/to/dataset_orig/ --batch_size 32
```


To pretrain the vanilla ConvMAE, run the following on 1 nodes with 8 GPUs each:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 main_pretrain.py \
--model convmae_convvit_base_patch16 --output /path/to/saved/weights/ \
--data_path_blurred /path/to/dataset_orig/ \
--data_path_orig /path/to/dataset_orig/ --batch_size 32
```


### Fine-tuning for classification
#### Data preparation
Preparing the dataset for classification follow this format:
```tree data
dataset
  ├── train
  │   ├── class1
  │   │   ├── img1.png
  │   │   ├── img2.png
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.png
  │   │   └── ...
  │   └── ...
  └── val
  │   ├── class1
  │   │   ├── img4.png
  │   │   ├── img5.png
  │   │   └── ...
  │   ├── class2
  │   │   ├── img6.png
  │   │   └── ...
  │   └── ...
  └── test
      ├── class1
      │   ├── img7.png
      │   ├── img8.png
      │   └── ...
      ├── class2
      │   ├── img9.png
      │   └── ...
      └── ...
```
Note that for fine-tuning the deblurring MIM approaches, you should also need to apply image blurring operation on the original images.

#### Training for classification
To finetune deblurring MAE or vanilla MAE training for classification, run the following on single GPU:
```bash
python main_finetune.py --seed 0 \
    --data_path  /path/to/dataset/  \
    --output_dir /path/to/saved/weights/ \
    --model vit_base_patch16 --finetune ${PRETRAIN_CHKPT} \
    --blr 1e-4 --batch_size 256
```
You can change the model parameter for deblurring ConvMAE or  vanilla ConvMAE.
#### Evaluation for classification
To evaluate the fine-tuned deblurring MAE or vanilla MAE for classification, run the following on single GPU:
```bash
python main_finetune.py --batch_size 256  \
--model vit_base_patch16 \
--data_path /path/to/dataset/ --nb_classes 2 \
--output_dir  /path/to/save/results/ \
--resume ${FINETUNE_CHKPT} --eval 
```

### Fine-tuning for segmentation
Please follow the [segmentation guide of ConvMAE](https://github.com/Alpha-VL/ConvMAE/blob/main/SEG/SEGMENTATION.md) to configure the environment.

#### Data preparation
Please follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md) to prepare the segmentation dataset. Note that for fine-tuning the deblurring MIM approaches, you should also apply image blurring operation on the original images.

#### Training for segmentation
Download the pretrained deblurring ConvMAE model [here](https://drive.google.com/file/d/19PdQpkOOEAi-4aJvVnwB2JTyP-KKa5pp/view?usp=drive_link).

```bash
./tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS>  --work-dir <SAVE_PATH> --options model.pretrained=<PRETRAINED_MODEL_PATH>
```

For example:
```bash
./tools/dist_train.sh \
    configs/de_convmae/upernet_de_convmae_us280k_base_gaussian_800_512_tn3k.py 8 \
    --work-dir /path/to/save \
    --options model.pretrained=/path/to/pretrained/weights
```
#### Evaluation for segmentation
```
./tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```
We provide the fine-tuned checkpoint [here]() tuned on the [TN3K dataset](https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view?usp=sharing).
Run 
```
./tools/dist_test.sh configs/de_convmae/upernet_de_convmae_us280k_base_gaussian_800_512_tn3k.py /path/to/finetuned/weights 8 --eval mIoU
```

This should give
```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background | 96.29 | 98.32 |
|   nodule   | 77.31 | 85.94 |
+------------+-------+-------+
```

### Acknowledgement
The pretraining and finetuning of our project are based on [MAE](https://github.com/facebookresearch/mae) and [ConvMAE](https://github.com/Alpha-VL/ConvMAE). The segmentation part are based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [mae_segmentation](https://github.com/implus/mae_segmentation). Thanks for their wonderful work.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.