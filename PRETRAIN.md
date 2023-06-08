## Pretraining Deblurring MAE and ConvMAE
## Usage

### Install
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

### Data preparation
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

### Training
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