. ~/conda/bin/activate && conda activate && cd /qingbo/ConvMAE-main/ && CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 main_pretrain.py \
--model mae_vit_base_patch16 --output /qingbo/pretrain_test/ \
--data_path_blurred /qingbo/ddpm/ddpm-segmentation/mae/data/us_zk_mae/ \
--data_path_orig /qingbo/ddpm/ddpm-segmentation/mae/data/us_zk_mae/ --batch_size 128
