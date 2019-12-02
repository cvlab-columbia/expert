#!/usr/bin/env bash
# Example of running script to train with a pointing loss and full attention
# Please modify the paths before executing
CUDA_VISIBLE_DEVICES=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9999 \
--nproc_per_node=1 \
main.py \
--fp16 \
--pretrained_cnn \
--train_batch_size 32 \
--test_batch_size 128 \
--num_train_epochs 200 \
--name pointing_full_example \
--vm_loss_margin 1 \
--include_whole_img \
--vm_loss_lambda 30 \
--pointing \
--attn_masking full  \
--dataset EpicKitchensMultiple \
--pointing_loss_lambda 30 \
-j 4 \
--runs_dir /path/to/your/runs \
--checkpoint_dir /path/to/your/checkpoints \
--results_dir /path/to/your/results \
--img_root /path/to/epic-kitchens/data/raw/rgb \
--annotation_root /path/to/epic-kitchens/data/annotations