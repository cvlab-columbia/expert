#!/usr/bin/env bash
# Example of a test script to test pointing to new words for both EXPERT models and baseline.
# Please modify the resume_name to select the correct checkpoint
# Modify the test_masking_policy to "new_verbs" (if testing new verbs), or "new_nouns" (if testing new nouns)
# Modify the negs_per_pos to 2 if you want to replicate the 2:1 ratio shown in the paper

CUDA_VISIBLE_DEVICES=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9999 \
--nproc_per_node=1 \
main.py \
--fp16 \
--pretrained_cnn \
--dataset EpicKitchensMultiple \
--img_root /path/to/epic-kitchens/data/raw/rgb \
--annotation_root /path/to/epic-kitchens/data/annotations \
--test_batch_size 64 \
--vm_loss_margin 1 \
--resume \
--resume_name name_of_the_checkpoint \
--test accuracy \
--test_masking_policy new_verb \
--min_positives 0 \
--min_negatives 0 \
--max_positives 100 \
--max_negatives 100 \
--pointing \
--include_whole_img \
--negs_per_pos 1 \
--attn_masking full \
