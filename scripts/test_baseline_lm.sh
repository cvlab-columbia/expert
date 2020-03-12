#!/usr/bin/env bash
# Example of a test script to test the language model for baselines. This tests both the seen words and the compositions
# Please modify the resume_name to select the correct checkpoint
# In case the baseline uses vision, remove the max_img_seq_len parameter (to use the default one)
# Modify the test_masking_policy to "seen_verbs" (if testing seen verbs), "seen_nouns" (if testing seen nouns), 
#  "new_combo_seen_noun_seen_verb_merge" (if testing new compositions) or "seen_combo_seen_noun_seen_verb_merge" (if testing seen compositions).

CUDA_VISIBLE_DEVICES=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9999 \
--nproc_per_node=1 \
main.py \
--fp16 \
--pretrained_cnn \
--test_batch_size 128 \
--name test_baseline_lm \
--include_whole_img \
-j 5 \
--dataset EpicKitchens \
--img_root /path/to/epic-kitchens/data/raw/rgb \
--annotation_root /path/to/epic-kitchens/data/annotations \
--attn_masking isolate_attn \
--test accuracy \
--test_masking_policy seen_nouns \
--max_img_seq_len 0 \
--resume \
--resume_name name_of_the_checkpoint
