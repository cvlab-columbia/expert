import argparse
import os
import random
import socket
import warnings

warnings.simplefilter("ignore")

from datetime import datetime

import numpy as np
import torch.utils.data
from pytorch_transformers import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import datasets
import utils
from trainer import Trainer
import models


def get_args():
    # ------------------------------- GENERAL CONFIGURATION ----------------------------- #
    parser = argparse.ArgumentParser()

    # Required parameter for training
    parser.add_argument("--name", type=str, required=False,
                        help="Name of the experiment. It will be the name of the checkpoint and test results directory")
    # Define task
    parser.add_argument('-e', '--validate', action='store_true', help='validate model on val set')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--test_masking_policy', default='random', type=str, help='masking policy during eval')
    parser.add_argument('--dataset', default='EpicKitchens', type=str, help='dataset name')
    parser.add_argument('--pointing', action='store_true',
                        help="Pointing mode. The default points from the context embedding of the masked word to the "
                             "context embeddings of the other input words. '--input_pointing' extends this behavior")
    parser.add_argument('--input_pointing', action='store_true',
                        help="Pointing to the original text embeddings, not the contextual ones. Using queries for "
                             "pointing, not values direcly.")
    parser.add_argument('--attn_masking', default='isolate_attn', type=str,
                        help='Type of attention masking. How sequences and modalities can attend to each other',
                        choices=['bottleneck', 'isolate_attn', 'full', 'full_target_query', 'full_target_query_key'])

    # Task parameters
    parser.add_argument("--p_mask_img", default=1 / 6, type=float, help="Probability of masking an image bounding box.")
    parser.add_argument("--p_mask_txt", default=1 / 3, type=float, help="Probability of masking a text token.")
    parser.add_argument("--p_clobber_other_tgt_txt", default=5 / 6, type=float,
                        help="Probability of masking a target text token in a non-target sequence in input pointing "
                             "mode.")
    parser.add_argument("--p_clobber_other_txt", default=1 / 6, type=float,
                        help="Probability of masking a non-target text token in a non-target sequence in input "
                             "pointing mode.")
    parser.add_argument("--lm_loss_lambda", default=1, type=float, help="LM loss weight.")
    parser.add_argument("--vm_loss_lambda", default=1, type=float, help="LM loss weight.")
    parser.add_argument("--pointing_loss_lambda", default=1, type=float,
                        help="Pointing loss weight (if the pointing parameter is true).")
    parser.add_argument("--input_pointing_loss_lambda", default=1, type=float,
                        help="Input pointing loss weight (if the input_pointing parameter is true).")
    parser.add_argument("--vm_loss_margin", default=1, type=float,
                        help="VM triplet loss margin.")
    parser.add_argument('--bbox_size', type=int, default=112)
    parser.add_argument("--max_img_seq_len", type=int, default=12)
    parser.add_argument("--max_txt_seq_len", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--pretrained_cnn',
                        action='store_true',
                        help="Use pretrained CNN to embed image regions. Overwritten by --resume")
    parser.add_argument('--pretrained_bert',
                        action='store_true',
                        help="Start from pretrained BERT weights. Overwrites model config. Overwritten by --resume")
    parser.add_argument('--include_whole_img',
                        action='store_true',
                        help="Include token for whole image in input")
    parser.add_argument('--max_negatives', type=int, default=2)
    parser.add_argument('--max_positives', type=int, default=2)
    parser.add_argument('--min_negatives', type=int, default=0)
    parser.add_argument('--min_positives', type=int, default=0)
    parser.add_argument('--config_file', default='config.json')

    # Directories
    parser.add_argument('--runs_dir', default='/path/to/your/runs')
    parser.add_argument('-c', '--checkpoint_dir', default='/path/to/your/checkpoints')
    parser.add_argument('--results_dir', default='/path/to/your/results')
    parser.add_argument('--resume', action='store_true', help='resume model training from checkpoint')
    parser.add_argument('--resume_name', help='Experiment name from which to resume')
    parser.add_argument('--no_strict', action='store_true',
                        help='If True (default), the model we load has to have the exact same parameters')
    parser.add_argument('--resume_latest', action='store_true',
                        help='resume model training from latest, not best, checkpoint')
    parser.add_argument('--img_root', default='/path/to/epic-kitchens/data/raw/rgb')
    parser.add_argument('--annotation_root', default='/path/to/epic-kitchens/data/annotations')

    # Optimization
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--test_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-4, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fused_adam', action='store_true',
                        help="If fp16 is selected, whether to use Fused Adam optimizer (will set opt_level to O2 and "
                             "keep_batchnorm_fp32 to False)")
    parser.add_argument('--opt_level', default='O1', help='optimization level for fp16 training')
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # Other
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--print_freq', '-p', default=100, type=int, help='print frequency')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--negs_per_pos', type=int, default=1,
                        help='number of negative examples per positive example in pointing evaluation')
    parser.add_argument('--debug', action='store_true', help="Debug (no writing to disk at all)")

    args = parser.parse_args()

    # control value of args
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not args.resume:
        assert args.name is not None and len(args.name) > 0
        args.name = args.name + '_' + current_time + '_' + socket.gethostname()
    else:
        assert args.resume_name is not None and len(args.resume_name) > 0
        args.name = args.resume_name

    assert not (args.validate and args.test), \
        "--validate and --test cannot be active at the same time. Please choose one"

    if args.pointing:
        assert 'Multiple' in args.dataset, 'The pointing loss can only work with multiple sequences'
    if args.input_pointing:
        assert args.pointing, 'Input pointing mode implies doing pointing'

    return args


def main():
    args = get_args()

    seed = args.seed
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args.runs_dir = os.path.join(args.runs_dir, args.name)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.name)
    args.results_dir = os.path.join(args.results_dir, args.name)

    if not args.debug:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.step_n_gpus = args.n_gpu
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        args.step_n_gpus = torch.distributed.get_world_size()

    # -------------------------------- INSTANTIATE MAIN ACTORS ----------------------------- #

    # --------------- Create dataset ---------------- #
    tokenizer = torch.load(os.path.join(args.checkpoint_dir, 'tokenizer.pth')) \
        if os.path.exists(os.path.join(args.checkpoint_dir, 'tokenizer.pth')) and args.resume else None
    if args.dataset == 'EpicKitchens':
        train_dataset = datasets.EpicKitchens(split='train', bbox_transform=datasets.train_transform,
                                              tokenizer=tokenizer, **vars(args))
        test_dataset = datasets.EpicKitchens(split='val', bbox_transform=datasets.test_transform,
                                             tokenizer=train_dataset.tokenizer, **vars(args))
    elif args.dataset == 'EpicKitchensMultiple':
        train_dataset = datasets.EpicKitchensMultiple(split='train', bbox_transform=datasets.train_transform,
                                                      tokenizer=tokenizer,
                                                      **vars(args))
        test_dataset = datasets.EpicKitchensMultiple(split='val', bbox_transform=datasets.test_transform,
                                                     tokenizer=train_dataset.tokenizer,
                                                     **vars(args))
    else:
        raise Exception('The dataset you selected is not implemented')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                               shuffle=args.local_rank == -1,
                                               num_workers=args.workers, pin_memory=True,
                                               sampler=DistributedSampler(
                                                   train_dataset) if args.local_rank != -1 else None,
                                               collate_fn=utils.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              shuffle=args.local_rank == -1, num_workers=args.workers, pin_memory=True,
                                              sampler=DistributedSampler(
                                                  test_dataset) if args.local_rank != -1 else None,
                                              collate_fn=utils.collate_fn)

    # -------------- Create model --------------- #
    try:
        tokenizer.img_token = tokenizer.bos_token
        tokenizer.txt_token = tokenizer.cls_token
    except:
        pass
    if args.resume:
        try:
            model = models.load_arch(args.checkpoint_dir, args, pretrained=args.pretrained_bert, tok=tokenizer,
                                     fn_cfg=args.config_file)
        except FileNotFoundError:  # no tokenizer saved: old serialization paradigm
            model = models.load_arch(args.checkpoint_dir, args, pretrained=args.pretrained_bert,
                                     tok=train_dataset.tokenizer, fn_cfg=args.config_file)
    else:
        model = models.load_arch('defaults', args=args, pretrained=args.pretrained_bert, tok=train_dataset.tokenizer,
                                 fn_cfg=args.config_file)
    model.to(device)

    if not model.tokenizer:
        model.tokenizer = train_dataset.tokenizer
        model.embeddings.tokenizer = train_dataset.tokenizer

    if args.fp16:
        try:
            from apex.optimizers import FusedAdam
            from apex import amp, optimizers
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        if args.fused_adam:
            args.opt_level = "O2"
            args.loss_scale = None
            args.keep_batchnorm_fp32 = False
            optim = FusedAdam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, bias_correction=False)
        else:
            args.keep_batchnorm_fp32 = None
            optim = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False, eps=args.adam_epsilon)

        if args.loss_scale == 0:
            args.loss_scale = None

        model, optim = amp.initialize(model, optim, opt_level=args.opt_level, loss_scale=args.loss_scale,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32, verbosity=0)
    else:
        amp = None
        optim = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False, eps=args.adam_epsilon)

    if args.resume:
        epoch, global_step = utils.load_checkpoint(model, optim, args.checkpoint_dir, amp=amp,
                                                   load_best=not args.resume_latest, strict=not args.no_strict)
    else:
        epoch = global_step = -1

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            from torch.nn.parallel import DistributedDataParallel as DDP
            print('Using PyTorch DDP - could not find Apex')
        model = DDP(model, delay_allreduce=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # When using torch.distributed.launch, the batch size in args is per GPU. Without it, when using DataParallel,
    # the batch size is in total. It is important to consider this second case because when measuring the maximum size
    # of the elements in the batch, in the training loop all the elements are available, but in the forward pass only
    # the elements for each GPU are available. In the torch.distributed.launch case, the training loop is done
    # independently in every GPU.

    # --------------- Instantiate trainer --------------- #
    # print('Instantiating trainer', flush=True)
    # test_loader_total = {'val': test_loader, 'train': train_loader, 'test': test_loader}
    trainer = Trainer(model, optim, train_loader, test_loader, args, epoch, global_step=max(global_step, 0),
                      test_mode=args.test is not '')

    # ------------------------- Others ----------------------- #
    args.writer = SummaryWriter(
        log_dir=args.runs_dir if not args.debug and args.test is '' else '/tmp') if args.local_rank <= 0 else None

    # ----------------------------------- TRAIN ------------------------------------------ #

    if args.validate:
        trainer.run_epoch(epoch=None, train=False)
    elif args.test:
        trainer.test(args.test_masking_policy)
    else:
        trainer.train()


if __name__ == '__main__':
    main()
