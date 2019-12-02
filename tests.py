import itertools
from collections import defaultdict

import torch
from torch.distributed import get_rank
from tqdm import tqdm

import masker
import utils
from masker import gen_pointing_text_mask_locs


def accuracy(output, target, topk=(1,), tok_groups=None, tok_group_labels=None, as_list=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    mask = target != -1

    output = output[mask]
    target = target[mask]

    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])

    res = {}
    if tok_groups:
        try:
            correct = correct.split(tok_groups, dim=1)
        except:
            return res
        try:
            correct = torch.stack([grp.sum(dim=1).cumsum(dim=0) == torch.ones_like(grp).sum(dim=1) for grp in correct],
                                  dim=-1)
        except:
            print('wrong', get_rank())
    for k in topk:
        try:
            if as_list:
                res[f'top{k}'] = correct[:k].max(dim=0).values
            else:
                correct_k = correct[:k].max(dim=0).values.sum(dtype=torch.float32)
                res[f'top{k}'] = ((correct_k * (100.0 / correct.shape[1])).item(), correct.shape[1])
        except:  # if no items are masked in batch, max operation doesn't work since correct is empty tensor
            if as_list:
                res[f'top{k}'] = [0]
            else:
                res[f'top{k}'] = (0, 0)

    if tok_group_labels:
        label_set = set(tok_group_labels)
        for label in label_set:
            for k in topk:
                correct_ = correct[:, [_ == label for _ in tok_group_labels]]
                correct_k = correct_[:k].max(dim=0).values.sum(dtype=torch.float32)
                res[f'{label}_top{k}'] = ((correct_k * (100.0 / correct_.shape[1])).item(), correct_.shape[1])

    return res


def test_accuracy(trainer, masking_policy):
    trainer.masker.test_masking_policy = masking_policy
    trainer.test_loader.dataset.test_masking_policy = masking_policy
    # behavior only differs in testmasker and accuracy function call.
    avg_meters = defaultdict(lambda: utils.AverageMeter())

    # Switch to eval mode
    trainer.model.eval()

    counter_masking = utils.AverageMeter()
    with torch.no_grad(), tqdm(trainer.test_loader, desc=f'Testing {masking_policy}',
                               disable=trainer.args.local_rank > 0) as t:
        for batch_idx, data in enumerate(t):
            imgs = data['imgs'].cuda()
            if trainer.args.pointing:
                text_mask_locs, text_no_mask_locs = gen_pointing_text_mask_locs(data, only_mask_tgt_toks=True)
                mask_kwargs = {
                    'mask_locs': text_mask_locs,
                    'no_mask_locs': text_no_mask_locs,
                }
                text, lm_labels, input_pointing_labels, *_ = \
                    trainer.masker.mask_text(data['text'].cuda(), input_pointing=trainer.args.input_pointing,
                                             **mask_kwargs, **data)
                if sum(map(len, data['tok_groups'])) > 0:
                    tok_groups = data['tok_groups']
                    tok_group_labels = data['tok_group_labels']
                else:
                    tok_groups = None
                    tok_group_labels = None
            else:
                try:
                    mask_kwargs = {
                        'actions': [trainer.test_loader.dataset.actions[v][a] for v, a in
                                    zip(data['vid_ids'], data['act_ids'])]
                    }
                except:
                    mask_kwargs = {
                        'actions': [trainer.test_loader.dataset.actions[trainer.test_loader.dataset.indices[i]] for i in
                                    data['indices'].squeeze().tolist()]
                    }
                text, lm_labels, input_pointing_labels, tok_groups, tok_group_labels = \
                    trainer.masker.mask_text(data['text'].cuda(), input_pointing=trainer.args.input_pointing,
                                             **mask_kwargs, **data)

            img_bboxes = data['img_bboxes'].cuda()
            imgs_len = data['imgs_len'].cuda()
            text_len = data['text_len'].cuda()

            img_locs = txt_locs = None

            if trainer.args.pointing:
                attn_mask, img_locs, txt_locs = masker.attn_mask_pointing(imgs_len, text_len, data['seq_type'],
                                                                          data['num_seqs'].cuda(),
                                                                          trainer.args.attn_masking, counter_masking)

            else:
                img_attn_mask = \
                    torch.arange(trainer.args.max_img_seq_len, device=imgs.device)[None, :] < imgs_len[:, None]
                text_attn_mask = \
                    torch.arange(trainer.args.max_txt_seq_len, device=imgs.device)[None, :] < text_len[:, None]
                attn_mask = torch.cat((text_attn_mask[:, :1], img_attn_mask, text_attn_mask[:, 1:]), dim=1)

            # text starts with [IMG] token that gets moved to beginning of input in forward pass

            lm_preds, vm_preds, input_pointing_pred, hidden_states, *_ = \
                trainer.model(imgs, text, img_bboxes, attention_mask=attn_mask, img_lens=imgs_len,
                              txt_lens=text_len, img_locs=img_locs, txt_locs=txt_locs)

            if trainer.args.pointing:
                pointing_loss, pointing_acc, chance_pointing_acc, pointing_scores = \
                    trainer.losses.pointing_loss(data, hidden_states, lm_labels, text, text_len, txt_locs, tok_groups,
                                                 tok_group_labels, log=True)
                avg_meters['pointing acc'].update(*pointing_acc)
                avg_meters['chance pointing acc'].update(*chance_pointing_acc)
                non_padding_text = (torch.arange(text.shape[1], device=text.device)[None, :] <
                                    text_len.cumsum(dim=1)[:, -1][:, None])
                if trainer.args.input_pointing:
                    input_pointing_loss, input_pointing_acc, chance_input_pointing_acc = \
                        trainer.losses.input_pointing_pointing_loss(input_pointing_pred[0], input_pointing_pred[1], input_pointing_labels,
                                                              txt_locs, lm_labels, text, tok_groups, tok_group_labels,
                                                              log=True, data=data)
                    avg_meters['input_pointing acc'].update(*input_pointing_acc)
                    avg_meters['chance input_pointing acc'].update(*chance_input_pointing_acc)
                if tok_groups is not None:
                    tok_groups = list(itertools.chain.from_iterable(tok_groups))
                    tok_group_labels = list(itertools.chain.from_iterable(tok_group_labels))
                lm_labels = lm_labels[non_padding_text]

            results = accuracy(lm_preds, lm_labels, topk=(1, 5), tok_groups=tok_groups,
                               tok_group_labels=tok_group_labels)
            for k in results:
                avg_meters[k].update(*results[k])

    for k, v in avg_meters.items():
        out = f'{trainer.masker.test_masking_policy} {k}: {utils.gather_score(v.avg, v.count)} on ' \
              f'{utils.gather_score(v.count, 1) * trainer.args.n_gpu} examples'
        if trainer.args.local_rank <= 0:
            print(out, flush=True)

    print(f'Mean number of masked squares: {counter_masking.avg}', flush=True)
