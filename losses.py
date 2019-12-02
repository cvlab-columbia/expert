import random
from math import sqrt

import ipdb
import torch
from torch import nn

from tests import accuracy
from utils import AverageMeter


class Losses:
    def __init__(self, cfg, args, vm_loss_margin=1, **kwargs):
        self.cfg = cfg
        self.args = args
        self.lm_loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.vm_loss_func = nn.CosineSimilarity()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.vm_loss_margin = vm_loss_margin

    def lm_loss(self, lm_preds, masked_lm_labels=None):
        # lm_preds are already filtered (no paddings), with names text_predictions.
        # lm_labels still contain the paddings, so we remove them with the non_padding_text indices.
        # The labels contain -1 at the positions that do not need to be predicted so that is not a problem
        if masked_lm_labels is None:
            return torch.Tensor([0]).to(lm_preds.device)
        return self.lm_loss_func(lm_preds.view(-1, self.cfg.vocab_size), masked_lm_labels.view(-1))

    def vm_loss(self, vm_preds, positive_samples=None, negative_samples=None, embedder=None):
        # vm_preds are already filtered (no paddings), with names image_predictions.
        # vm_labels still contain the paddings, so we remove them with the non_padding_imgs indices.
        # The labels contain -1 at the positions that do not need to be predicted so that is not a problem
        if vm_preds is None or positive_samples is None or positive_samples.mean() == -1 or \
                negative_samples is None or negative_samples.mean() == -1 or embedder is None:
            return torch.Tensor([0]).cuda()  # no images
        try:
            pred_idxs = positive_samples.mean(dim=(2, 3, 4)) != -1
            vm_preds = vm_preds[pred_idxs]
            positive_samples = positive_samples[pred_idxs]
            negative_samples = negative_samples[pred_idxs]
        except IndexError:
            pass
        with torch.no_grad():
            positive_samples = torch.cat([embedder(_) for _ in positive_samples.split(len(vm_preds))])
            negative_samples = torch.cat([embedder(_) for _ in negative_samples.split(len(vm_preds))])
        pos_sim = self.vm_loss_func(vm_preds, positive_samples)
        neg_sim = self.vm_loss_func(vm_preds, negative_samples)
        return (neg_sim - pos_sim + self.vm_loss_margin).clamp(min=0).mean()

    def seq_alignment_loss(self, aux_preds, ys):
        y_hats = aux_preds[:, 0]
        return nn.CrossEntropyLoss()(y_hats, ys)

    def tok_alignment_loss(self, aux_preds, imgs_len, text_len, eps=1e-4, M=1):
        text_len -= 3  # special tokens
        img_reprs = aux_preds[:, 1:self.args.max_img_seq_len + 1]
        txt_reprs = aux_preds[:, self.args.max_img_seq_len + 2:-1]
        img_dot_txt = torch.bmm(img_reprs, txt_reprs.permute(0, 2, 1))
        img_norm_dot_txt_norm = torch.bmm(img_reprs.norm(dim=-1).unsqueeze(-1),
                                          txt_reprs.norm(dim=-1).unsqueeze(-1).permute(0, 2, 1)).clamp(min=eps)
        img_txt_cos_dist = img_dot_txt / img_norm_dot_txt_norm
        img_mask = torch.arange(self.args.max_img_seq_len, device=aux_preds.device)[None, :] < imgs_len[:, None]
        txt_mask = torch.arange(self.args.max_txt_seq_len - 3, device=aux_preds.device)[None, :] < text_len[:, None]
        mask = torch.bmm(img_mask.unsqueeze(-1).to(float), txt_mask.unsqueeze(-1).permute(0, 2, 1).to(float)).to(bool)
        # implicitly ignore values at padding since we look for max value in each item in batch
        img_txt_cos_dist[~mask] = -100
        best_pairs = img_txt_cos_dist.view(len(aux_preds), -1).max(dim=1).values.clone()
        img_txt_cos_dist[ ~mask] = 100
        return (M - best_pairs).clamp(min=0).mean()

    def pointing_loss(self, data, hidden_states, lm_labels, text, text_len, txt_locs, tok_groups=None,
                      tok_group_labels=None, log=False, as_list=False):
        pointing_loss = torch.zeros(()).to(text.device)
        num_toks = 0
        acc = AverageMeter()
        chance_acc = AverageMeter()
        tok_groups = tok_groups or [None for _ in text]
        tok_group_labels = tok_group_labels or [None for _ in text]
        if as_list:
            acc_list = []
        scores_list = []
        for i, (tloc, tlen, hstate, seqtype, target_ids, lm_label, txt) in enumerate(
                zip(txt_locs, text_len, hidden_states, data['seq_type'], data['target_token_ids'],
                    lm_labels, text)):
            cumlens = [0] + tlen.cumsum(0).tolist()
            istart = seqtype.tolist().index(0)
            iend = istart + 1
            other_seq_mask = torch.ones(tlen.sum()).bool().to(txt.device)
            other_seq_mask[cumlens[istart]:cumlens[iend]] = False
            text_hstate = hstate[tloc]
            other_hstate = text_hstate[other_seq_mask]
            tgt_hstate = text_hstate[~other_seq_mask]
            tok_idxs = []
            corr_map = torch.zeros((len(target_ids[target_ids > 0]), len(other_hstate))).bool().to(txt.device)
            if any(_ == 0 for _ in corr_map.shape):
                continue  # no target tokens or no other sequences
            for j, tok_id in enumerate(target_ids.to(txt.device)):
                if tok_id == 0:
                    continue
                tgt_eq_idxs = lm_label[:len(other_seq_mask)][~other_seq_mask] == tok_id
                try:
                    tgt_eq_idx = random.choice(torch.where(tgt_eq_idxs)[0])
                    tok_idxs.append(tgt_eq_idx)
                # sometimes a positive example may not be found? this is probably taken care of in dataloader
                except IndexError:
                    pass
                oth_eq_idxs = txt[:len(other_seq_mask)][other_seq_mask] == tok_id
                try:
                    oth_eq_idx = random.choice(torch.where(oth_eq_idxs)[0])
                    corr_map[j][oth_eq_idx] = True
                except:
                    pass  # no positives
            tgt_hstate = tgt_hstate[[tok_idxs]]
            scores = torch.matmul(tgt_hstate, other_hstate.t()) / sqrt(tgt_hstate.shape[1])
            if log:
                scores_list.extend(nn.Softmax(dim=-1)(scores).tolist())
            gt_scores = corr_map.argmax(dim=1)
            gt_scores[corr_map.max(dim=-1).values == False] = -1
            if self.args.local_rank <= 0 and len(scores) != len(gt_scores):
                ipdb.set_trace()
            pointing_loss += nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)(scores, gt_scores)
            num_toks += len(gt_scores)
            with torch.no_grad():
                try:
                    if as_list:
                        acc_list.append(accuracy(scores, gt_scores, tok_groups=tok_groups[i],
                                                 tok_group_labels=tok_group_labels[i])['top1'][0])
                    acc.update(*accuracy(scores, gt_scores, tok_groups=tok_groups[i],
                                         tok_group_labels=tok_group_labels[i])['top1'])
                    if log:
                        chance_acc.update(100 / len(other_hstate), len(scores))
                except:
                    pass
        if num_toks > 0:
            pointing_loss = pointing_loss / num_toks
        ret = pointing_loss, (acc.avg, acc.count)
        if as_list:
            ret = pointing_loss, acc_list
        if log:
            ret = (*ret, (chance_acc.avg, chance_acc.count), scores_list)
        return ret

    def input_pointing_pointing_loss(self, queries, embedding_outputs, input_pointing_labels, text_locs, lm_labels,
                                     tok_groups=None, tok_group_labels=None, log=False, data=None, as_list=False):
        # Take into account that the episodic_labels are referenced without taking the target sequence positions into
        # account, and thus we have to filter the embeddings with the `other_seq_mask`.
        acc = AverageMeter()
        chance_acc = AverageMeter()
        if as_list:
            acc_list = []
        if log:
            tok_groups = tok_groups or [None for _ in queries]
            tok_group_labels = tok_group_labels or [None for _ in queries]
        loss = torch.tensor(0.).to(queries.device)
        counter = 0
        for i, (query, embedding, eplabel, text_loc, lm_label) in \
                enumerate(zip(queries, embedding_outputs, input_pointing_labels, text_locs, lm_labels)):
            masked_words_indices = [random.choice(torch.where(lm_label == tok)[0]) for tok in
                                    data['target_token_ids'][i].to(query.device) if tok != 0]
            cumlens = [0] + data['text_len'][i].cumsum(0).tolist()
            istart = data['seq_type'][i].tolist().index(0)
            iend = istart + 1
            other_seq_mask = torch.ones(data['text_len'][i].sum()).bool().to(query.device)
            other_seq_mask[cumlens[istart]:cumlens[iend]] = False
            query = query[text_loc][[masked_words_indices]]
            embedding = embedding[text_loc][other_seq_mask]  # The embedding acts as a key
            prod = torch.matmul(embedding, query.transpose(0, 1))
            for j in range(len(masked_words_indices)):  # for all the masked words
                eplabs = eplabel[j]
                for lab in eplabs:  # There is at least one reference of the masked word in the previous sequences
                    # We are considering the tokens from the target sequence as negatives in the denominator
                    gt = torch.LongTensor([lab]).to(prod.device)
                    loss += self.cross_entropy(prod[:, j].unsqueeze(0), gt) / len(eplabs)
                    counter += 1 / len(eplabs)
            if log and sum(map(len, eplabel)) > 0:
                # discard predictions that do not have a ground truth
                prod = prod[:, [len(l) > 0 for l in eplabel]]
                gt = torch.LongTensor([random.choice(lab) for lab in eplabel if len(lab) > 0]).to(prod.device)
                if as_list:
                    acc_list.append(accuracy(prod.transpose(0, 1), gt, tok_groups=tok_groups[i],
                                             tok_group_labels=tok_group_labels[i])['top1'][0])
                acc.update(*accuracy(prod.transpose(0, 1), gt, tok_groups=tok_groups[i],
                                     tok_group_labels=tok_group_labels[i])['top1'])
                chance_acc.update(100 / prod.shape[0], prod.shape[1])

        if counter > 0:
            loss /= counter
        else:
            pass
        if log:
            return loss, (acc.avg, acc.count) if not as_list else acc_list, (chance_acc.avg, chance_acc.count)
        return loss

    def text2image_pointing(self, hidden_states, img_locs, txt_locs, lm_labels, imgs_len, text_len, seq_type):
        # The hidden states are the contextual embedding values
        pointing_loss = torch.zeros(()).to(hidden_states.device)

        if img_locs is None:  # not multiple sequences
            img_attn_mask = torch.arange(self.args.max_img_seq_len, device=hidden_states.device)[None, :] < \
                            imgs_len[:, None].to(hidden_states.device)
            img_locs = torch.zeros([hidden_states.shape[0], hidden_states.shape[1]]).bool()
            img_locs[:, 1:self.args.max_img_seq_len + 1] = img_attn_mask

        for i in range(hidden_states.shape[0]):
            masked_tokens = torch.where(lm_labels[i] > 0)
            if len(masked_tokens[0]) == 0:
                continue
            for masked_token in masked_tokens:
                # Select as positive the image in the target sequence that maximizes the product with the text
                with torch.no_grad():
                    text_embedding = hidden_states[i][2 + self.args.max_img_seq_len + masked_token].transpose(0, 1)
                    positive_location = 1 + torch.argmax(torch.matmul(hidden_states[i][img_locs[i]], text_embedding))
                img_locs_neg_candidates = img_locs
                img_locs_neg_candidates[i, positive_location] = False
                img_locs_neg_candidates = img_locs_neg_candidates.view(-1)
                idx_img_locs_neg_candidates = random.sample(list(torch.where(img_locs_neg_candidates)[0].cpu().numpy()),
                                                            torch.min(torch.tensor(20),
                                                                      img_locs_neg_candidates.sum()).item())
                selected_negatives = hidden_states.view(-1)[idx_img_locs_neg_candidates]
                positive = torch.matmul(hidden_states[i, positive_location], text_embedding)
                positive_and_negatives = torch.cat((positive, selected_negatives))
                # label position of the positive is always 0
                target = torch.tensor([0]).to(hidden_states.device)
                pointing_loss += self.cross_entropy(positive_and_negatives.unsqueeze(0), target)
        return pointing_loss
