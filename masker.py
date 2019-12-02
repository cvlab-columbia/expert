import os
import random

import torch

import utils


# some code from https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_lm_finetuning.py
class Masker:
    def __init__(self, tok, p_mask_img=1 / 6, p_mask_txt=1 / 3, p_clobber_other_tgt_txt=5 / 6,
                 p_clobber_other_txt=1 / 6, **kwargs):
        self.tokenizer = tok
        self.special_token_ids = tok.convert_tokens_to_ids(tok.all_special_tokens)
        self.p_mask_img = p_mask_img
        self.p_mask_txt = p_mask_txt
        self.p_clobber_other_txt = p_clobber_other_txt
        self.p_clobber_other_tgt_txt = p_clobber_other_tgt_txt

    def gen_p_mask_imgs(self, inputs, no_mask_locs=None, mask_locs=None, **kwargs):
        mask = torch.full(inputs.shape[:2], self.p_mask_img)
        if no_mask_locs is not None:
            mask[no_mask_locs] = 0
        if mask_locs is not None:
            mask[mask_locs] = 1
        return mask,

    def mask_imgs(self, inputs, **kwargs):
        """
        At the output, we have a tensor of `labels` and a tensor of `neg_labels`. These have images with all "-1" in the
        images that we do not want to predict in the visual loss. For the images that we want to predict, the `labels`
        contains the original image, and the `neg_vm_labels` contains some negative image (obtained from permuting the
        input). At the loss, the output of the model at that position will have to be closer to the embedding of the
        `labels` than to the embedding of the `neg_labels`. The `inputs` vector is a vector with the original images,
        but at the positions where `vm_labels` and `neg_vm_labels` are NOT -1 (this is, the positions we have to
        predict), `inputs` will contain 0's most of the time (randomly at some places we still maintain the original
        image).\
        """
        labels = inputs.clone()
        neg_labels = inputs.clone()
        for i in range(len(neg_labels)):
            if len(neg_labels[i]) > 1:
                neg_labels[i] = neg_labels[i][utils.gen_derangement(len(neg_labels[i]))]
            else:
                neg_labels[i] = neg_labels[i] * 0 - 1  # We do not compute the visual loss

        # We sample a few tokens in each sequence for masked-LM training
        p_mask, *other_output = self.gen_p_mask_imgs(inputs, **kwargs)
        masked_indices = torch.bernoulli(p_mask).bool()
        img_sums = inputs.sum(dim=tuple(range(2, len(inputs.shape))))
        masked_indices[img_sums == 0] = 0  # zero out masking indicator at padding locations
        labels[~masked_indices] = -1
        neg_labels[~masked_indices] = -1

        # 90% of the time, we replace masked input tokens with zeroes
        indices_replaced = torch.bernoulli(torch.full(labels.shape[:2], 0.9)).bool() & masked_indices
        inputs[indices_replaced] = 0

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return (inputs, labels, neg_labels, *other_output)

    def gen_p_mask_text(self, inputs, no_mask_locs=None, mask_locs=None, p=None, **kwargs):
        mask = torch.full(inputs.shape, self.p_mask_txt if p is None else p)
        if no_mask_locs is not None:
            mask[no_mask_locs] = 0
        if mask_locs is not None:
            mask[mask_locs] = 1
        return mask,

    def mask_text(self, inputs, input_pointing=False, prob_mask=0.8, **kwargs):
        """

        :param prob_mask: probability of masking with [MASK] token if the input is selected to be masked
        :param mask: attention mask to be used in forward pass, value 1 for special tokens

        Returns:
        - `labels`: it contains `-1` at the positions where we do NOT have to predict any token, and the ground truth
        token at the positions where we do have to predict the token. The padding positions are also -1. These
        positions/tokens to be predicted are the ones to be predicted in the language model loss, the pointing loss and
        the episodic pointing loss. For the episodic and the pointing loss, they will not be computed for a masked word
        if the masked word cannot be found previously in the other sequences. When the tokens to be predicted (and thus
        masked) have to be exactly (and only) the ones in `target_token_ids`, we take care of it when creating the
        `no_mask_locs`. Otherwise there may be some other random words in the target sentence that are also masked.
        - `inputs`: input tokens, where the positions that in `labels` are not -1, now contain either the token
        \[MASK\], or with less probability the original token or some random token. Important: this text, as the one in
        `data["text_tokens"]` still does NOT contain the \[SEP\] token. Just the 0 (\[PAD\]) padding to make the batch
        collate possible.
        - `input_pointing_labels`: for each element in the batch, they contain a list of length equal to the masked
        tokens, where masked are all the tokens in `target_token_ids`, that may have been replaced by \[MASK\] or not.
        This is, the tokens to be predicted. Each element of the list is another list with the positions where that
        token (the ground truth, before masking) appears previously in the sentence. This is, where we have to point to.
        These positions are indexed according to the position in `text` (this is, without any \[SEP\]) BUT AFTER
        REMOVING the target sequence.
        """
        labels = inputs.clone()

        # We sample a few tokens in each sequence for masked-LM training
        p_mask, *other_output = self.gen_p_mask_text(inputs, **kwargs)
        masked_indices = torch.bernoulli(p_mask).bool()
        for t in self.special_token_ids:
            # do not predict on special tokens, zero out masking indicator at padding locations
            masked_indices[inputs == t] = 0
        labels[~masked_indices] = -1

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, prob_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(labels.shape, (1 - prob_mask) / 2)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]

        input_pointing_labels = None
        if input_pointing:
            input_pointing_labels = []
            p_mask_input_pointing, *other_output_input_pointing = \
                self.gen_p_mask_text(inputs, no_mask_locs=~kwargs['no_mask_locs'], p=self.p_clobber_other_txt)
            for t in self.special_token_ids:
                p_mask_input_pointing[inputs == t] = 0
            for i in range(inputs.shape[0]):  # iterate over the batch
                input_pointing_labels_i = []
                cumlens = [0] + kwargs['text_len'][i].cumsum(0).tolist()
                istart = kwargs['seq_type'][i].tolist().index(0)
                iend = istart + 1
                tgt_seq_start = cumlens[istart]
                tgt_seq_end = cumlens[iend]
                for token in kwargs['target_token_ids'][i].to(inputs.device):  # words that we masked
                    # append word indices in input that are the same as the word we masked
                    if token == 0:
                        continue
                    tmp = []
                    # position excluding target sequence, since loss is calculated over other sequence tokens only
                    pos_skip_tgt = 0
                    for position in range(inputs.shape[1]):
                        if tgt_seq_start <= position < tgt_seq_end:
                            continue
                        if inputs[i, position] == token:
                            p_mask_input_pointing[i][position] = self.p_clobber_other_tgt_txt
                            tmp.append(pos_skip_tgt)
                        pos_skip_tgt += 1
                    input_pointing_labels_i.append(tmp)
                input_pointing_labels.append(input_pointing_labels_i)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return (inputs, labels, input_pointing_labels, *other_output)


class TestMasker(Masker):
    def __init__(self, masking_policy, annotation_root, **kwargs):
        super().__init__(**kwargs)
        self.test_masking_policy = masking_policy
        try:
            self.split_data = torch.load(os.path.join(annotation_root, 'splits.pth'))
        except:  # hacky for flickr test
            self.split_data = torch.load(os.path.join(annotation_root, 'splits_truecomps.pth'))

    def gen_p_mask_text(self, inputs, actions, **kwargs):
        if self.test_masking_policy == 'random':
            return (*super().gen_p_mask_text(inputs), None, None)
        tok_groups = []
        tok_group_labels = []
        p_mask = torch.full(inputs.shape, 0)
        for idx, (input, action) in enumerate(zip(inputs, actions)):
            # print('before compute item mask')
            item_p_mask, pos_list = compute_item_mask(action, input, self.split_data, self.test_masking_policy,
                                                      self.tokenizer)
            # print('after compute item mask')

            if item_p_mask is None:
                continue

            p_mask[idx][:len(item_p_mask)] = item_p_mask

            for pos, l, grp in pos_list:
                tok_groups.append(l)
                tok_group_labels.append(grp)
            # only relevant parts of speech are masked. we can keep track of length of each token to calculate accuracy
            # grouped by POS in test time. no need to keep track of positions.
        return p_mask, tok_groups, tok_group_labels


def gen_pointing_text_mask_locs(data, only_mask_tgt_toks=False):
    text_mask_locs = torch.zeros_like(data['text']).bool()
    text_no_mask_locs = torch.zeros_like(data['text']).bool()
    for i, (text, seq_types, text_lens, tgt_toks) in enumerate(
            zip(data['text'], data['seq_type'], data['text_len'], data['target_token_ids'])):
        cum_lens = text_lens.cumsum(0)
        tgt_seq_idx = seq_types.tolist().index(0)
        tgt_seq_end = cum_lens[tgt_seq_idx]
        tgt_seq_start = tgt_seq_end - text_lens[tgt_seq_idx]
        tgt_seq_slice = slice(tgt_seq_start, tgt_seq_end)
        for tok in tgt_toks:
            if tok == 0:
                continue
            try:
                eq_idx = random.choice(torch.where(text[tgt_seq_slice] == tok)[0])
                text_mask_locs[i][tgt_seq_slice][eq_idx] = True
            except:
                pass
        text_no_mask_locs[i][:tgt_seq_start] = True
        text_no_mask_locs[i][tgt_seq_end:] = True
    if only_mask_tgt_toks:
        text_no_mask_locs = ~text_mask_locs
    return text_mask_locs, text_no_mask_locs


def compute_item_mask(action, input, split_data, policy, tokenizer):
    item_p_mask = torch.zeros(len(input))
    if 'comps' in action and 'combo' in policy:
        if action['comps']:
            if 'seen_combo' in policy:
                try:
                    verb, noun = random.choice(list(action['comps']))
                except:
                    return None, None
            else:
                verb = None
                for v, n in action['comps']:
                    try:
                        i_v = split_data['vocab']['verbs'].index(v)
                        i_n = split_data['vocab']['nouns'].index(n)
                        if split_data['train'][i_v, i_n] == 0:
                            noun = n
                            verb = v
                    except:
                        pass
                if verb is None:
                    return None, None
            verb = (
                action['tokens'][
                    random.choice([i for i, _ in enumerate(action['lemmas']) if _ == verb])], verb)
            noun = (
                action['tokens'][
                    random.choice([i for i, _ in enumerate(action['lemmas']) if _ == noun])], noun)
        else:
            return None, None
    else:
        try:
            noun = random.choice(
                [(tok, lem) for tok, tag, lem in zip(action['tokens'], action['tags'], action['lemmas']) if
                 tag == 'NN' and lem in split_data['vocab']['nouns']])
        except:
            noun = ('[UNK]', '[UNK]')
            if 'noun' in policy:
                return None, None
        try:
            verb = random.choice(
                [(tok, lem) for tok, tag, lem in zip(action['tokens'], action['tags'], action['lemmas']) if
                 tag == 'VB' and lem in split_data['vocab']['verbs']])
        except:
            verb = ('[UNK]', '[UNK]')
            if 'verb' in policy:
                return None, None
    noun_ids = tokenizer.encode(noun[0])
    noun = noun[1]
    verb_ids = tokenizer.encode(verb[0])
    verb = verb[1]
    pos_list = []
    input = input.tolist()
    # print('are you sure you didn\'t forget to uncomment this?')
    if not is_relevant_example(noun, verb, split_data, policy):
        return None, None
    noun_pos = -1
    verb_pos = -1
    if 'noun' in policy:
        for i in range(0, len(input) - len(noun_ids) + 1):
            if input[i:i + len(noun_ids)] == noun_ids:
                noun_pos = i
                break
    if 'verb' in policy:
        for i in range(0, len(input) - len(verb_ids) + 1):
            if input[i:i + len(verb_ids)] == verb_ids:
                verb_pos = i
                break
    if len(set(noun_ids) & set(verb_ids)):
        return None, None
    if not ('noun' in policy and 'verb' in policy and (
            noun_pos < 0 or verb_pos < 0)):
        if noun_pos > -1:
            item_p_mask[noun_pos:noun_pos + len(noun_ids)] = 1
            pos_list.append((noun_pos, len(noun_ids), 'noun'))
        if verb_pos > -1:
            item_p_mask[verb_pos:verb_pos + len(verb_ids)] = 1
            pos_list.append((verb_pos, len(verb_ids), 'verb'))
        if 'merge' in policy:
            pos_list = [(min(noun_pos, verb_pos), len(noun_ids) + len(verb_ids), 'noun_and_verb')]

    pos_list = sorted(pos_list)
    return item_p_mask, pos_list


def is_relevant_example(noun, verb, split_data, policy):
    try:
        noun_idx = split_data['vocab']['nouns'].index(noun)
        noun_seen = sum(split_data['train'][:, noun_idx]) > 0
    except:
        if 'noun' not in policy:
            noun_seen = False
        else:
            return False
    try:
        verb_idx = split_data['vocab']['verbs'].index(verb)
        verb_seen = sum(split_data['train'][verb_idx]) > 0
    except:
        if 'verb' not in policy:
            verb_seen = False
        else:
            return False
    try:
        noun_and_verb_seen = split_data['train'][verb_idx, noun_idx] > 0
    except:
        if 'noun' not in policy or 'verb' not in policy:
            noun_and_verb_seen = False
        else:
            return False
    want_noun_seen = 'seen_noun' in policy
    want_verb_seen = 'seen_verb' in policy
    want_noun_and_verb_seen = 'seen_combo' in policy
    dontcare_noun = 'all_noun' in policy or 'noun' not in policy
    dontcare_verb = 'all_noun' in policy or 'verb' not in policy
    dontcare_noun_and_verb = 'all_combo' in policy or 'combo' not in policy
    ok_to_continue = (want_noun_seen == noun_seen or dontcare_noun) and (
            want_verb_seen == verb_seen or dontcare_verb) and (
                             want_noun_and_verb_seen == noun_and_verb_seen or dontcare_noun_and_verb)
    return ok_to_continue


def attn_mask_pointing(imgs_len, text_len, seq_types, num_seqs, attn_masking):
    """
    attn_masking options:
    - bottleneck: isolate attention but target images can also look at other images
    - isolate_attn: each image/text can only look at themselves and their corresponding text/images
    - full: full attention across sequences and modalities, no restrictions
    - full_target_query: sequences can attend within themselves (and cross-modality), and target attends everywhere
    - full_target_query_key: same as full_target_query, but now all sequences can also attend to the target
    At the output: The format of the three outputs corresponds to the format of the input of the encoder. It has length
    K. The first token is "\[IMG\]", that signals the start of the images. Then there are the images for the first
    sequence. Then another token ("\[SEP\]") that signals the beginning of the images for the second sequence. Then the
    images for the second sequence, and so on. These images do NOT have padding. When the S sequences of images finish,
    there is the token "\[TXT\]" representing the beginning of the text tokens. Between the last image and the "\[TXT\]"
    token there is a "\[SEP]" token. Then there is the text of the first sequence, then another "\[SEP\]",
    then the text of the second sequence and so on, until the end. Then the rest until K is padding ("\[PAD\]" token).
    Between the last text token and the first "\[PAD\]" token there is a "\[TXT\]" token too
    - img_locs: len K, True where there are text token embeddings.
    - txt_locs: len K, True where there are image features.
    - attn_mask: tensor(K x K) containing the product of query and key where we limit the attention. Depends on
    `attn_masking` and follows the same structure described above.
    """
    max_num_toks = (imgs_len.sum(dim=1) + text_len.sum(dim=1) + num_seqs * 2).max() + 2
    # all images and text, all separation tokens (2 per seq - one for images, one for text), [IMG] and [TXT]
    # aggregation tokens. This assumes all sequences come with >0 image and text tokens
    attn_mask = torch.zeros((len(imgs_len), max_num_toks, max_num_toks))  # batch size x # queries x # keys
    img_locs = torch.zeros((len(imgs_len), max_num_toks)).bool()
    txt_locs = torch.zeros((len(imgs_len), max_num_toks)).bool()
    attn_mask[:, :1] = 1  # "[IMG]" token can attend everywhere
    if attn_masking == 'full':
        attn_mask[:, :] = 1
    for i, (img_len, txt_len, seq_type, seq_cnt) in enumerate(
            zip(imgs_len, text_len, seq_types, num_seqs)):
        img_start = 1
        txt_start = img_start + sum(img_len) + seq_cnt + 1
        attn_mask[i][txt_start - 1, :] = 1  # [TXT] can attend everywhere
        for j, (ilen, tlen, seq) in enumerate(zip(img_len, txt_len, seq_type)):
            if j == seq_cnt:
                break
            if seq == 0 and not attn_masking == 'isolate_attn':
                if attn_masking == 'bottleneck':
                    # text from target sequence can only attend to its image (and itself)
                    attn_mask[i][txt_start:txt_start + tlen + 1, txt_start:txt_start + tlen + 1] = 1
                    attn_mask[i][txt_start:txt_start + tlen + 1, img_start:img_start + ilen + 1] = 1
                    # img from target sequence can attend to itself, to its text, and all other images
                    attn_mask[i][img_start:img_start + ilen + 1, txt_start:txt_start + tlen + 1] = 1
                    img_start_ = 1
                    for k, (ilen_, _, _) in enumerate(zip(img_len, txt_len, seq_type)):
                        attn_mask[i][img_start:img_start + ilen + 1, img_start_:img_start_ + ilen_ + 1] = 1
                        img_start_ += ilen_ + 1
                # full_target_query and full_target_query_key
                elif attn_masking == 'full_target_query' or attn_masking == 'full_target_query_key':
                    # target sequence can attend to all the other sequences
                    attn_mask[i][img_start:img_start + ilen + 1, :] = 1
                    attn_mask[i][txt_start:txt_start + tlen + 1, :] = 1
                    if attn_masking == 'full_target_query_key':
                        # all other sequences can attend to target sequence
                        attn_mask[i][:, img_start:img_start + ilen + 1] = 1
                        attn_mask[i][:, txt_start:txt_start + tlen + 1] = 1
            else:  # positive and negative sequences can only attend to their inputs (both text and image)
                attn_mask[i][img_start:img_start + ilen + 1, img_start:img_start + ilen + 1] = 1
                attn_mask[i][txt_start:txt_start + tlen + 1, txt_start:txt_start + tlen + 1] = 1
                attn_mask[i][img_start:img_start + ilen + 1, txt_start:txt_start + tlen + 1] = 1
                attn_mask[i][txt_start:txt_start + tlen + 1, img_start:img_start + ilen + 1] = 1
            img_locs[i][img_start:img_start + ilen] = True
            txt_locs[i][txt_start:txt_start + tlen] = True
            img_start += ilen + 1
            txt_start += tlen + 1
        attn_mask[i][:, txt_start:] = 0  # we don't want the original sequence to attend to padding
        attn_mask[i][txt_start:] = 0  # we don't want the original sequence to attend to padding

    return attn_mask, img_locs, txt_locs
