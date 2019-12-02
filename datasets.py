import csv
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from masker import compute_item_mask
from models import VLBertTokenizer
from utils import collate_fn

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = lambda s: transforms.Compose(
    [transforms.Resize(s), transforms.RandomCrop(s), transforms.ToTensor(), normalize])
test_transform = lambda s: transforms.Compose(
    [transforms.Resize(s), transforms.CenterCrop(s), transforms.ToTensor(), normalize])


class EpicKitchens(data.Dataset):
    def __init__(self, split, img_root, annotation_root, bbox_size, max_img_seq_len, max_txt_seq_len,
                 bbox_transform=None, include_whole_img=False, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer or VLBertTokenizer.from_pretrained('bert-base-uncased')
        self.max_img_seq_len = max_img_seq_len
        self.max_txt_seq_len = max_txt_seq_len
        self.include_whole_img = include_whole_img
        self.split = split
        self.img_root = img_root
        self.annotation_root = annotation_root
        self.bbox_transform = bbox_transform(bbox_size) or transforms.Lambda(lambda x: x)
        self.actions = self.process_actions(
            torch.load(os.path.join(annotation_root, 'processed_EPIC_train_action_labels.pth'))['actions'])
        with open(os.path.join(annotation_root, 'EPIC_train_object_labels.csv')) as f:
            self.objects = self.process_objects(csv.DictReader(f))
        with open(os.path.join(annotation_root, 'EPIC_video_info.csv')) as f:
            self.video_info = self.process_video_info(csv.DictReader(f))
        split_data = torch.load(os.path.join(annotation_root, 'splits.pth'))
        self.split_data = split_data
        vocab = split_data['vocab']
        self.vocab = vocab
        if split == 'train':
            split_data = split_data['train_action_uids']
        elif split == 'val':
            split_data = split_data['test_action_uids']
        self.action_uids = set(e for r in split_data for c in r for e in c)
        self.sel_action_uids = set(
            e for i, r in enumerate(split_data) for j, c in enumerate(r) for e in c if
            i in vocab['sel_verb_idxs'] and j in vocab['sel_noun_idxs'])
        self.frames = self.get_frames()

    def get_frames(self):
        frames = []
        for vid_id, d in self.actions.items():
            for act_id, row in enumerate(d):
                if int(row['uid']) in self.action_uids:
                    i = int(row['start_frame'])
                    j = int(row['stop_frame'])
                    frames.extend([(vid_id, act_id, n) for n in range(i, j) if self.objects[vid_id][n]])
        return frames

    def get_raw_image(self, index, bbox=False):
        vid_id, act_id, frame_id = self.frames[index]
        participant_id = vid_id.split('_')[0]
        img_path = os.path.join(self.img_root, participant_id, vid_id, f'frame_{frame_id:010d}.jpg')
        # To use high resolution images, the videos need to be downloaded first
        img = default_loader(img_path)  # this loads a smaller version of the image
        if bbox:
            img_bboxes = []
            objects = self.objects[vid_id][frame_id]
            orig_w, orig_h = self.video_info[vid_id]['res']
            img_w, img_h = img.size
            for obj in objects:
                for t, l, h, w in obj['bbox']:
                    h_scale = img_h / orig_h
                    w_scale = img_w / orig_w
                    t *= h_scale
                    h *= h_scale
                    l *= w_scale
                    w *= w_scale
                    if h < 10 or w < 10: continue  # too thin or narrow? do not add bbox
                    bbox = [int(l), int(t), int(l + w), int(t + h)]
                    img_bboxes.append(img.crop(bbox))
            return img, img_bboxes
        return img

    @staticmethod
    def process_video_info(d):
        ret = {}
        for row in d:
            ret[row['video']] = {'res': list(map(int, row['resolution'].split('x'))), 'fps': row['fps'],
                                 'dur': float(row['duration'])}
        return ret

    @staticmethod
    def process_actions(d):
        ret = defaultdict(list)
        for row in d:
            ret[row['video_id']].append(row)
        return ret

    @staticmethod
    def process_objects(d):
        ret = defaultdict(lambda: defaultdict(list))
        for row in d:
            obj = row
            obj['bbox'] = obj['bounding_boxes']
            if obj['bbox']: obj['bbox'] = eval(obj['bbox'])
            if len(obj['bbox']):
                ret[row['video_id']][int(row['frame'])].append(obj)
        return ret

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i, pad=True, add_special_toks=True):
        vid_id, act_id, frame_id = self.frames[i]
        action = self.actions[vid_id][act_id]
        objects = self.objects[vid_id][frame_id]
        participant_id = vid_id.split('_')[0]

        # Text
        text_tokens = self.tokenizer.tokenize(action['narration'])
        if add_special_toks:
            text_tokens = self.tokenizer.add_special_tokens_vl(text_tokens)
        if pad:
            text_tokens, text_len = self.pad_text(text_tokens)
        else:
            text_tokens = text_tokens[:self.max_txt_seq_len]
            text_len = len(text_tokens)
        text_token_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(text_tokens))

        # Images
        img_path = os.path.join(self.img_root, participant_id, vid_id, f'frame_{frame_id:010d}.jpg')
        img = default_loader(img_path)
        imgs = []
        img_bboxes = []
        img_nouns = []
        if self.include_whole_img and self.max_img_seq_len:
            imgs.append(img)
            img_nouns.append('whole image')
            img_bboxes.append((0, 0, 100, 100))
        orig_w, orig_h = self.video_info[vid_id]['res']
        img_w, img_h = img.size
        for obj in objects:
            if len(imgs) >= self.max_img_seq_len:
                break
            for t, l, h, w in obj['bbox']:
                h_scale = img_h / orig_h
                w_scale = img_w / orig_w
                t *= h_scale
                h *= h_scale
                l *= w_scale
                w *= w_scale
                if h < 10 or w < 10: continue  # too thin or narrow? do not add bbox
                bbox = (l, t, l + w, t + h)
                rel_bbox = (l / img_w, t / img_h, (l + w) / img_w, (t + h) / img_h)
                rel_bbox = tuple(min(100, round(100 * _)) for _ in rel_bbox)
                img_bboxes.append(rel_bbox)
                img_nouns.append(obj['noun'])
                imgs.append(img.crop(bbox))

        img_len = len(imgs)
        if img_len == 0:  # no candidate bboxes apply, create padding
            imgs.append(Image.fromarray(np.zeros((10, 10, 3)).astype(np.uint8)))
            img_bboxes = [(0, 0, 0, 0)]

        img_sequence = torch.stack([self.bbox_transform(img) for img in imgs]) if len(imgs) > 0 else torch.tensor([])
        img_bboxes = torch.LongTensor(img_bboxes)

        if img_len == 0 and not pad:
            img_sequence = img_sequence[:0]
            img_bboxes = img_bboxes[:0]

        if pad:
            img_sequence = self.pad_imgs(img_sequence)
            img_bboxes = self.pad_imgs(img_bboxes)

        return {'imgs': img_sequence, 'text': text_token_ids, 'text_tokens': '/'.join(text_tokens),
                'img_nouns': img_nouns, 'img_bboxes': img_bboxes, 'imgs_len': img_len, 'text_len': text_len,
                'vid_ids': vid_id, 'frame_ids': frame_id, 'act_ids': act_id, 'indices': i}

    def pad_text(self, txt, l=None):
        """
        :param txt: list of text tokens
        """
        l = l or self.max_txt_seq_len
        txt = txt[:l]
        return (txt + [self.tokenizer.pad_token for i in range(l - len(txt))]), len(txt)

    def pad_imgs(self, imgs):
        ret = torch.zeros((self.max_img_seq_len, *tuple(imgs.shape[1:]))).to(dtype=imgs.dtype, device=imgs.device)
        k = min(len(imgs), self.max_img_seq_len)
        ret[:k] = imgs[:k]
        return ret


class EpicKitchensMultiple(EpicKitchens):
    """
    Return multiple related examples in the same sample.

    EpicKitchensMultiple outputs (for each element):

    - `imgs`: tensor(N x 3 x H x W), where N is the total number of images for all the sequences
    - `text`: tensor(T). Contains the IDs of the text for all the sequences, concatenated, and without any special token
    - `text_tokens`: List of length S (number of sequences). Each element contains the text of a sequence, represented
    by the (text) tokens separated by `/`
    - `img_nouns`: List (size S) of lists. Each sublist contains words representing the nouns in the different bboxes of
    the sequence,
    - `img_bboxes`: tensor(N x 4). Bounding boxes for each of the N images
    - `imgs_len`: tensor(S). Number of images for each sequence. Its sum adds up to N
    - `text_len`: tensor(S). Number of tokens for each sequence. Its sum adds up to T
    - `vid_ids`: List (size S).
    - `frame_ids`: List (size S).
    - `act_ids`: List (size S).
    - `indices`: List (size S).
    - `seq_type`: tensor(S). For each sequence, indicates if it is the target (0), a positive (1) of a negative (-1)
    - `target_token_ids`: tensor(M). List of the tokens (represented by token ID, not position) that have to be masked.
    They are determined according to the mode and masking policy. Note that in `text` they are still NOT masked
    - `num_seqs`: integer representing S,
    - `num_tgt_toks`: integer representing M,
    - `tok_groups`: List of length L where L is the number of words to be predicted. For each word, `tok_group` contains
    the number of tokens that represent that word. The sum of all the values in `tok_groups` is M. These tokens are in
    order in `target_token_ids`, so if `tok_groups` is \[2, 3\], it means that `target_token_ids` has 5 elements, the
    first 2 represent one word, and the last 3 another word. `tok_groups` is only used in testing, when
    self.test_masking_policy is specified and not `random`. During training it is [].
    - `tok_group_labels`: List of length L, where each element contains a string mentioning the part of speech of the
    corresponding target word. During training it is [].
    """

    def __init__(self, split, img_root, annotation_root, bbox_size, max_img_seq_len, max_txt_seq_len,
                 bbox_transform=None, include_whole_img=False, only_verb_noun=False, tokenizer=None, max_negatives=0,
                 max_positives=0, min_negatives=0, min_positives=0, test_masking_policy=None, negs_per_pos=1, **kwargs):
        super().__init__(split, img_root, annotation_root, bbox_size, max_img_seq_len, max_txt_seq_len,
                         bbox_transform=bbox_transform, include_whole_img=include_whole_img,
                         only_verb_noun=only_verb_noun, tokenizer=tokenizer, **kwargs)

        self.min_negatives = min_negatives
        self.min_positives = min_positives
        self.max_negatives = max_negatives
        self.max_positives = max_positives
        self.negs_per_pos = negs_per_pos
        self.test_masking_policy = test_masking_policy

        self.frame_words = defaultdict(list)

        frame_words_path = os.path.join(annotation_root, f'{self.split}_frame_words.pth')
        try:
            self.frame_words = torch.load(frame_words_path)
        except:
            for i, (vid_id, act_id, frame_id) in tqdm(enumerate(self.frames), total=len(self.frames),
                                                      desc=f'Creating frame words for dataset {self.split} split'):
                narration = self.actions[vid_id][act_id]['narration']
                token_ids = self.tokenizer.encode(narration)
                for token_id in token_ids:
                    self.frame_words[token_id].append((i, vid_id, act_id))
            torch.save(self.frame_words, frame_words_path)

    def __getitem__(self, i):
        data0 = super().__getitem__(i, pad=False, add_special_toks=False)
        data0['seq_type'] = 0

        data = [data0]
        vid_id = data0['vid_ids']

        eval_mask = self.test_masking_policy is not None and self.test_masking_policy != 'random'

        if eval_mask:
            vid_id, act_id, frame_id = self.frames[data0['indices']]
            item_p_mask, pos_list = compute_item_mask(self.actions[vid_id][act_id], data0['text'],
                                                      self.split_data, self.test_masking_policy,
                                                      self.tokenizer)

            tok_groups = []
            tok_group_labels = []
            tok_starts_ends = []
            try:
                for pos, l, grp in pos_list:
                    tok_groups.append(l)
                    tok_group_labels.append(grp)
                    tok_starts_ends.append((pos, pos + l))
                tgt_token_ids = list(set(data0['text'][item_p_mask.bool()].tolist()))
                n_neg = len(tgt_token_ids) * self.negs_per_pos  # farm as many negatives as positives
            except:
                n_neg = 0
                tgt_token_ids = []

        else:
            n_pos = random.randint(self.min_positives, self.max_positives)
            n_neg = random.randint(self.min_negatives, self.max_negatives)

            tgt_token_ids = set(
                filter(lambda t: t not in self.tokenizer.convert_tokens_to_ids(self.tokenizer.all_special_tokens),
                       data0['text'].tolist()))
            tgt_token_ids = random.sample(tgt_token_ids, n_pos) if len(tgt_token_ids) > n_pos else tgt_token_ids

        pos_txt_ids = set()
        cnt_positives = 0
        for idx, token_id in enumerate(tgt_token_ids):
            if token_id in pos_txt_ids:
                continue
            if cnt_positives >= self.max_positives:
                # tgt_token_ids[idx] = 0
                break
            # select positive
            candidate_positives = self.frame_words[token_id]
            if not len(candidate_positives):
                tgt_token_ids[idx] = 0
                continue
            # remove all frames belonging to the same video
            candidate_positives_same = [i for i, _, _ in filter(lambda x: x[1] == vid_id, candidate_positives)]
            candidate_positives_different = [i for i, _, _ in filter(lambda x: x[1] != vid_id, candidate_positives)]
            if len(candidate_positives_different) > 0:
                j = random.choice(candidate_positives_different)
            else:
                # only if that word does not exist in any other example, use element from the same action as positive
                j = random.choice(candidate_positives_same)
            data_pos = super().__getitem__(j, pad=False, add_special_toks=False)
            if data_pos['imgs'] is None:
                tgt_token_ids[idx] = 0
                continue
            data_pos['seq_type'] = 1
            data.append(data_pos)
            pos_txt_ids.update(data_pos['text'].tolist())
            cnt_positives += 1

        n_neg = min(n_neg, self.max_negatives)
        n_neg = max(n_neg, self.min_negatives)
        while n_neg:
            j = random.randint(0, self.__len__() - 1)
            data_neg = super().__getitem__(j, pad=False, add_special_toks=False)
            if data_neg['imgs'] is None or set(data_neg['text'].tolist()) & set(tgt_token_ids):
                continue
            data_neg['seq_type'] = -1
            data.append(data_neg)
            n_neg -= 1

        random.shuffle(data)

        collated_data = collate_fn(data, cat_tensors=True)
        collated_data['target_token_ids'] = torch.LongTensor(list(tgt_token_ids))
        collated_data['num_seqs'] = len(data)
        collated_data['num_tgt_toks'] = len(tgt_token_ids)
        if eval_mask:
            collated_data['tok_groups'] = tok_groups
            collated_data['tok_group_labels'] = tok_group_labels
        else:
            collated_data['tok_groups'] = []
            collated_data['tok_group_labels'] = []
        return collated_data
