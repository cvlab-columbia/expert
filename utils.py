import collections
import os
import shutil
from functools import partial

import numpy as np
from torch._six import int_classes, string_classes, container_abcs
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format, default_collate

try:
    import torch
    from torchvision.transforms import functional as F
except ImportError:
    print('Torch not imported. Will not be able to use functions using it')


def gen_derangement(n):
    ord_rng = torch.arange(n)
    rnd_rng = torch.randperm(n)
    i = 1
    while any(ord_rng == rnd_rng):
        rnd_rng = torch.randperm(n)
        i += 1
    return rnd_rng


# ------------------------------- GENERAL UTILS ---------------------------------- #

def save_checkpoint(model, optim, tokenizer, is_best, epoch, path, amp=None, global_step=-1, fn='checkpoint.pth',
                    fn_best='checkpoint_best.pth',
                    fn_cfg='config.json',
                    fn_tok='tokenizer.pth',
                    args=None):
    model = model.module if hasattr(model, 'module') else model
    checkpoint_fn = os.path.join(path, fn)
    args = vars(args)
    torch.save({'model': model.state_dict(), 'optim': optim.state_dict(), 'epoch': epoch,
                'amp': amp.state_dict() if amp else None, 'global_step': global_step,
                'args': {k: v for k, v in args.items() if k != 'writer'}},
               checkpoint_fn)
    if is_best:
        shutil.copyfile(checkpoint_fn, os.path.join(path, fn_best))
    model.config.to_json_file(os.path.join(path, fn_cfg))
    torch.save(tokenizer, os.path.join(path, fn_tok))
    print(f'Checkpoint saved at: {os.path.join(path, fn_tok)}')


def load_checkpoint(model, optim, path, amp=None, fn='checkpoint.pth', fn_best='checkpoint_best.pth', load_best=False,
                    strict=True):
    model = model.module if hasattr(model, 'module') else model
    checkpoint_fn = os.path.join(path, fn_best if load_best else fn)
    checkpoint = torch.load(checkpoint_fn, map_location=torch.device("cuda", torch.cuda.current_device()))
    model.load_state_dict(checkpoint['model'], strict=strict)
    try:
        optim.load_state_dict(checkpoint['optim'])
    except:
        print('Warning! Not loading optimizer')
    if amp: amp.load_state_dict(checkpoint['amp'])
    return checkpoint['epoch'], checkpoint.get('global_step', -1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)


def gather_score(x, n):
    if torch.distributed.is_initialized():
        xn = torch.Tensor([x * n, n]).cuda()
        torch.distributed.all_reduce(xn)
        x, n = xn
        return x / n
    else:
        return x


# as data gets more complicated, this may be too general and require a custom method in the dataset class
# we are lucky that the index of the [PAD] token in text is 0 so we can naively 0 pad
def collate_fn(batch, ignore_lists=True, pad_before_stack=True, cat_tensors=False):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # taken from pytorch source code - pytorch data collater, but does not do anything with lists (avoids zip behavior)
    f = partial(collate_fn, ignore_lists=ignore_lists, pad_before_stack=pad_before_stack, cat_tensors=cat_tensors)
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if cat_tensors:
            return torch.cat(batch, 0, out=out)
        else:
            if pad_before_stack:
                return pad_sequence(batch, batch_first=True)
            else:
                return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return f([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: f([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(f(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        if ignore_lists:
            return batch
        else:
            transposed = zip(*batch)
            return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
