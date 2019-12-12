import time
from collections import defaultdict

import torch
from tqdm import trange, tqdm

import masker
import tests
import utils
from losses import Losses
from masker import Masker, TestMasker

amp = None


class Trainer:
    """ Class implementing the trainer for the project """

    def __init__(self, model, optimizer, train_loader, test_loader, args, epoch=-1, global_step=0, test_mode=False):

        if args.fp16:
            try:
                from apex import amp
                global amp
                amp = amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = epoch
        self.module = model.module if hasattr(model, 'module') else model  # for data parallel
        self.masking_policies = ['random', 'seen_noun', 'seen_verb', 'seen_combo_seen_noun_seen_verb', 'new_noun',
                                 'new_verb', 'new_combo_seen_noun_seen_verb', 'new_combo_new_noun_new_verb',
                                 'seen_combo_seen_noun_seen_verb_merge', 'new_combo_seen_noun_seen_verb_merge',
                                 'new_combo_new_noun_new_verb_merge']
        if test_mode and not args.pointing:
            self.masker = TestMasker(annotation_root=args.annotation_root, masking_policy=args.test_masking_policy,
                                     tok=self.train_loader.dataset.tokenizer,
                                     p_mask_img=args.p_mask_img, p_mask_txt=args.p_mask_txt)
        else:
            self.masker = Masker(self.train_loader.dataset.tokenizer, **vars(args))
        self.losses = Losses(self.module.cfg, args, **vars(args))
        self.global_step = global_step

    def train(self):

        best_eval = 0
        try:
            for epoch in trange(self.epoch + 1, self.args.num_train_epochs, desc='Training model'):
                if self.args.local_rank != -1:
                    self.train_loader.sampler.set_epoch(epoch)

                self.run_epoch(epoch)

                # Evaluate on validation set
                # The last one is the one that we take into account for the checkpoints
                val_score = self.run_epoch(epoch, train=False)
                # Remember best eval score and save checkpoint
                is_best = val_score > best_eval
                best_eval = max(val_score, best_eval)
                if self.args.local_rank <= 0 and not self.args.debug:
                    print('Saving checkpoint')
                    utils.save_checkpoint(self.model, self.optimizer, self.train_loader.dataset.tokenizer, is_best,
                                          epoch, self.args.checkpoint_dir, amp=amp, global_step=self.global_step,
                                          args=self.args)

        except KeyboardInterrupt:
            if self.args.local_rank <= 0: print(f'You decided to finish the training at epoch {epoch}')

    def run_epoch(self, epoch, train=True):
        """
        During the training loop, we find the following arrays:
        - text_mask_locs:
        Tensor of size B x T, T being the maximum of all the B T's. Each element contains a boolean tensor that contains
        True if the token at that position MUST be masked. This will depend on the `target_token_ids` and whether or not
        the token at position belongs to the target sequence. This masking means that the specific token will be
        predicted (true?) in all the text losses (language model, pointing, episodic), but will not necessarily be
        substituted by a [MASK] token, as this is random and sometimes it stays the same or is substuted by a random
        word.
        - text_no_mask_locs:
        Tensor of size B x T, each element containing a boolean tensor that contains True if in that position the token
        CANNOT be masked.
        - img_no_mask_locs similarly.
        """
        torch.cuda.synchronize()

        # Initialize meters
        avg_batch_time = utils.AverageMeter()
        avg_data_time = utils.AverageMeter()

        list_losses = ['total', 'lm', 'vm']
        list_losses.extend(['pointing'] if self.args.pointing else [])
        list_losses.extend(['input_pointing'] if self.args.input_pointing else [])

        average_meters = defaultdict(lambda: utils.AverageMeter())

        if not train:
            avg_lm_top1 = utils.AverageMeter()
            avg_lm_top5 = utils.AverageMeter()
            avg_pointing_acc = utils.AverageMeter()
            avg_input_pointing_acc = utils.AverageMeter()

        # Switch to train mode
        if train:
            self.model.train()
        else:
            self.model.eval()

        end = time.time()

        with torch.set_grad_enabled(train), \
             tqdm(self.train_loader if train else self.test_loader,
                  desc=f'Training epoch {epoch}' if train else f'Validating {f"epoch {epoch}" if epoch else ""}',
                  disable=self.args.local_rank > 0) as t:
            for batch_idx, data in enumerate(t):
                # Measure data loading time
                avg_data_time.update(time.time() - end)

                # -------------- Organize inputs ------------- #

                img_no_mask_locs = None
                text_no_mask_locs = None
                text_mask_locs = None
                with torch.no_grad():
                    if self.args.pointing:
                        text_mask_locs, text_no_mask_locs = masker.gen_pointing_text_mask_locs(data)

                imgs, vm_labels, neg_vm_labels = self.masker.mask_imgs(data['imgs'].cuda(),
                                                                       no_mask_locs=img_no_mask_locs)

                # Note that this does not mask sep tokens
                text, lm_labels, input_pointing_labels = \
                    self.masker.mask_text(data['text'].cuda(), self.args.input_pointing, no_mask_locs=text_no_mask_locs,
                                          mask_locs=text_mask_locs, **data)
                img_bboxes = data['img_bboxes'].cuda()
                imgs_len = data['imgs_len'].cuda()
                text_len = data['text_len'].cuda()

                img_locs = txt_locs = None

                if self.args.pointing:
                    attn_mask, img_locs, txt_locs = masker.attn_mask_pointing(imgs_len, text_len, data['seq_type'],
                                                                              data['num_seqs'].cuda(),
                                                                              self.args.attn_masking)
                    # The input to the model is:
                    # imgs = [[img0, img1, ..., imgN1, PAD, ..., PAD], [...], [[img0, img1, ..., imgNk, PAD, ..., PAD]]]
                    # where the padding is such that all K in the batch have the same total lenght (minimal padding)
                    # The N images include all the images from all the sequences, concatenated. Only padding at the end

                else:
                    img_attn_mask = \
                        torch.arange(self.args.max_img_seq_len, device=imgs.device)[None, :] < imgs_len[:, None]
                    text_attn_mask = \
                        torch.arange(self.args.max_txt_seq_len, device=imgs.device)[None, :] < text_len[:, None]
                    attn_mask = torch.cat((text_attn_mask[:, :1], img_attn_mask, text_attn_mask[:, 1:]), dim=1)

                # text starts with [IMG] token that gets moved to beginning of input in forward pass

                # -------------- Forward pass ---------------- #

                lm_preds, vm_preds, input_pointing_pred, hidden_states, *_ = \
                    self.model(imgs, text, img_bboxes, attention_mask=attn_mask, img_lens=imgs_len,
                               txt_lens=text_len, img_locs=img_locs, txt_locs=txt_locs)

                # -------------- Compute losses -------------- #

                loss_values = {}
                if self.args.pointing:
                    non_padding_text = (torch.arange(text.shape[1], device=text.device)[None, :] <
                                        text_len.cumsum(dim=1)[:, -1][:, None])
                    non_padding_imgs = (torch.arange(imgs.shape[1], device=imgs.device)[None, :] <
                                        imgs_len.cumsum(dim=1)[:, -1][:, None])
                    loss_values['lm'] = self.losses.lm_loss(lm_preds, lm_labels[non_padding_text])
                    loss_values['vm'] = self.losses.vm_loss(vm_preds, vm_labels[non_padding_imgs],
                                                            neg_vm_labels[non_padding_imgs],
                                                            embedder=self.module.embeddings.img_embeddings)
                else:
                    loss_values['lm'] = self.losses.lm_loss(lm_preds, lm_labels)
                    loss_values['vm'] = self.losses.vm_loss(vm_preds, vm_labels, neg_vm_labels,
                                                            embedder=self.module.embeddings.img_embeddings)
                loss = self.args.lm_loss_lambda * loss_values['lm'] + self.args.vm_loss_lambda * loss_values['vm']

                if self.args.pointing:
                    pointing_loss, (pointing_acc, pointing_cnt) = \
                        self.losses.pointing_loss(data, hidden_states, lm_labels, text, text_len, txt_locs)
                    loss_values['pointing'] = pointing_loss
                    loss += self.args.pointing_loss_lambda * loss_values['pointing']

                if self.args.input_pointing:
                    input_pointing_loss, (input_pointing_acc, input_pointing_cnt), *_ = \
                        self.losses.input_pointing_pointing_loss(
                            input_pointing_pred[0], input_pointing_pred[1],
                            input_pointing_labels, txt_locs, lm_labels,
                            data=data, log=True)
                    loss_values['input_pointing'] = input_pointing_loss
                    loss += self.args.input_pointing_loss_lambda * loss_values['input_pointing']

                if self.args.n_gpu > 1:
                    loss = loss.mean()
                loss_values['total'] = loss

                # --------------- Update model -------------- #

                if train:
                    if self.args.fp16:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        (loss / self.args.gradient_accumulation_steps).backward()

                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    for loss_name in list_losses:  # Record losses
                        average_meters[loss_name].update(loss_values[loss_name].item() /
                                                         self.args.gradient_accumulation_steps, imgs.size(0))

                    if train:
                        if self.args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                        # scheduler.step()  # no scheduler for now
                        self.model.zero_grad()

                # Measure elapsed time
                avg_batch_time.update(time.time() - end)
                end = time.time()

                # ------------- Show information ------------ #

                postfix_kwargs = {}

                if not train:
                    if self.args.pointing:
                        lm_labels = lm_labels[non_padding_text]
                        avg_pointing_acc.update(pointing_acc, pointing_cnt)
                        postfix_kwargs['PointingAcc'] = avg_pointing_acc.avg
                        if self.args.input_pointing:
                            avg_input_pointing_acc.update(input_pointing_acc, input_pointing_cnt)
                            postfix_kwargs['input_pointingAcc'] = avg_input_pointing_acc.avg
                    results = tests.accuracy(lm_preds, lm_labels, topk=(1, 5))
                    avg_lm_top1.update(*results['top1'])
                    avg_lm_top5.update(*results['top5'])
                    postfix_kwargs['LMTop1'] = avg_lm_top1.avg
                    postfix_kwargs['LMTop5'] = avg_lm_top5.avg

                for loss_name in list_losses:
                    postfix_kwargs[loss_name] = average_meters[loss_name].avg

                t.set_postfix(
                    DataTime=avg_data_time.avg,
                    BatchTime=avg_batch_time.avg,
                    **postfix_kwargs
                )

                if train:
                    if self.global_step % self.args.print_freq == 0 and self.args.writer and not self.args.debug:
                        self.args.writer.add_scalars('train/loss', {**postfix_kwargs},
                                                     self.global_step * self.args.train_batch_size * self.args.step_n_gpus)

                self.global_step += 1

        if not train:
            cnt = average_meters['total'].count

            if epoch is not None:
                loss_scalars = {}
                for loss_name in list_losses:
                    loss_scalars[loss_name] = utils.gather_score(average_meters[loss_name].avg, cnt)

                acc_scalars = {
                    'lm_top1': utils.gather_score(avg_lm_top1.avg, cnt),
                    'lm_top5': utils.gather_score(avg_lm_top5.avg, cnt)
                }
                if self.args.pointing:
                    acc_scalars['pointing_acc'] = utils.gather_score(avg_pointing_acc.avg, cnt)
                if self.args.input_pointing:
                    acc_scalars['input_pointing_acc'] = utils.gather_score(avg_input_pointing_acc.avg, cnt)
                if self.args.writer and not self.args.debug:
                    self.args.writer.add_scalars('val/loss', loss_scalars, epoch)
                    self.args.writer.add_scalars('val/acc', acc_scalars, epoch)

            return utils.gather_score(avg_lm_top5.avg, cnt)

    def test(self, masking_policy=None):
        torch.cuda.synchronize()
        if masking_policy == 'all_acc_tests':
            for p in self.masking_policies:
                self.test(p)
        else:
            tests.test_accuracy(self, masking_policy)

