import json
import os.path

import pytorch_transformers.modeling_bert as mb
import torch
import torch.nn.functional as F
from pytorch_transformers import *
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchvision.models import resnet18


class VLBertTokenizer(BertTokenizer):
    TOKENS = {'bos_token': '[IMG]', 'cls_token': '[TXT]', 'additional_special_tokens': [f'[NEW{i}]' for i in range(50)]}

    def __init__(self, vocab_file, **kwargs):
        super().__init__(vocab_file, **kwargs)
        self.add_special_tokens(self.TOKENS)
        self.img_token = self.bos_token
        self.txt_token = self.cls_token

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        t = super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        t.add_special_tokens(cls.TOKENS)
        return t

    def add_special_tokens_vl(self, text_input_ids):
        start = [self.bos_token, self.cls_token]
        end = [self.sep_token]
        return start + text_input_ids + end


class VLBertConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BertVMPredictionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transform = mb.BertPredictionHeadTransform(cfg)
        self.decoder = nn.Linear(cfg.hidden_size,
                                 cfg.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertLayerExtended(nn.Module):
    """
    Small modification from mb.BertLayer to output three outputs and not just one
    """

    def __init__(self, config):
        super(BertLayerExtended, self).__init__()
        self.attention = mb.BertAttention(config)
        self.intermediate = mb.BertIntermediate(config)
        self.output_keys = mb.BertOutput(config)
        self.output_queries = mb.BertOutput(config)
        self.output_values = mb.BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output_keys = self.output_keys(intermediate_output, attention_output)
        layer_output_queries = self.output_queries(intermediate_output, attention_output)
        layer_output_values = self.output_values(intermediate_output, attention_output)
        outputs = (layer_output_keys, layer_output_queries, layer_output_values) + attention_outputs[1:]
        return outputs


class BertEncoderExtended(nn.Module):
    """
    Small modification from mb.BertEncoder to return three outputs in the last layer, instead of one. These three
    values represent "queries", "keys" and "values" to use for the pointing, mimicking the self-attention the model has
    inside.
    """

    def __init__(self, config):
        super(BertEncoderExtended, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([mb.BertLayer(config) for _ in range(config.num_hidden_layers - 1)])
        self.last_layer = BertLayerExtended(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_outputs = self.last_layer(hidden_states, attention_mask, head_mask[-1])
        layer_outputs_keys = layer_outputs[0]
        layer_outputs_queries = layer_outputs[1]
        layer_outputs_values = layer_outputs[2]
        layer_outputs_attention = layer_outputs[3]
        hidden_states = layer_outputs_values

        if self.output_attentions:
            all_attentions = all_attentions + (layer_outputs_attention,)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = ((layer_outputs_keys, layer_outputs_queries, layer_outputs_values),)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer output (keys, queries, values), (all hidden states), (all attentions)


class VLBertEmbeddings(mb.BertEmbeddings):
    def __init__(self, cfg, args, tok):
        super().__init__(cfg)
        self.pointing = args.pointing
        self.cfg = cfg
        self.tokenizer = tok
        self.args = args
        self.img_embeddings = resnet18(pretrained=args.pretrained_cnn)
        self.img_embeddings.fc = nn.Linear(512, cfg.hidden_size)
        self.img_position_embeddings = nn.Embedding(101, cfg.hidden_size // 4)  # relative bbox

    # noinspection PyMethodOverriding
    def forward(self, img_input, text_input_ids, img_position_ids, img_lens=None, txt_lens=None,
                **kwargs):

        device = img_input.device

        if self.pointing:
            content_embeddings = []
            position_embeddings = []
            seq_embeddings = []
            type_embeddings = []

            text_embeddings = self.word_embeddings(text_input_ids)

            if img_input.shape[1] > 0:
                img_embeddings = torch.cat(
                    [self.img_embeddings(img_input_col.squeeze(1)).unsqueeze(1) for img_input_col in
                     img_input.split(1, dim=1)],
                    dim=1)
                img_embeddings[img_input.sum(dim=(2, 3, 4)) == 0] = 0
                img_pos_embeddings = torch.cat(
                    [self.img_position_embeddings(img_pos_col.squeeze(1)).unsqueeze(1) for img_pos_col in
                     img_position_ids.split(1, dim=1)],
                    dim=1).reshape_as(img_embeddings)
            else:
                img_embeddings = []

            img_tok_emb, txt_tok_emb, sep_tok_emb = self.word_embeddings(torch.LongTensor(
                self.tokenizer.convert_tokens_to_ids(
                    [self.tokenizer.img_token, self.tokenizer.txt_token, self.tokenizer.sep_token])).to(
                device))

            for i, (img_len, txt_len) in enumerate(zip(img_lens, txt_lens)):
                content_embedding = [img_tok_emb]
                is_text = [True]
                seq_ids = [2]
                for j, (s, f) in enumerate(zip([0] + img_len.cumsum(0).tolist(), img_len.cumsum(0).tolist())):
                    if s == f:
                        continue
                    content_embedding.extend(img_embeddings[i][s:f])
                    content_embedding.append(sep_tok_emb)
                    is_text.extend([False for _ in range(s, f)])
                    is_text.append(True)
                    seq_ids.extend([j + 2 for _ in range(s, f + 1)])
                content_embedding.append(txt_tok_emb)
                is_text.append(True)
                seq_ids.append(2)
                for k, (s, f) in enumerate(zip([0] + txt_len.cumsum(0).tolist(), txt_len.cumsum(0).tolist())):
                    if s == f:
                        continue
                    content_embedding.extend(text_embeddings[i][s:f])
                    content_embedding.append(sep_tok_emb)
                    is_text.extend([True for _ in range(s, f + 1)])
                    seq_ids.extend([k + 2 for _ in range(s, f + 1)])
                content_embeddings.append(torch.stack(content_embedding))
                is_text = torch.BoolTensor(is_text).to(device)
                type_embeddings.append(self.token_type_embeddings(is_text.to(int)))
                seq_mat = torch.LongTensor(seq_ids).to(device)
                seq_mat[seq_mat >= self.cfg.type_vocab_size] = self.cfg.type_vocab_size - 1
                seq_embeddings.append(
                    self.token_type_embeddings(seq_mat))
                pos_mat = torch.arange(len(content_embedding), dtype=torch.long, device=device)
                pos_mat[pos_mat >= self.cfg.max_position_embeddings] = self.cfg.max_position_embeddings - 1
                position_embedding = self.position_embeddings(pos_mat)
                if img_input.shape[1] > 0:
                    position_embedding[~is_text] = img_pos_embeddings[i][:sum(img_len)]
                position_embeddings.append(position_embedding)

            content_embeddings = pad_sequence(content_embeddings, batch_first=True)
            position_embeddings = pad_sequence(position_embeddings, batch_first=True)
            type_embeddings = pad_sequence(type_embeddings, batch_first=True)
            seq_embeddings = pad_sequence(seq_embeddings, batch_first=True)

            embeddings = content_embeddings + position_embeddings + type_embeddings + seq_embeddings

            # collate into format [IMG] { ... [SEP] } [TXT] { ... [SEP] }
            # single sequence is [IMG] ... [SEP] [TXT] ... [SEP]

        else:
            # embed text
            seq_length = text_input_ids.size(1)

            text_position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            text_position_ids = text_position_ids.unsqueeze(0).expand_as(text_input_ids)

            text_embeddings = self.word_embeddings(text_input_ids)
            text_pos_embeddings = self.position_embeddings(text_position_ids)
            text_type_embeddings = self.token_type_embeddings(torch.ones_like(text_input_ids))

            text_embeddings = text_embeddings + text_pos_embeddings + text_type_embeddings
            bos_embeddings = text_embeddings[:, :1]  # [IMG]
            text_embeddings = text_embeddings[:, 1:]  # [TXT] ... text here ... [SEP]

            if img_input.shape[1] == 0:  # no images
                embeddings = torch.cat((bos_embeddings, text_embeddings), dim=1)
            else:
                img_embeddings = torch.cat(
                    [self.img_embeddings(img_input_col.squeeze(1)).unsqueeze(1) for img_input_col in
                     img_input.split(1, dim=1)],
                    dim=1)
                img_embeddings[img_input.sum(dim=(2, 3, 4)) == 0] = 0
                # masked or padding images should have zeroed out features
                img_pos_embeddings = torch.cat(
                    [self.img_position_embeddings(img_pos_col.squeeze(1)).unsqueeze(1) for img_pos_col in
                     img_position_ids.split(1, dim=1)],
                    dim=1).reshape_as(img_embeddings)
                img_type_embeddings = self.token_type_embeddings(
                    torch.zeros(img_input.shape[:2], dtype=torch.long, device=device))

                img_embeddings = img_embeddings + img_pos_embeddings + img_type_embeddings  # ... image regions here ...

                embeddings = torch.cat((bos_embeddings, img_embeddings, text_embeddings), dim=1)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VLBert(BertPreTrainedModel):
    MAX_VOCAB_SIZE = 32000

    def __init__(self, cfg, args, tok):
        super().__init__(cfg)
        self.tokenizer = tok
        self.embeddings = VLBertEmbeddings(cfg, args, tok)
        if args.input_pointing:
            self.encoder = BertEncoderExtended(cfg)
        else:
            self.encoder = mb.BertEncoder(cfg)
        self.pooler = mb.BertPooler(cfg)
        self.text_prediction = mb.BertLMPredictionHead(cfg)
        self.img_prediction = BertVMPredictionHead(cfg)
        self.cfg = cfg
        self.args = args
        try:
            self.apply(self.init_weights)
        except:
            self.init_weights()
        self.tie_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        with torch.no_grad():
            old_emb_wt = model.embeddings.word_embeddings.weight
            new_emb = nn.Embedding(cls.MAX_VOCAB_SIZE, model.config.hidden_size)
            new_emb.weight[:len(old_emb_wt)] = old_emb_wt
            model.embeddings.word_embeddings = new_emb
            new_dec = nn.Linear(model.config.hidden_size, cls.MAX_VOCAB_SIZE, bias=False)
            new_dec.weight[:len(old_emb_wt)] = old_emb_wt
            model.text_prediction.decoder = new_dec
            model.text_prediction.bias = nn.Parameter(torch.zeros(cls.MAX_VOCAB_SIZE))
            model.config.vocab_size = cls.MAX_VOCAB_SIZE
        model.tie_weights()
        return model

    def tie_weights(self):
        self._tie_or_clone_weights(self.text_prediction.decoder,
                                   self.embeddings.word_embeddings)

    def forward(self, img_input, text_input_ids, img_position_ids, attention_mask=None, **kwargs):
        """
        Returns:
        - `pooled_output`: tensor(B x C): general representation of all the sequences in the element
        - `text_predictions`: tensor(sum(txt_locs) x vocab_size). Predictions for all the positions where the input was
        a text. sum(txt_locs) is <= than B x T because it does not include the paddings (it would be sum(T_i), for the
        individual T_i before the collate)
        - `image_predictions`: tensor(sum(img_locs) x C). Similar to the text ones.

        The embedding_output and attention_mask create a structure like:
        [IMG]{img_k,0  img_k,1  ... img_k,n_k   [SEP]}(xK)[TXT]{txt_k,0  img_k,1  ... txt_k,n_k   [SEP]}(xK)

        `embedding_output`: tensor(B x K x C) has the format described above, where C is the size of the embedding. The
        embeddings here already have randomization where they need to. These embeddings also contain information about
        the sequence number they belong to, the position of each word of the text, and the type of input they represent
        (img/text). All the outputs of the model follow the same format and thus have length K.
        """
        embedding_output = self.embeddings(img_input, text_input_ids, img_position_ids, **kwargs)

        attention_mask = attention_mask.to(embedding_output.device) if attention_mask is not None else \
            torch.ones(embedding_output.shape[:2]).to(embedding_output.device)
        head_mask = [None] * self.cfg.num_hidden_layers

        # This is only necessary in the DataParallel
        embedding_output = F.pad(embedding_output, (0, 0, 0, attention_mask.shape[1] - embedding_output.shape[1]))
        extended_attention_mask = attention_mask

        while len(extended_attention_mask.shape) < 4:
            extended_attention_mask = extended_attention_mask.unsqueeze(1)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)  # seq_output, (hidden_states), (attentions)
        sequence_output = encoder_outputs[0]

        input_pointing_predictions = None
        if self.args.input_pointing:
            queries = sequence_output[1]
            sequence_output = sequence_output[2]  # for the predictions, use the values
            input_pointing_predictions = (queries, embedding_output)

        if self.args.pointing:
            text_seq_output = sequence_output[kwargs['txt_locs']]
            img_seq_output = sequence_output[kwargs['img_locs']]
        else:
            text_seq_output = torch.cat((sequence_output[:, :1], sequence_output[:, self.args.max_img_seq_len + 1:]),
                                        dim=1)
            img_seq_output = sequence_output[:, 1:self.args.max_img_seq_len + 1]

        pooled_output = self.pooler(sequence_output)  # this naive pooler only gets the first hidden vector

        # only decode tokens corresponding to the correct modality
        # decoders are local and do not attend to entire sequences
        # we can crop the sequence of hidden states before inputting
        # to save computation

        text_predictions = self.text_prediction(text_seq_output)
        image_predictions = self.img_prediction(img_seq_output) if img_input.shape[1] > 0 else None  # no images

        sequence_outputs = (sequence_output,)

        outputs = (text_predictions, image_predictions, input_pointing_predictions, *sequence_outputs, pooled_output,) + \
                  encoder_outputs[1:]
        # add hidden_states and attentions if they are here
        return outputs


def load_arch(path, args, fn_cfg='config.json', pretrained=False, tok=None):
    if pretrained:
        cfg = VLBertConfig.from_pretrained('bert-base-uncased')
        cfg_overwritten = VLBertConfig.from_json_file(os.path.join(path, fn_cfg))
        # we only keep the "output_attentions" from the cfg_overwritten
        cfg.output_attentions = cfg_overwritten.output_attentions
        model = VLBert.from_pretrained('bert-base-uncased', args, tok=tok, config=cfg)
    else:
        cfg = VLBertConfig.from_json_file(os.path.join(path, fn_cfg))
        # modify config to output attentions if we need it for testing
        with open(os.path.join('defaults', fn_cfg)) as f:
            output_attentions = json.load(f)['output_attentions']
        cfg.output_attentions = output_attentions
        model = VLBert(cfg, args, tok)
    return model


# fusedlayernorm does not work without keep_batchnorm_fp32=False, bad for stability
mb.BertLayerNorm = torch.nn.LayerNorm
