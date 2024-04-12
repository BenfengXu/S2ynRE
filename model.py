from transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, MarginRankingLoss
import json
from pathlib import Path
import torch.nn.functional as F

class BertForRelationExtraction(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_ft = nn.Linear(2 * config.hidden_size, num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        pseudo_labels=None,
        is_synsetic=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sub_start_marker_pos=None,
        obj_start_marker_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sub_start_marker_output = outputs[0][range(sub_start_marker_pos.shape[0]), sub_start_marker_pos, :]
        obj_start_marker_output = outputs[0][range(obj_start_marker_pos.shape[0]), obj_start_marker_pos, :]
        ent_pair_rep = torch.cat((sub_start_marker_output, obj_start_marker_output), dim=1)
        ent_pair_rep = self.dropout(ent_pair_rep)

        logits = self.classifier_ft(ent_pair_rep)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if pseudo_labels is not None:
                pseudo_labels = F.softmax(pseudo_labels, dim=-1)
                loss_distill = - pseudo_labels * F.log_softmax(logits, dim=-1)[:, None, :].repeat([1, pseudo_labels.shape[1], 1])
                loss_distill = loss_distill.sum(dim=-1).mean(dim=-1)
                loss = loss * (1 - is_synsetic) + loss_distill * is_synsetic

            loss = loss.mean()
            # ranking loss
            # loss_fct = MarginRankingLoss()
            # bs = labels.shape[0]
            # positives = logits[range(bs), labels].unsqueeze(1).repeat(1, self.num_labels)
            # targets = torch.ones(positives.view(-1).size()).to(logits.device)
            # loss = loss_fct(positives.view(-1), logits.view(-1), targets) * self.num_labels / (self.num_labels - 1)

        return loss, logits


class BertForSyntheticPretrain(BertPreTrainedModel):
    def __init__(self, config, num_labels, temp):
        super().__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_distant_distill = nn.Linear(2 * config.hidden_size, num_labels)
        self.temp = temp

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_hard=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sub_start_marker_pos=None,
        obj_start_marker_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sub_start_marker_output = outputs[0][range(sub_start_marker_pos.shape[0]), sub_start_marker_pos, :]
        obj_start_marker_output = outputs[0][range(obj_start_marker_pos.shape[0]), obj_start_marker_pos, :]
        ent_pair_rep = torch.cat((sub_start_marker_output, obj_start_marker_output), dim=1)
        ent_pair_rep = self.dropout(ent_pair_rep)

        logits = self.classifier_distant_distill(ent_pair_rep)

        loss = None
        lambda_ = 0
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_hard = loss_fct(logits.view(-1, self.num_labels), labels_hard.view(-1))

            labels = F.softmax(labels / self.temp, dim=-1)
            # Multi-Teacher
            loss_distill = - labels * F.log_softmax(logits / self.temp, dim=-1)[:, None, :].repeat([1, labels.shape[1], 1])
            # loss_distill = - labels * F.log_softmax(logits / self.temp, dim=-1)
            loss_distill = loss_distill.sum(dim=-1).mean()

            # loss = loss_hard + lambda_ * loss_distill
            loss = loss_distill

        return loss, logits