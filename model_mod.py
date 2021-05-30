import torch
from torch import nn
from transformers import DistilBertModel


class DistilBertForSequenceClassification(nn.Module):
    '''def init_weights(self, module):

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)'''

    def __init__(self, config, num_labels=2):
        super(DistilBertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.config = config
        self.bert = DistilBertModel.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english',
            output_hidden_states=False
        )
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.dim, num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.classifier.bias is not None:
          self.classifier.bias.data.zero_()
        self.pre_classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.pre_classifier.bias is not None:
          self.pre_classifier.bias.data.zero_()        

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None,):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        distilbert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        '''last_hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )'''
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)

        '''pooled_output = torch.mean(last_hidden[0], dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)'''
        return logits