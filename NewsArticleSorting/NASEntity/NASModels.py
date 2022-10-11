from  transformers import AutoModel
import torch.nn as nn


class NASModel(nn.Module):

    def __init__(self, model_name):
        super(NASModel, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        _, o2 = self.bert(
            input_ids,
            attention_mask=attention_mask
        )

        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
