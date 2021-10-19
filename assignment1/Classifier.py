import torch.nn as nn
from transformers import DistilBertModel

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes = 1, pre_trained_model_name = "distilbert-base-uncased"):
    super(SentimentClassifier, self).__init__()
    self.distilbert = DistilBertModel.from_pretrained(pre_trained_model_name, return_dict=False)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.distilbert.config.hidden_size, n_classes)


  def forward(self, input_ids, attention_mask):
    a = self.distilbert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    #_, pooled_output = a
    pooled_output = a[0]
    output = self.drop(pooled_output)
    res = self.out(output)
    return res
