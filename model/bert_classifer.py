import os

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from model.model import Model


class BertClassifier(BertForSequenceClassification, Model):
    def __init__(self, class_num, device, load_pretrained=True, load_model_path=""):
        super().__init__(class_num, device, load_pretrained=True, load_model_path="")

    def forward(self, token_ids, valid_length, segment_ids):
        return super().forward(token_ids, valid_length, segment_ids)

    def train(self, epoch, train_dataloader, valid_dataloader, optimizer, scheduler=None):
        super().train(epoch, train_dataloader, valid_dataloader, optimizer, scheduler)

    def test(self, test_dataloader):
        return super().test(test_dataloader)

    def inference(self, token_id, valid_length, segment_id):
        return super().inference(token_id, valid_length, segment_id)

    def inference_by_dataloader(self, test_dataloader):
        return super().inference_by_dataloader(test_dataloader)

    def get_pretrained(self, class_num):
        return BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=class_num)

    def get_init_model(self, class_num):
        return BertForSequenceClassification(num_labels=class_num)