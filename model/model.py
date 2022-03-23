import os

import torch
from torch import Module
from tqdm import tqdm


class Model(Module.nn):
    def __init__(self, class_num, device, load_pretrained, load_model_path=):
        if load_model_path != "":
            self.load_state_dict(load_model(load_model_path))
        elif load_pretrained == True:
            self = self.get_pretrained(class_num=class_num)
        else:
            self = self.get_init_model(class_num=class_num)
        self.class_num = class_num
        self.device = device

    def forward(self, token_ids, valid_length, segment_ids):
        return super().forward(token_ids, valid_length, segment_ids)

    def train(self, epoch, train_dataloader, valid_dataloader, optimizer, scheduler=None):
        self.zero_grad()
        for e in range(0, epoch):
            self.train()
            total_loss, total_accuracy = 0, 0
            total_true_positive = 0
            total_data_number = 0
            for idx, batch in enumerate(train_dataloader):
                batch = tuple(index.to(self.device) for index in batch)
                ids, masks, labels, = batch
                outputs = self(ids, token_type_ids=None, attention_mask=masks, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
                true = [label for label in labels.cpu().numpy()]
                true_positive, batch_size = count_true_positive(true, pred)
                total_true_positive = total_true_positive + true_positive
                total_data_number = total_data_number + batch_size
                total_accuracy = total_data_number / total_true_positive
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                self.zero_grad()
                avg_loss = total_loss / len(train_dataloader)
                avg_accuracy = total_accuracy / len(train_dataloader)
                print(f" {epoch + 1} Epoch Average train loss : {avg_loss}")
                print(f" {epoch + 1} Epoch Average train accuracy : {avg_accuracy}")
            acc = self.test(valid_dataloader)
            os.makedirs("results", exist_ok=True)
            f = os.path.join("results", f'epoch_{epoch + 1}_evalAcc_{acc * 100:.0f}.pth')
            print('Saved checkpoint:', f)

    def test(self, test_dataloader):
        self.eval()
        total_accuracy = 0
        for batch in test_dataloader:
            batch = tuple(index.to(self.device) for index in batch)
            ids, masks, labels = batch
            with torch.no_grad():
                outputs = self(ids, token_type_ids=None, attention_mask=masks)
            pred = [torch.argmax(logit).cpu().detach().item() for logit in outputs.logits]
            true = [label for label in labels.cpu().numpy()]
            accuracy = accuracy_score(true, pred)
            total_accuracy += accuracy
            avg_accuracy = total_accuracy / len(test_dataloader)
            print(f"test AVG accuracy : {avg_accuracy: .2f}")
        return avg_accuracy

    def inference(self, token_id, valid_length, segment_id):
        self.eval()
        out = self(token_id, valid_length, segment_id)
        return torch.argmax(out)

    def inference_by_dataloader(self, test_dataloader):
        out_list = []
        for batch_id, ((token_ids, valid_length, segment_ids), label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length = valid_length
            out = self(token_ids, valid_length, segment_ids)
            out_list.append(torch.argmax(out).data)
        return out_list

def load_model(model_file):
    torch.load(model_file)
    return

def gen_attention_mask(token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
        attention_mask[i][:v] = 1
    return attention_mask.float()

def save_model(model, epoch, train_acc, test_acc):
    if os.path.exists("checkpoint") == False:
        os.mkdir("checkpoint")
    torch.save(model.state_dict(), 'checkpoint/model_epoch_{:.3f}_train_acc_{:.3f}_test_acc_{:.3f}.pth'.format(epoch, train_acc, test_acc))

def accuracy_score(predict_list, label_list):
    true_positive_count, batch_size = count_true_positive(predict_list, label_list)
    return true_positive_count / batch_size

def count_true_positive(predict_list, label_list):
    ''' count true positive per batch

    :param predict_list: predict list per 1 batch
    :param label_list: true label list per 1 batch
    :return:
    '''
    max_predict_value_list, max_predict_indices = torch.max(predict_list, 1)
    max_label_value_list, max_label_indices = torch.max(label_list, 1)
    true_positive_count = 0
    for pred_idx, label_idx in zip(max_predict_indices, max_label_indices):
        if pred_idx == label_idx:
            true_positive_count = true_positive_count + 1
    return true_positive_count, max_predict_indices.size()[0]