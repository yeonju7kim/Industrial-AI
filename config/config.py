import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

labeled_txt_file = 'data/1. 실습용자료.txt'
unlabeled_txt_file = 'data/2. 모델개발용자료.txt'

pkl_tag = 'tokenizer_kobert_embedding_kobert'
category_dataset_pkl_file = f'assets/tokenized_data/CategoryDataloader-{pkl_tag}.pkl'
category_file = 'data/한국표준산업분류(10차)_국문.xlsx'
max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 4
max_grad_norm = 1
log_interval = 200
learning_rate = 4e-5
train_portion = 0.7

