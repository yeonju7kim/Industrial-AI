import os.path


import pickle
from torch.utils.data import DataLoader, Dataset

from config.config import *
from data_mining.category_manager import *
from tqdm import tqdm

from data_mining.transform import get_bert

'''
데이터는 아래와 같이 구성되었다.
1. 대분류
2. 중분류
3. 소분류
4. 사업 대상, 단어1로 부르겠다.
5. 사업 방법, 단어2
6. 사업 취급 품목, 단어3
'''

class CategoryDataset(Dataset):
    def __init__(self, sentence_list, label_list, transform, category_manager):
        ''' Category Dataset

        :param sentence_list: 문장을 담은 list
        :param label_list: 문장의 소분류 label을 담은 list
        :param transform: tokenize하고 embedding하는 transform
        :param category_manager: 카테고리의 정보를 담은 category_manager
        '''
        self.sentence_list = []
        self.label_list = []
        for sentence in tqdm(sentence_list):
            self.sentence_list.append(transform([sentence]))
        for label in tqdm(label_list):
            if label != '':
                self.label_list.append(category_manager.get_one_hot_by_code('%03d' % int(label)))
            else:
                self.label_list.append('-1')

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        try :
            return self.sentence_list[idx], torch.FloatTensor(self.label_list[idx])
        except:
            return self.sentence_list[idx], None

    @staticmethod
    def newCategoryDataset(sentence_list, label_list, transform, category_manager):
        ''' 새로운 CategoryDataset을 만드는 함수

        :param sentence_list: 전처리를 거친 문장의 리스트
        :param label_list: 라벨의 리스트
        :param transform: tokenize -> embedding을 하는 transform
        :param category_manager: 카테고리 정보를 담는 category manager
        :return: category dataset
        '''

        category_dataset = CategoryDataset(sentence_list, label_list, transform, category_manager)

        return category_dataset

class INDEX:
    '''
    한국표준산업분류 xlsx 파일에 있는 column idx
    '''
    ID_IDX = 0 # data id의 column
    BIG_IDX = 1 # 대분류의 column
    MID_IDX = 2 # 중분류의 column
    SMALL_IDX = 3 # 소분류의 column
    TEXT_OBJ_IDX = 4 # 사업 대상, 단어1
    TEXT_MTHD_IDX = 5 # 사업 방법, 단어2
    TEXT_DEAL_IDX = 6 # 사업 취급 품목, 단어3

def rawdata_to_sentence(rawdata):
    ''' raw 문장으로 전처리 한 문장 만드는 것

    :param rawdata: txt파일에서 읽어온 그대로의 문장
    :return: 단어1 + 단어2 + 단어3로 만든 문장
    '''
    words = rawdata.split('|')
    return words[INDEX.TEXT_OBJ_IDX] + ' ' + words[INDEX.TEXT_MTHD_IDX] + ' ' + words[INDEX.TEXT_DEAL_IDX], words[INDEX.SMALL_IDX]

def read_txt_file(filename):
    ''' txt 파일을 읽어서 단어1 + 단어2 + 단어3로 문장으로 만들고, 소분류를 label로 만드는 함수

    :param filename: txt 파일
    :return: 문장 리스트, 라벨 리스트
    '''
    sentence_list = []
    label_list = []
    lines = read_raw_txt_file(filename)

    for line in lines:
        sentence, label = rawdata_to_sentence(line)
        sentence_list.append(sentence)
        label_list.append(label)

    return sentence_list, label_list

def read_raw_txt_file(filename):
    ''' txt 파일의 각 줄의 문장을 list로 표현한 것

    :param filename: txt 파일
    :return: 모든 줄의 raw 문장 list
    '''
    f = open(filename)
    f.readline()
    lines = f.readlines()

    return lines

def get_category_dataset(filename, transform, category_manager):
    sentence_list, label_list = read_txt_file(filename)
    category_dataset = CategoryDataset.newCategoryDataset(sentence_list, label_list, transform, category_manager)
    return category_dataset

def get_category_dataloader(category_dataset, batch_size, train_portion, shuffle=True):
    ''' category dataloader 얻는 함수

    :param category_dataset: dataset
    :param batch_size: batch size
    :param train_portion: trainset의 비율 (0~1)
    :param shuffle: shuffle 유무
    :return: trainDataLoader, validDataLoader
    '''

    dataset_size = len(category_dataset)
    train_size = (int)(train_portion * dataset_size)
    train_set, val_set = torch.utils.data.random_split(category_dataset, [train_size, dataset_size - train_size])

    trainDataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle) if len(train_set) != 0 else None
    validDataLoader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle) if len(val_set) != 0 else None

    return trainDataLoader, validDataLoader

def fill_answer_sentence(sentence, big_category, mid_category, small_category):
    ''' 빈칸에 추론한 답을 채워서 문장을 만드는 함수

    :param sentence: 빈칸으로 된 문장
    :param big_category: 추론한 대분류
    :param mid_category: 추론한 중분류
    :param small_category: 추론한 소분류
    :return: 추론한 값들로 다시 구성한 문장
    '''
    sentence_list = sentence.split('|')
    sentence_list.insert(1, small_category)
    sentence_list.insert(1, mid_category)
    sentence_list.insert(1, big_category)
    sentence.join('|')
    return sentence

def save_dataset_pickle(tokenizer_name, embedding_name, dataset):
    category_dataset_pkl_file = f'assets/tokenized_data/CategoryDataset-{tokenizer_name}-{embedding_name}.pkl'
    _save_pickle(category_dataset_pkl_file, dataset)

def get_saved_dataset(tokenizer_name, embedding_name, dataset_name):
    category_dataset_pkl_file = f'assets/tokenized_data/CategoryDataset-{tokenizer_name}-{embedding_name}-{dataset_name}.pkl'
    return _get_pickle(category_dataset_pkl_file)

def _save_pickle(filename, object):
    if os.path.exists('assets/tokenized_data') == False:
        os.mkdir('assets/tokenized_data')
    with open(filename, 'wb') as f:
        pickle.dump(object, f)

def _get_pickle(pkl_file):
    try:
        with open(pkl_file, 'rb') as f:
            pickle_file = pickle.load(f)
            return pickle_file
    except:
        raise AssertionError

def unit_test():
    bert, transform = get_bert(max_len)
    category_manager = CategoryManager.new_category_manager(category_file)
    dataset = get_category_dataset(filename=labeled_txt_file, transform= transform, category_manager=category_manager)
    train_dataloader, valid_dataloader = get_category_dataloader(dataset, batch_size, train_portion)
    for batch_id, ((token_ids, valid_length, segment_ids), label) in enumerate(train_dataloader):
        print(f'token_ids : {token_ids}')
        print(f'valid_length : {valid_length}')
        print(f'segment_ids : {segment_ids}')
        print(f'label : {label}')
        break
    save_dataset_pickle(tokenizer_name, embedding_name, dataset)
    new_dataset = get_saved_dataset(tokenizer_name, embedding_name, 'train')
    for old_train_data, new_train_data in zip(dataset, new_dataset):
        if old_train_data != new_train_data:
            print('Reloaded data is changed. Something is wrong.')


if __name__ == '__main__':
    unit_test()