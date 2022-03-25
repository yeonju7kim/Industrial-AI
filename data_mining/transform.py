import gluonnlp as nlp
from data_mining.kobert.pytorch_kobert import get_pytorch_kobert_model
from data_mining.kobert.utils import get_tokenizer
from config.config import *
from konlpy.tag import Kkma, Komoran, Okt, Mecab
import sys

def get_bert(max_len):
    ''' pretrained된 bert model, tokenize -> embedding 하는 transform

    :param max_len: (int) 문장의 최대 길이
    :return: pretrained된 bert model, transform
    '''
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    bert_tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    print(bert_tokenizer('저는 학생입니다.'))
    transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=True, pair=False)
    # BERTSentenceTransform : 토큰화와 패딩
    return bertmodel, transform

class Tokenizer():
    def __init__(self, tokenizer_name, vocab):
        self.name = tokenizer_name
        self._set_tokenizer()

    def _set_tokenizer(self, vocab):
        if self.name == TOKENIZER.KOBERT:
            tokenizer = get_tokenizer()
            self.tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        elif self.name == TOKENIZER.KKMA:
            self.tokenizer = Kkma()
        elif self.name == TOKENIZER.KOMARAN:
            self.tokenizer = Komoran()
        elif self.name == TOKENIZER.OKT:
            self.tokenizer = Okt()
        elif self.name == TOKENIZER.MECAB:
            self.tokenizer = Mecab()

    def __call__(self, lines):
        if self.name == TOKENIZER.KOBERT:
            return self.tokenizer(lines)
        elif self.name == TOKENIZER.KKMA or tokenizer_name == TOKENIZER.KOMARAN or tokenizer_name == TOKENIZER.OKT:
            return self.tokenizer.pos(lines, flatten=False, join=True)
        elif self.name == TOKENIZER.MECAB:
            return self.tokenizer.pos(lines, norm=True, stem=True, join=True)


def unit_test():
    txt_list = '저는 학생입니다.'
    token = Tokenizer(TOKENIZER.KOBERT)
    print(token(txt_list))
    token = Tokenizer(TOKENIZER.KKMA)
    print(token(txt_list))

if __name__ == '__main__':
    unit_test()