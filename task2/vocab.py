from utils import read_corpus,pad_sents

from typing import List
from collections import Counter
from itertools import chain
import json
import torch

# typing 模块是用来类型检查 也可以作为开发文档附加说明 方便使用者调用时传入和返回参数类型
# collections 中的 counter 是 python 中自带的计数器
# itertools 使用 chain() 的一个常见场景 是当想对不同的集合中的所有元素执行某些操作的时候

class VocabEntry(object):
    def __init__(self, word2id = None) -> None:
        '''
        初始化 vocabEntry 
        param word2id : mapping word to indices
        '''
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<PAD>'] = 0
            self.word2id['<UNK>'] = 1
        self.unk_id = self.word2id['<UNK>']
        self.id2word = {v:k for k,v in self.word2id.items()}

    def __getitem__(self,word):
        '''
        获取 word 的 idx 
        '''
        return self.word2id.get(word,self.unk_id)
    
    def __contains__(self,word):
        return word in self.word2id

    def __setitem__(self,key,value):
        return ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % (len(self.word2id))

    def add(self,word):
        if word not in self.word2id:
            wid = self.word2id[word] = len(self.word2id)
            self.id2word[wid] = word
            return wid
        else:
            return self.word2id[word]

    def words2indices(self,sents):
        '''
        将 sents 转为 number index
        param sents: list(word) or list(list(word))
        '''
        if type(sents[0]) == list:
            return [[self.word2id.get(w,self.unk_id) for w in s] for s in sents]
        else:
            return [self.word2id.get(s,self.unk_id) for s in sents]

    def indices2words(self,idxs):
        return [self.id2word[id] for id in idxs]

    def to_input_tensor(self,sents:List[List[str]],device:torch.device):
        '''
        将原始句子 list 转为 tensor ， 同时将句子 PAD 成 max_len
        param sents: list or list<str>
        param device
        '''       
        sents = self.words2indices(sents)
        sents = pad_sents(sents,self.word2id['<PAD>'])
        sents_var = torch.tensor(sents, device = device)
        return sents_var

    @staticmethod
    # staticmethod 用于修饰类中的方法,使其可以在不创建类实例的情况下调用方法，这样做的好处是执行效率比较高。
    def from_corpus(corpus,size,min_feq = 3):
        '''
        从给定语料中创建VocabEntry
        '''     
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        # *parameter 是用来接受任意多个参数并将其放在一个 tuple 中
        # **parameter 是用来接受任意多个参数并将其放在一个 dict 中
        valid_words = word_freq.most_common(size - 2)
        # Counter.most_common() 找出序列中出现次数最多的元素
        valid_words = [word for word, value in valid_words if value >= min_feq]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
        .format(len(word_freq),min_feq,len(valid_words)))
        for word in valid_words:
            vocab_entry.add(word)

        return vocab_entry

class Vocab(object):
    '''
    src 、 tgt 的词汇类
    '''
    def __init__(self, src_vocab: VocabEntry, labels: dict) -> None:
        self.vocab = src_vocab
        self.labels = labels

    @staticmethod
    def build(src_sents, labels, vocab_size, min_feq):
        
        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents,vocab_size,min_feq)
        
        return Vocab(src,labels)

    def save(self,file_path):
        with open(file_path,'w') as fint:
            json.dump(dict(src_word2id = self.vocab.word2id,labels = self.labels),fint,indent = 2)
        # dump() 它将 Python 对象转换为适当的 json 对象

    @staticmethod
    def load(file_path):
        with open(file_path,'r') as fout:
            entry = json.load(fout)
        src_word2id = entry['src_word2id']
        labels = entry['labels']

        return Vocab(VocabEntry(src_word2id),labels)

    def __repr__(self) -> str:
        '''
        Representation of Vocab to be used
        when printing the object.
        '''   
        return 'Vocab(source %d words)' % (len(self.vocab))

if __name__ == '__main__':
    src_sents,labels = read_corpus('task2\data\cnews\cnews.train.txt')
    labels = {label:idx for idx,label in enumerate(labels)}

    vocab = Vocab.build(src_sents,labels,50000,3)
    print('generated vocabulary, source %d words' % (len(vocab.vocab)))
    vocab.save('./vocab.json')
    
        