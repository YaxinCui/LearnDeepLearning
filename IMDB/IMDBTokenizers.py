# IMDB模型
# 为了统一模型接口，模型的输入都是batch 句子，也就是一个句子List，输出为各个句子为不同种类的概率
import numpy as np
from tqdm import tqdm

import re
from itertools import zip_longest


class IMDBTokenizer():
    
    def __init__(self, vocab_path, glove_path, word_dim=50, special_tokens=['<PAD>', '<UNK>']):
        
        self.special_tokens = special_tokens
        
        self.vocab2index = {special_token:i for i, special_token in enumerate(special_tokens)}
        self.index2vocab = {i:special_token for i, special_token in enumerate(special_tokens)}
        self.vocabList = list(special_tokens)
        self.vocabSet = set(self.vocabList)
        
        self.word_dim = word_dim
        
        wordEmbed, word2index = self.readGlove(glove_path, word_dim)
        self.buildVocabVector(vocab_path, wordEmbed, word2index, word_dim)
        
    def readGlove(self, glove_path, word_dim):
        word2index={}
        wordEmbed = None
        
        with open(glove_path, 'r', encoding='utf8') as f:
            
            gloveLines = f.readlines()
            wordEmbed = np.zeros(shape=(len(gloveLines), word_dim))
            
            for i, line in tqdm(enumerate(gloveLines)):
                line = line.strip().split()
                word2index[line[0]] = i
                
                wordEmbed[i]=line[1:]
                
        return wordEmbed, word2index
                
    def buildVocabVector(self, vocab_path, sourceVector, sourceVocab2index, vectorDim=300):
        vocabList = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                vocabList.append(line.strip())
                
        self.vocabList.extend(list( set(vocabList).intersection(set(sourceVocab2index.keys())) ))
        
        self.vocabVector = np.zeros(shape=(len(self.vocabList), vectorDim))
        # 对除了PAD以外的special_token随机初始化词向量
        
        for i in range(1, len(self.special_tokens)):
            self.vocabVector[i] = np.random.randn(vectorDim)
        
        for i in tqdm(range(len(self.special_tokens), len(self.vocabList))):
            
            vocab = self.vocabList[i]
            self.vocab2index[vocab] = i
            self.index2vocab[i] = vocab
            self.vocabVector[i] = sourceVector[sourceVocab2index[vocab]]
        
        
    def tokenizer(self, sentence, pad2maxlength=False, max_length=256):
        wordList = re.compile("[^a-z^A-Z^0-9^ ]").sub('', sentence).lower().strip().split()
        tokenList = []
        for word in wordList:
            if word in self.vocabSet:
                tokenList.append(self.vocab2index[word])
            else:
                tokenList.append(self.vocab2index['<UNK>'])
                
        if pad2maxlength:
            if len(sentence)>max_length:
                tokenList = tokenList[:max_length]
            else:
                tokenList.extend([0 for i in range(len(tokenList), max_length)])
                
        return tokenList
    
    def tokenizerBatch(self, batchSentenceList, batch_first=True, pad2maxlength=False, max_length=256):
        
        sentencesTokens = []
        for sentence in batchSentenceList:
            sentencesTokens.append(self.tokenizer(sentence, pad2maxlength, max_length))
            
        if not batch_first:
            sentencesTokens = zip_longest(*sentencesTokens, fillvalue=self.vocab2index['<PAD>'])
        
        return sentencesTokens
    
    def __call__(self, batchSentenceList, batch_first=True, pad2maxlength=False, max_length=256):

        return self.tokenizerBatch(batchSentenceList, batch_first=True, pad2maxlength=False, max_length=256)