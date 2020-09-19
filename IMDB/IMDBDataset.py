from torch.utils.data import Dataset, DataLoader
import os
# 读取数据

class IMDBDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.negPath = os.path.join(file_path, 'neg')
        self.posPath = os.path.join(file_path, 'pos')
        
        self.negDirList = os.listdir(self.negPath)
        self.posDirList = os.listdir(self.posPath)

        self.lenNeg = len(self.negDirList)
        self.lenPos = len(self.posDirList)
        
        self.length = self.lenNeg + self.lenPos
        self.label2index = {'neg':0, 'pos':1}
        
        print("共有pos数据 {} 条, neg数据 {} 条，共 {} 条".format(self.lenPos, self.lenNeg, self.length))
        
    def __getitem__(self, index):
        if index < self.lenNeg:
            with open(os.path.join(self.negPath, self.negDirList[index]), 'r') as f:
                sentence = f.read()
            return (sentence, self.label2index['neg'])
        
        else:
            with open(os.path.join(self.posPath, self.posDirList[index-self.lenNeg]), 'r') as f:
                sentence = f.read()
            return (sentence, self.label2index['pos'])
        
    def __len__(self):
        return self.length