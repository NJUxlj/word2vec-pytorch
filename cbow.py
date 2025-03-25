
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

class CBOWDataset(Dataset):
    def __init__(self, corpus, window_size=2):
        self.data = []
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = self.build_vocab(corpus)
        
        for sentence in corpus:
            indices = [self.word2idx[word] for word in sentence]
            for i in range(window_size, len(indices)-window_size):
                context = indices[i-window_size:i] + indices[i+1:i+window_size+1]
                target = indices[i]
                self.data.append((context, target))

    def build_vocab(self, corpus, min_freq=2):
        counter = Counter()
        for sentence in corpus:
            counter.update(sentence)
        
        vocab = [word for word, freq in counter.items() if freq >= min_freq]
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        return vocab

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        avg_embeds = torch.mean(embeds, dim=1)  # 上下文嵌入平均 
        out = self.linear(avg_embeds)   # 投影到词汇表空间 
        return out




def train():
    # 使用示例
    corpus = [
        ["deep", "learning", "transforms", "nlp"],
        ["pytorch", "is", "a", "great", "framework"],
        ["word", "embeddings", "capture", "semantic", "meaning"]
    ]

    # 超参数
    EMBED_DIM = 100
    BATCH_SIZE = 2
    WINDOW_SIZE = 2
    EPOCHS = 100

    dataset = CBOWDataset(corpus, WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CBOW(len(dataset.vocab), EMBED_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(EPOCHS):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # 获取词向量
    word_vectors = model.embeddings.weight.data
    print(f"Word vector for 'nlp': {word_vectors[dataset.word2idx['nlp']]}")



def test():
    
    # 使用示例
    corpus = [
        ["deep", "learning", "transforms", "nlp"],
        ["pytorch", "is", "a", "great", "framework"],
        ["word", "embeddings", "capture", "semantic", "meaning"]
    ]

    # 超参数
    EMBED_DIM = 100
    BATCH_SIZE = 2
    WINDOW_SIZE = 2
    EPOCHS = 100

    dataset = CBOWDataset(corpus, WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CBOW(len(dataset.vocab), EMBED_DIM)
    
    test_sentence = ["this", "is", "a", "test"]  
    test_context = [dataset.word2idx[w] for w in test_sentence[1:3]]  
    test_input = torch.tensor([test_context])  
    pred = model(test_input)  
    print(f"Predicted word index: {torch.argmax(pred)}")  

if __name__ == '__main__':
    train()