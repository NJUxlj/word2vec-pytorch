
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class SkipGramDataset(Dataset):
    def __init__(self, text, window_size=2, vocab_size=10000):
        self.text = text
        self.window_size = window_size
        self.vocab = self.build_vocab()
        self.data = self.generate_pairs()
        
    def build_vocab(self):
        word_counts = Counter(self.text)
        vocab = {'<PAD>':0, '<UNK>':1}
        for idx, (word, count) in enumerate(word_counts.most_common(10000), 2):
            if idx >= 10000: break
            vocab[word] = idx
        return vocab
    
    def generate_pairs(self):
        pairs = []
        for i, center_word in enumerate(self.text):
            context_start = max(0, i - self.window_size)
            context_end = min(len(self.text), i + self.window_size + 1)
            context_words = self.text[context_start:i] + self.text[i+1:context_end]
            
            for context_word in context_words:
                pairs.append((
                    self.vocab.get(center_word, 1), 
                    self.vocab.get(context_word, 1)
                ))
        return pairs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.LongTensor([self.data[idx][0]]), torch.LongTensor([self.data[idx][1]])

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out

# 使用示例
if __name__ == "__main__":
    # 示例输入数据
    corpus = ["the quick brown fox jumps over the lazy dog".split()]
    window_size = 2
    
    # 初始化数据集
    dataset = SkipGramDataset(corpus[0], window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 模型配置
    vocab_size = len(dataset.vocab)
    model = SkipGram(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(10):
        total_loss = 0
        for center, context in dataloader:
            optimizer.zero_grad()
            
            # 调整输入形状 [batch_size, 1] -> [batch_size]
            center = center.squeeze(1)
            context = context.squeeze(1)
            
            logits = model(center)
            loss = criterion(logits, context)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch} Loss: {total_loss/len(dataloader)}")
