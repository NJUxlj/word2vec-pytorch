
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from tqdm import tqdm

class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bias = nn.Parameter(torch.randn(vocab_size))
        self.context_bias = nn.Parameter(torch.randn(vocab_size))
        
        # 初始化参数
        nn.init.uniform_(self.embedding.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.uniform_(self.context_embedding.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.constant_(self.bias, 0.0)
        nn.init.constant_(self.context_bias, 0.0)
    
    def forward(self, i, j):
        # i: 中心词索引, j: 上下文词索引
        w_i = self.embedding(i)
        w_j = self.context_embedding(j)
        b_i = self.bias[i]
        b_j = self.context_bias[j]
        return torch.sum(w_i * w_j, dim=1) + b_i + b_j

class GloVeDataset(torch.utils.data.Dataset):
    def __init__(self, cooccurrence, x_max=100, alpha=0.75):
        self.cooccurrence = cooccurrence
        self.x_max = x_max
        self.alpha = alpha
    
    def __len__(self):
        return len(self.cooccurrence)
    
    def __getitem__(self, idx):
        i, j, x_ij = self.cooccurrence[idx]
        weight = (x_ij / self.x_max) ** self.alpha if x_ij < self.x_max else 1.0
        return torch.LongTensor([i]), torch.LongTensor([j]), torch.FloatTensor([x_ij]), torch.FloatTensor([weight])

def build_cooccurrence(corpus, window_size=5):
    cooccur = defaultdict(float)
    vocab = list(set(corpus))
    word2idx = {w: i for i, w in enumerate(vocab)}
    
    for center_pos in range(len(corpus)):
        center_word = corpus[center_pos]
        for offset in range(1, window_size+1):
            left_pos = center_pos - offset
            right_pos = center_pos + offset
            
            for pos in [left_pos, right_pos]:
                if 0 <= pos < len(corpus):
                    context_word = corpus[pos]
                    cooccur[(word2idx[center_word], word2idx[context_word])] += 1.0 / offset
    
    return [(i, j, x) for (i, j), x in cooccur.items()], word2idx

# 示例用法
def train():
    # 1. 准备样例语料
    corpus = ["natural", "language", "processing", "with", "pytorch"]
    
    # 2. 构建共现矩阵（内存优化版）
    cooccurrence, word2idx = build_cooccurrence(corpus, window_size=3)
    
    # 3. 初始化模型
    vocab_size = len(word2idx)
    embedding_dim = 50
    model = GloVe(vocab_size, embedding_dim)
    
    # 4. 配置训练参数
    dataset = GloVeDataset(cooccurrence)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adagrad(model.parameters(), lr=0.05)
    epochs = 100
    
    # 5. 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for i, j, x_ij, weights in tqdm(dataloader):
            optimizer.zero_grad()
            
            outputs = model(i.squeeze(), j.squeeze())
            log_x_ij = torch.log(x_ij.squeeze())
            
            loss = torch.mean(weights * (outputs - log_x_ij) ** 2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    # 6. 合并向量（最佳实践）
    embeddings = model.embedding.weight.data + model.context_embedding.weight.data






if __name__ == "__main__":
    train()
    print("Done")