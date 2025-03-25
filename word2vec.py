
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

class Word2VecDataset(Dataset):
    def __init__(self, text, window_size=2, vocab_size=10000):
        # 最佳实践：采用子采样技术处理高频词
        word_counts = Counter(text)
        total_words = len(text)
        self.text = [word for word in text if (np.sqrt(word_counts[word]/ (0.001 * total_words)) + 1) * (0.001 * total_words) / word_counts[word] > 0.4]
        
        self.vocab = self._build_vocab(vocab_size)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        self.data = self._generate_samples(window_size)
    
    def _build_vocab(self, vocab_size):
        # 处理稀有词的最佳实践：过滤出现次数少于5次的词
        word_counts = Counter(self.text)
        filtered_words = [word for word, count in word_counts.items() if count >= 5]
        return [word for word, _ in Counter(filtered_words).most_common(vocab_size)]

    def _generate_samples(self, window_size):
        data = []
        for i, target in enumerate(self.text):
            if target not in self.word_to_idx:
                continue
            context_indices = [
                i + j for j in range(-window_size, window_size + 1)
                if j != 0 and 0 <= i + j < len(self.text)
            ]
            for j in context_indices:
                context_word = self.text[j]
                if context_word in self.word_to_idx:
                    data.append((self.word_to_idx[target], 
                               self.word_to_idx[context_word]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target, context = self.data[idx]
        # 最佳实践：使用PyTorch LongTensor
        return torch.LongTensor([target]), torch.LongTensor([context])

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        # 最佳实践：同时使用输入和输出嵌入矩阵
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(vocab_size, embedding_dim)
        # 最佳实践：初始化嵌入权重
        self._init_embeddings()
    
    def _init_embeddings(self):
        init_range = 0.5 / self.input_embedding.embedding_dim
        nn.init.uniform_(self.input_embedding.weight, -init_range, init_range)
        nn.init.constant_(self.output_embedding.weight, 0)
    
    def forward(self, target, context, negative_samples=None):
        # 正样本计算
        target_embed = self.input_embedding(target)
        context_embed = self.output_embedding(context)
        positive_score = torch.matmul(target_embed, context_embed.transpose(1,2)).squeeze()
        
        # 负采样计算（最佳实践：动态生成负样本）
        if negative_samples is not None:
            negative_embed = self.output_embedding(negative_samples)
            negative_score = torch.bmm(negative_embed, 
                                      target_embed.unsqueeze(2)).squeeze().mean(dim=1)
            return positive_score, negative_score
        return positive_score

def negative_sampling(batch_size, vocab_size, num_neg_samples=5):
    # 最佳实践：遵循原论文的噪声分布(P(w)^0.75)
    return torch.randint(0, vocab_size, (batch_size, num_neg_samples))



def train():
    
    # 参数配置（参考Google News Word2Vec参数）
    EMBEDDING_DIM = 300
    BATCH_SIZE = 32
    EPOCHS = 5
    WINDOW_SIZE = 5
    
    # 示例语料（替换为实际数据）
    corpus = ["natural language processing".split()]*1000 + \
             ["computer vision".split()]*800 + \
             ["machine learning".split()]*1200
    
    # 准备数据集
    flat_corpus = [word for sent in corpus for word in sent]
    dataset = Word2VecDataset(flat_corpus, window_size=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 模型初始化
    model = Word2Vec(len(dataset.vocab), EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环（最佳实践：异步数据加载）
    for epoch in range(EPOCHS):
        total_loss = 0
        for target, context in dataloader:
            # 生成负样本（动态生成节省内存）
            print("negative sampling ~~~")
            neg_samples = negative_sampling(target.size(0), len(dataset.vocab))
            
            # 前向传播
            pos_score, neg_score = model(target, context, neg_samples)
            
            # 损失计算（参考原论文的负采样目标）
            loss = -(torch.log(torch.sigmoid(pos_score)).mean() + 
                    torch.log(torch.sigmoid(-neg_score)).mean())
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # 保存词向量（最佳实践：保存输入+输出向量的平均值）
    embeddings = (model.input_embedding.weight.data + 
                 model.output_embedding.weight.data) / 2
    torch.save(embeddings, "word_vectors.pt")







if __name__ == "__main__":
    train()