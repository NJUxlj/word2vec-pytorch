
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np

class NegativeSamplingLoss(nn.Module):
    def __init__(self, vocab_size, embed_dim, noise_dist=None, num_neg_samples=5):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        self.num_neg_samples = num_neg_samples
        self.noise_dist = noise_dist
        
        # 初始化参数
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward(self, target, context):
        batch_size = target.size(0)
        
        # 正样本得分
        target_embeds = self.in_embed(target)       # [batch, embed_dim]
        context_embeds = self.out_embed(context)    # [batch, embed_dim]
        pos_score = torch.mul(target_embeds, context_embeds).sum(dim=1)  # [batch]
        pos_loss = torch.log(torch.sigmoid(pos_score)).mean()  # 最大化log概率
        
        # 负采样
        if self.noise_dist is None:
            neg_samples = torch.randint(0, len(self.noise_dist), 
                                      (batch_size*self.num_neg_samples,))
        else:
            neg_samples = torch.multinomial(self.noise_dist, 
                                          batch_size*self.num_neg_samples, 
                                          replacement=True)
        
        # 负样本得分  
        neg_embeds = self.out_embed(neg_samples.to(target.device))
        neg_score = torch.bmm(neg_embeds.view(batch_size, self.num_neg_samples, -1),
                            target_embeds.unsqueeze(2)).squeeze()  # [batch, neg_samples]
        neg_loss = torch.log(torch.sigmoid(-neg_score)).mean()  # 最小化负样本概率
        
        return -(pos_loss + neg_loss)

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim, noise_dist=None, num_neg_samples=5):
        super().__init__()
        self.loss_fn = NegativeSamplingLoss(vocab_size, embed_dim, 
                                          noise_dist, num_neg_samples)
        
    def forward(self, target, context):
        return self.loss_fn(target, context)
    
    def get_embeddings(self):
        return self.loss_fn.in_embed.weight.detach().cpu().numpy()

def build_noise_dist(vocab, counts, power=0.75):
    """构建负采样分布"""
    word_freqs = np.array([counts[w] for w in vocab])
    probs = word_freqs ** power
    probs /= probs.sum()
    return torch.from_numpy(probs.astype(np.float32))

# 示例使用流程
if __name__ == "__main__":
    # 假设已构建的词汇表和预处理数据
    vocab = ["apple", "banana", "fruit", "juice", "market"]
    counts = Counter({"apple": 5, "banana":3, "fruit":8, "juice":2, "market":4})
    noise_dist = build_noise_dist(vocab, counts)
    
    # 初始化模型
    embed_dim = 128
    model = Word2Vec(len(vocab), embed_dim, noise_dist)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 示例训练批次
    target_batch = torch.tensor([0, 2])  # "apple", "fruit"
    context_batch = torch.tensor([2, 3]) # "fruit", "juice"
    
    # 训练步骤
    model.train()
    optimizer.zero_grad()
    loss = model(target_batch, context_batch)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print("Embeddings shape:", model.get_embeddings().shape)
