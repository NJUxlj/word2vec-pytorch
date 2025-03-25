
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, word=None, freq=0):
        self.word = word    # 叶子节点存储词语
        self.freq = freq    # 词频
        self.left = None
        self.right = None
        self.code = []      # Huffman编码（0/1序列）
        self.point = []     # 路径节点索引

    # 用于堆排序
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanTree:
    def __init__(self, word_freq):
        """ 构建Huffman树的最佳实践 """
        self.nodes = []
        self.word2node = {}
        
        # 初始化叶子节点
        for word, freq in word_freq.items():
            node = HuffmanNode(word, freq)
            heapq.heappush(self.nodes, node)
            self.word2node[word] = node
        
        # 构建内部节点（论文级优化：最小堆方法）
        while len(self.nodes) > 1:
            left = heapq.heappop(self.nodes)
            right = heapq.heappop(self.nodes)
            parent = HuffmanNode(freq=left.freq + right.freq)
            parent.left, parent.right = left, right
            heapq.heappush(self.nodes, parent)
        
        self.root = heapq.heappop(self.nodes)
        
        # 预计算编码路径和节点索引（加速关键）
        self._precompute_paths()
    
    def _precompute_paths(self):
        """ DFS遍历生成编码和路径索引 """
        stack = [(self.root, [], [])]
        node_index = {}  # 节点到唯一索引的映射
        idx = 0
        
        while stack:
            node, code, path = stack.pop()
            if node.word:  # 叶子节点
                node.code = code
                node.point = path
                # 分配唯一索引（用于嵌入层）
                node_index[node] = idx
                idx +=1
            else:  # 内部节点
                node_index[node] = idx
                idx +=1
                stack.append( (node.right, code + [1], path + [node_index[node]]) )
                stack.append( (node.left, code + [0], path + [node_index[node]]) )

class HuffmanWord2VecDataset(Dataset):
    def __init__(self, corpus, window_size=5, vocab_size=30000):
        # 词频统计与子采样
        word_counts = Counter(corpus)
        self.word_freq = {w: c/len(corpus) for w, c in word_counts.most_common(vocab_size)}
        
        # 构建Huffman树（训练加速核心）
        self.huffman_tree = HuffmanTree(self.word_freq)
        
        # 生成训练样本
        self.samples = self._generate_samples(corpus, window_size)
    
    def _generate_samples(self, corpus, window_size):
        samples = []
        for i, target in enumerate(corpus):
            if target not in self.word_freq:
                continue
                
            # 动态窗口采样
            curr_window = np.random.randint(1, window_size+1)
            context_words = corpus[max(0,i-curr_window):i] + corpus[i+1:i+curr_window+1]
            
            # 构建样本
            for context in context_words:
                if context in self.word_freq:
                    node = self.huffman_tree.word2node[target]
                    samples.append( (
                        self.huffman_tree.word2node[context].code,  # 正样本编码路径
                        node.code,                                  # 目标编码路径
                        node.point                                  # 路径节点索引
                    ))
        return samples

class HuffmanWord2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim, tree):
        super().__init__()
        # 输入词向量（叶子节点）
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        # 内部节点向量（论文级优化：节点数=2V-1）
        self.node_embed = nn.Embedding(2*vocab_size -1, embed_dim)
        
        # 初始化策略（参考Mikolov论文）
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.constant_(self.node_embed.weight, 0.0)
        
        # 缓存树结构信息
        self.register_buffer('node_indices', torch.tensor(
            [n.index for n in tree.nodes], dtype=torch.long))
    
    def forward(self, target_words, context_codes, context_points):
        """ 
        target_words : [batch_size]
        context_codes: [batch_size, path_length] 
        context_points:[batch_size, path_length]
        """
        # 获取输入向量
        input_vectors = self.in_embed(target_words)  # [B, D]
        
        # 获取路径节点向量
        node_vectors = self.node_embed(context_points)  # [B, L, D]
        
        # 计算路径概率（层次Softmax核心）
        scores = torch.bmm(node_vectors, input_vectors.unsqueeze(2))  # [B, L, 1]
        probs = torch.sigmoid(scores.squeeze(2))  # [B, L]
        
        # 计算损失（带掩码的交叉熵）
        mask = (context_codes != -1).float()  # 填充位置掩码
        loss = -torch.sum( 
            context_codes * torch.log(probs + 1e-7) +
            (1 - context_codes) * torch.log(1 - probs + 1e-7)
        ) * mask
        
        return loss.mean()

# 训练流程优化 ####################################
def train(model, dataset, epochs=10, batch_size=1024):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=4)
    optimizer = optim.SGD(model.parameters(), lr=0.025)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            target, codes, points = batch
            loss = model(target, codes, points)
            
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（防止层次Softmax梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}")

def collate_fn(batch):
    """ 动态填充批次数据 """
    max_length = max(len(codes) for _, codes, _ in batch)
    
    padded_codes = []
    padded_points = []
    targets = []
    
    for t, codes, points in batch:
        pad_len = max_length - len(codes)
        padded_codes.append( codes + [-1]*pad_len )
        padded_points.append( points + [-1]*pad_len )
        targets.append(t)
    
    return (
        torch.LongTensor(targets),
        torch.FloatTensor(padded_codes),  # 0/1编码
        torch.LongTensor(padded_points)   # 节点索引
    )

# 使用示例 ####################################
if __name__ == "__main__":
    # 示例语料
    corpus = [...] # 输入文本数据（需先分词）
    
    # 构建数据集
    dataset = HuffmanWord2VecDataset(corpus)
    
    # 初始化模型
    model = HuffmanWord2Vec(
        vocab_size=len(dataset.word_freq),
        embed_dim=300,
        tree=dataset.huffman_tree
    )
    
    # 训练
    train(model, dataset, epochs=10)
