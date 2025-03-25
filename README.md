# word2vec-pytorch
复现一下经典的 Word2Vec模型，顺便对其原理做一下笔记，免得面试的时候搞我。


- Word2Vec 是一种基于语言模型的词向量

1. 为什么需要他
- 为了更好地训练word embedding矩阵，使得每个向量都拥有语义含义

- 我们希望得到一种词向量，使得向量关系能反映语义关系，比如：
  - cos（你好， 您好） >  cos(你好，天气） 
  - 即词义的相似性反映在向量的相似性

- 国王 - 男人 = 皇后 -女人
  - 即向量可以通过数值运算反映词之间的关系

同时，不管有多少词，向量维度应当是固定的

- Word Embedding 和 Word Vector的区别？
  - 两者本质相同，都是用向量来代表词
  - Word Embedding 是指随机初始化的词向量或句向量
  - Word2Vec是一种训练WordEmbedding的方法，使得向量相似度反应语义相似度





1.2 语言模型的作用？

假设：
每段文本中的每一个词，由它前面的n个词决定
- 给定一段文本，让他预测下一个字是什么
- 小明 要 -> 学习
- 小明 要 学习 -> NLP
- 小明 要 学习 NLP -> 来理解
- 小明 要 学习 NLP 来理解 -> 大模型

   本质上，如果我们得到词表里有5000个词，“大模型”是第100个词
   他其实就是一个5000分类的任务，只要预测出的类别是100就行


- 因此，语言模型和多分类器没啥区别



3. 达成的效果
预测下一个词的正确率比较好


4. 语言模型的好处
  - 自监督：不用人工标注训练样本，因为下一个字本身就是标签
  - 一开始我们已经知道：小明要学习NLP来理解大模型。
  - 如果我们给语言模型：“小明要学习NLP来理解”，让他预测下一个词，这时，“大模型”就自动成为了标签。
  

下面论文片段的含义：相似的词会有相似的向量
![image](https://github.com/user-attachments/assets/cca554df-c068-4425-935b-ad0e8f978b2e)

其中， probability function就是我们说的模型




- 改论文提出的主要方法
- ![image](https://github.com/user-attachments/assets/d4eb4fc1-cdf5-4788-9fd0-fe7c67caf949)
简而言之，所提出方法的核心思想可以概括如下：

1. 为词汇表中的每个单词关联一个分布式**词特征向量**（一个实值向量，属于 \( \mathbb{R}^m \)）。  
2. 将**词序列的联合概率函数**用这些单词的特征向量来表示。  
3. **同时学习**这些词特征向量以及该概率函数(Word2Vec模型)的参数。 【词向量和模型参数，一起训练！！！】




## Word2Vec语言模型的模型结构

![image](https://github.com/user-attachments/assets/72d7163c-26d6-4d8f-b1e9-6f49b6f20f15)

![image](https://github.com/user-attachments/assets/1c0df209-afe8-4098-a287-f87af959901a)






## Word2Vec的输出和loss
- 输出是第n个词的概率(已知前n-1个词）, 即，词表上的概率分布向量
- Loss  = crossEntropy, 

