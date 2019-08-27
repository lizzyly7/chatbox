# ChatBox

## 生成式问答算法的实现

### 语料

来自于没有整理过的网络对话大概40w
全部为繁体字，需要进行繁体到简体转换
内容地址：
https://pan.baidu.com/s/1M-qc0Zo0rC3PWTAkeEvocg

### 模型

seq2seq全家桶

借鉴：Tensorflow中的Seq2Seq全家桶 - 王岳王院长的文章 

#### Teacher Forcing

![teacher forcing](https://img-blog.csdnimg.cn/20181110194356146.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pteDE5OTY=,size_16,color_FFFFFF,t_70) 

#### Helper

在训练过程中使用teacher forcing 可以保证训练的过程中预测语句不会出错，但是在predict过程中就没有语句进行校正，所以这时候对于每一时间点的输入，我们都要计算每个点的最大概率输出，然后直到时间点结束。
TrainingHelper：
 train 阶段新建一个用 TrainingHelper 的模型，训练完了保存模型参数
GreedyEmbeddingHelper：
 test 阶段再新建另一个用 GreedyEmbeddingHelper 的模型，直接加载训练好的参数就可以用dynamic_decode 函数类似于 dynamic_rnn，帮你自动执行 rnn 的循环，返回完整的输出序列

#### Attention

<ATTENTION IS ALL YOU NEED>
  
跟之前基础 seq2seq 模型的区别，就是给 decoder 多提供了一个输入“c”。因为 encoder把很长的句子压缩只成了一个小向量“u”，decoder在解码的过程中没准走到哪一步就把“u”中的信息忘了，所以在decoder 解码序列的每一步中，都再把 encoder 的 outputs 拉过来让它回忆回忆。但是输入序列中每个单词对 decoder 在不同时刻输出单词时的帮助作用不一样，所以就需要提前计算一个 attention score 作为权重分配给每个单词，再将这些单词对应的 encoder output 带权加在一起，就变成了此刻 decoder 的另一个输入“c”
  
 #### Beam Search
 使用了 Beam Search，在每个时刻会选择 top K 的单词都作为这个时刻的输出，逐一作为下一时刻的输入参与下一时刻的预测，然后再从这 K*L（L为词表大小）个结果中选 top K 作为下个时刻的输出，以此类推。在最后一个时刻，选 top 1 作为最终输出。实际上就是剪枝后的深搜策略
 
#### Sequence Loss
  对于pad过的sentence，多次计算pad的预测会降低整体loss，所以对于句子来说，到第一个pad 就可以直接统计loss

