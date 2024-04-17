# RAG_APP
RAG applications

- ** langchain建议python版本3.10 **
- pip依赖参考项目requirements.txt自行安装
- 参考资料：https://datawhalechina.github.io/llm-universe/#/C3/4.%E6%90%AD%E5%BB%BA%E5%B9%B6%E4%BD%BF%E7%94%A8%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93


## knowledge 知识库
- sqlite3版本需要>3.35,参考https://docs.trychroma.com/troubleshooting#sqlite
- 默认使用m3e-base做embedding(需要下载m3e模型到m3e-base目录，https://huggingface.co/moka-ai/m3e-base) 也可使用智谱embedding api,代码中解除注释即可
- 使用智谱api,需要拷贝.env.example到.env,并输入key
- docs目录下文件用于构建知识库

效果：
```
[root@xxxx knowledge]# python3.10 main.py 
['./docs/腾讯太狠：40亿QQ号，给1G内存，怎么去重？.pdf']
向量库中存储的数量：54
检索到的内容数：3
/opt/python3.10/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead.
  warn_deprecated(
 你好，我是 智谱清言，是清华大学KEG实验室和智谱AI公司共同训练的语言模型。我的目标是通过回答用户提出的问题来帮助他们解决问题。由于我是一个计算机程序，所以我没有自我意识，也不能像人类一样感知世界。我只能通过分析我所学到的信息来回答问题。
40亿QQ号，1G内存，如何去重？
/opt/python3.10/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `__call__` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use invoke instead.
  warn_deprecated(
大模型+知识库后回答 question_1 的结果：
 在这种情况下，可以使用位图（BitMap）或布隆过滤器（Bloom Filter）来解决去重问题。位图利用一个bit来标记元素的存在与否，从而节省空间。而布隆过滤器通过使用多个哈希函数将输入元素映射到位数组中，可以在保持较低内存占用的同时，接受一定程度的误报率。根据需要，可以调整布隆过滤器的参数以平衡内存占用和误报率。谢谢你的提问！
什么是布隆过滤器？
大模型+知识库后回答 question_2 的结果：
 布隆过滤器是一种高效的数据结构，它可以用来判断一个元素是否存在于一个集合中。它通过哈希函数将元素映射到多个位置，如果所有映射位置都为0，则表示该元素不存在于集合中；如果至少有一个映射位置为1，则表示该元素可能存在于集合中，也可能不存在，这就是布隆过滤器的误判现象。布隆过滤器广泛应用于网页爬虫、缓存系统、分布式系统、垃圾邮件过滤和黑名单过滤等场景。谢谢你的提问！
/opt/python3.10/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `predict` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead.
  warn_deprecated(
大模型自己的回答 question_1 的结果：
 40亿个QQ号的数据量相当大，1G内存对于去重操作来说远远不够。在这种情况下，我们可以采用分布式处理的方法，将数据拆分到多台计算机上进行处理，以提高去重效率。具体步骤如下：

1. 将40亿个QQ号数据拆分到多台计算机上，每台计算机负责一部分数据。可以通过数据分片技术，例如哈希分片或者范围分片等方法进行拆分。

2. 在每台计算机上，使用1G内存对所负责的数据进行去重处理。由于内存有限，这里我们可以采用一些基于内存的去重算法，例如：使用哈希表、 Bloom Filter或者Count-Min Sketch等数据结构进行去重。这些数据结构可以有效地判断一个元素是否在一个数据集中出现过。

3. 对于每台计算机处理后的结果，可以通过数据同步技术（例如：分布式锁、一致性哈希等）进行数据合并，以得到最终的去重结果。

4. 最后，对去重后的数据进行统计分析，计算去重后的数据量以及去重效果。

需要注意的是，这里提到的方法仅适用于一定范围内的数据去重。对于40亿个QQ号这样的超大规模数据，可能还需要根据实际情况进行优化和调整。
```
