# Final-Project-Review
## 简介
本项目是《面向电子商务的可解释推荐系统研究》的部分源码，本文提出了基于注意力机制的可解释推荐系统（Explainable recommendation system based on Text Attention enabled Convolutional Neural Network，TACNN），旨在对可解释推荐系统进行进一步研究，使得模型在保持较高推荐准确度的同时给了用户句子级别的解释结果
## 实验细节
* 本实验使用[亚马逊公开数据集](https://nijianmo.github.io/amazon/index.html)中的Musical Instrument, Video Games, Toys and Games，数据集经过data-preprocess.ipynb处理，数据示例如下所示<br>
<img src="https://github.com/sleepindogg/Final-Project-Review/blob/master/imgs/data-img.png" height="250" alt="data" /><br/>
* 本实验模型存在models下，其中本文提出的模型是TACNN，TACNN-attention是模型去掉attention层的对比模型<br>
* 本模型分为三层，基于CNN的特征提取层，注意力层和预测层，详细结构见下图<br>
<img src="https://github.com/sleepindogg/Final-Project-Review/blob/master/imgs/model.png" height="250" alt="data" /><br/>
* 通过train.ipynb训练模型，train-visulizaiton.ipynb展示了包含句子级解释的返回，训练过程如下图所示
<img src="https://github.com/sleepindogg/Final-Project-Review/blob/master/imgs/train-img.png" height="250" alt="data" /><br/>
* 最终的训练结果如下所示<br>
<img src="https://github.com/sleepindogg/Final-Project-Review/blob/master/imgs/result1.png" height="50" alt="data" /><br/>
<img src="https://github.com/sleepindogg/Final-Project-Review/blob/master/imgs/result2.png" height="250" alt="data" /><br/>



##### 运行环境：python3.6 Tensorflow 1.15
