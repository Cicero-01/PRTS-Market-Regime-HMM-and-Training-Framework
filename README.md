# PRTS-Market-Regime-HMM-and-Training-Framework
HMM (Hidden Markov Model) is an unsupervised machine learning algorithm. It infers hidden real states from observable data and calculates the transition probabilities between different states. Here we represent PRTS, a HMM for Market Regime Classification along with relative training framework, to suits diffrent-source data analysis.

## 引言 Introduction

先前的研究发现，由于窗口大小和自带的滞后性，基于极值的简单分形算法无法复现人类感知中的趋势[[1]](https://github.com/Cicero-01/Kaltsit-Swing-Fractal-Regime-Detector#%E5%B1%80%E9%99%90%E4%B8%8E%E8%AE%A8%E8%AE%BA--limitations-and-discussion)。而人工制定更复杂的分类规则又会不可避免地陷入逻辑死板甚至过拟合的问题，而且分类结果的质量好坏也一定程度上与研究者主观判断和准确水平相关。相比之下，**机器学习（Machine Learning）** 作为一种让算法从数据自己学习规律的方式，则能较好地解决。

我们认为，市场状态分类本质上可以看作一个聚类问题（Clustering），因此天然适合采用**无监督学习（Unsupervised Learning）** 的方式进行研究。无监督学习中，机器只负责寻找数据的数学聚集特征，而**训练者仅通过控制输入的因子类型来决定最终状态的金融学意义，并不需要手动为数据打上标签**，这种方式比起监督学习能更好地避免人类交易员对于市场的主观判断对结果造成的影响。

基于这种思想，我们使用**隐马尔可夫模型（Hidden Markov Model, HMM）** 这一种经典的无监督学习算法构建了一个市场状态分类工具`PRTS`，并开发了配套的训练框架。隐马尔可夫模型的原理可以简单解释为：**HMM通过可观测的值推断背后的隐藏状态，并计算不同状态间转换的概率**。在具体的训练框架中，技术指标和因子（如成交量、K线价格数据、均线和其他因子）作为人类交易员可以观测的表象被输入模型，而市场的状态（多头/空头、趋势/震荡）作为我们希望推知的隐藏状态被模型输出；通过调用训练完成的模型，研究者也可以得到某一时间点市场所处状态。

此外，只需数据遵循标准命名格式`PELOS`，`PRTS`**及其配套的训练框架能支持将不定数量的自定义因子作为输入，并手动划分输出状态的数量**，具有较高的通用性。HMM的训练通过`hmmlearn`包完成调用，对训练者的数学背景、Python能力亦没有较高要求。此外，HMM可以依赖CPU完成训练，相比于需要GPU算力的神经网络对个人研究者更为友好。

## 如何开始 Quick Start
