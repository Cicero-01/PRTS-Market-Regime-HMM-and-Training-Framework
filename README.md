# PRTS-Market-Regime-HMM-and-Training-Framework
HMM (Hidden Markov Model) is an unsupervised machine learning algorithm. It infers hidden real states from observable data and calculates the transition probabilities between different states. Here we represent PRTS, a HMM for Market Regime Classification along with relative training framework, to suits diffrent-source data analysis.

## 1.引言 Introduction

先前的研究发现，由于窗口大小和自带的滞后性，基于极值的简单分形算法无法复现人类感知中的趋势[[1]](https://github.com/Cicero-01/Kaltsit-Swing-Fractal-Regime-Detector#%E5%B1%80%E9%99%90%E4%B8%8E%E8%AE%A8%E8%AE%BA--limitations-and-discussion)。而人工制定更复杂的分类规则又会不可避免地陷入逻辑死板甚至过拟合的问题，而且分类结果的质量好坏也一定程度上与研究者主观判断和准确水平相关。相比之下，**机器学习（Machine Learning）** 作为一种让算法从数据自己学习规律的方式，则能较好地解决。

Our previous research finds that the simple fractal alogirthms based on extrema cannot be equated with the macroscopic "major trends" perceived by the human eye[[1]](https://github.com/Cicero-01/Kaltsit-Swing-Fractal-Regime-Detector#%E5%B1%80%E9%99%90%E4%B8%8E%E8%AE%A8%E8%AE%BA--limitations-and-discussion). 

我们认为，市场状态分类本质上可以看作一个聚类问题（Clustering），因此天然适合采用**无监督学习（Unsupervised Learning）** 的方式进行研究。无监督学习中，机器只负责寻找数据的数学聚集特征，而**训练者仅通过控制输入的因子类型来决定最终状态的金融学意义，并不需要手动为数据打上标签**，这种方式比起监督学习能更好地避免人类交易员对于市场的主观判断对结果造成的影响。

基于这种思想，我们使用**隐马尔可夫模型（Hidden Markov Model, HMM）** 这一种经典的无监督学习算法构建了一个市场状态分类工具`PRTS`，并开发了配套的训练框架。隐马尔可夫模型的原理可以简单解释为：**HMM通过可观测的值推断背后的隐藏状态，并计算不同状态间转换的概率**。在具体的训练框架中，技术指标和因子（如成交量、K线价格数据、均线和其他因子）作为人类交易员可以观测的表象被输入模型，而市场的状态（多头/空头、趋势/震荡）作为我们希望推知的隐藏状态被模型输出；通过调用训练完成的模型，研究者也可以得到某一时间点市场所处状态。

此外，只需数据遵循标准命名格式`PELOS`，`PRTS`**及其配套的训练框架能支持将不定数量的自定义因子作为输入，并手动划分输出状态的数量**，具有较高的通用性。HMM的训练通过`hmmlearn`包完成调用，对训练者的数学背景、Python能力亦没有较高要求。此外，HMM可以依赖CPU完成训练，相比于需要GPU算力的神经网络对个人研究者更为友好。

## 2.如何开始 Quick Start
```
├──Main.py               # run this file / 运行这个
├──feature_store.csv     # necessary / 必须
└── results/             # Strategy results / 策略结果文件夹 
	
```
### 2.1 训练 Train

### 2.2 调用模型 

## 3.如何使用自己的数据 How to use your own data

**因子数与分类状态数**

**收敛形式**


## 4.训练诊断与性能评估 Diagnostics & Performance

为了在训练后验证HMM分类稳定性和聚类有效性，`PRTS`代码中内置一段可视化函数`plot_diagnostics()`，在训练结束后会自动绘制一个四分图诊断报告。在如下示例中，我们展示了一张基于经典双因子的HMM训练结果诊断(Fig 1.)。

To validate the stability and clustering efficacy of training, a visualization function`plot_diagnostics()` has been added to the `PRTS` framework, which could generate a diagnostic figure with four panels automatically. In our example below, we show a diagnostic chart based on a classical two-factor HMM training(Fig 1.).

<img width="1600" height="1200" alt="Result" src="https://github.com/user-attachments/assets/7b979f8c-9c71-4156-a4e6-a0a4239e121b" />

<p align="center">Fig 1. diagnostic report example; factors: "Return", "Kaufman_Efficiency"; Symbol: "BTCUSDT"; Data Source: Binance</p>

**参数**: 在Fig 1.所代表的示例中，我们使用基于一段BTC-USDT交易对现货(Binance Symbol: `BTCUSDT`)的日线数据计算出的2个经典指标的相关因子————收益率`Return`和考夫曼效率`Kaufman_Efficiency`进行训练。状态的语义特征锚定`Kaufman Efficency`，特征值大的状态为`Order`。其他参数保持默认。

**Parameters**: In the example shown in Fig 1., we use two factors related to classcial indicators from the daily Klines of BTC-USDT spot market (Binance Symbol: `BTCUSDT`), `Return` and `Kaufman_Efficiency`, to train our model. The semantic labels of the market states are anchored to factor `Kaufman Efficency`, where states with higher value is defined as `Order`. Other parameters stay at default settings.
 
### 解读诊断报告 Understanding the Diagnostics

**4.1 Top-left: 状态转移概率矩阵 Transition Probability Matrix** 

该矩阵(Fig 1.a)揭示了市场状态的马尔可夫记忆性(Markovian Memory)。图中每一行代表“当前状态”，每一列代表“下一个状态”，数值分别代表HMM分类出的状态**继续保持当前状态和切换至其他状态的概率**。

This matrix (Fig 1.a) reveals the Markovian Memory of market states. Each line represents the current state while each column represents the next state of market. The values indicate **the probability of current market state remains or switches respectively.**

示例中，`Chaos`状态和`Order`状态分别有95.38%和93.73%的可能继续维持当前状态，这代表`PRTS`较好地抓住了市场的惯性。分类出的市场状态具有很好的黏性，一旦形成就会持续一段时间，没有太强的“状态闪烁问题”。

In our example, the states `Chaos` and `Order` respectively hold a posibility of 95.38% and 93.73% to persist, which indicates `PRTS` successfully captures market inertia. The classified market regimes exhibit strong "stickiness" ——once formed, they tend to sustain for a period, effectively mitigating the "state flickering" issue.

**4.2 Top-right: EM收敛曲线 EM Convergence Log-Likelihood**

收敛曲线(Fig 1.b)展示期望最大化(Expectation-Maximization)算法在**训练迭代过程中的优化效率**。横轴是迭代次数，纵轴是对数似然。EM算法每迭代一次，都会尝试让似然值更高。

The convergence curve (Fig 1.b) **demonstrates the optimization efficiency of the Expectation-Maximization (EM) algorithm during training iterations**. The x-axis represents the number of iterations, and the y-axis represents the log-likelihood. With each iteration, the EM algorithm attempts to maximize this likelihood value.

在示例中，曲线在初期的快速上升后，在第10次迭代之后基本走平完成收敛，之后也没有剧烈震荡或下降。这说明在示例数据的结构比较干净，模型很容易就找到了最优解。

In this example, after a rapid initial ascent, the curve essentially flattens and converges around the 10th iteration, with no violent oscillations or drops thereafter. This suggests that the structure of the sample data is relatively clean, allowing the model to find the optimal solution swiftly and stably.

**4.3 Bottom-Left: 特征发射分布密度图 Emission Distribution by State**

特征发射分布密度图(Fig 1.c)体现了不同状态在锚定特征上的分布密度，**展示了模型是如何根据指定的锚定因子在物理层面切割市场的**。

The emission distribution density plot (Fig 1.c) visualizes the distribution of different states across the anchored feature, **illustrating how the model physically segments the market based on the specified anchor factor**.

训练数据中，特征锚定在因子`Kaufman_Efficiency`上，红色区域(State 0: `Chaos`)紧密聚集在较低值区间，而淡蓝色区域(State 1: `Order`)则分布在较高值区间。两座山峰的重叠度较低，证明HMM在完全无监督的条件下，划分出了“趋势”与“震荡”这2个有较高的区分度的状态。同时可以发现蓝色区域分布更宽，说明`Order`状态内部的效率值差异较大。

In the training data, the feature is anchored to factor`Kaufman_Efficiency`. The red area (State 0: `Chaos`) is densely clustered in the lower-value range, while the light blue area (State 1: `Order`) is distributed across the higher-value range. The minimal overlap between these two peaks proves that the HMM, under entirely unsupervised conditions, autonomously discovered a high degree of separation between "Order" and "Chaos". Furthermore, the wider distribution of the blue area indicates a greater variance in efficiency values within the `Order` state.

值得注意的是，大约在0.25-0.35之间存在两个状态都有分布的重叠区间。这说明`Kaufman_Efficiency`作为锚定特征是有效的，但它不是完美的分离标准。

Notably, there is an overlapping region between approximately 0.25 and 0.35 where both states coexist. This implies that while `Kaufman_Efficiency` is effective as an anchoring feature, it is not a perfect separation criterion.

**4.4 Bottom-right: State 1后验概率曲线 Posterior Probability of State 1**

后验概率曲线(Fig 1.d)是一张**很重要的图，衡量了模型对当前市场状态推断的置信度(Confidence)**。

The posterior probability curve (Fig 1.d) is a crucial chart that **measures the model's confidence in inferring the current market state**.

在展示的数据中，属于State 1(`Order`)的概率曲线在0和1之间硬切换，没有大量中间值，也极少在0.5(50% 决策边界)附近纠结徘徊.这说明`PRTS`的状态判定是极其果断明确、高置信度的。其状态切换频率和持续时间与Fig 1.的前三张图能够相互印证。这是HMM发射分布分离度较好的表现。

In the presented data, the probability curve for State 1 (Order) exhibits hard switching between 0 and 1, with very few intermediate values and rarely lingering around the 0.5 (50%) decision boundary. This demonstrates that state determination of `PRTS` is exceptionally decisive and highly confident. The frequency and duration of its state transitions perfectly corroborate the first three panels of Fig 1, which is a direct reflection of the excellent separation in the HMM's emission distributions.

