# PRTS-Market-Regime-HMM-and-Training-Framework
HMM (Hidden Markov Model) is an unsupervised machine learning algorithm. It infers hidden real states from observable data and calculates the transition probabilities between different states. Here we represent PRTS, a HMM for Market Regime Classification along with relative training framework, to suits diffrent-source data analysis.

## 1.引言 Introduction

先前的研究发现，由于窗口大小和自带的滞后性，基于极值的简单分形算法无法复现人类感知中的趋势[[1]](https://github.com/Cicero-01/Kaltsit-Swing-Fractal-Regime-Detector#%E5%B1%80%E9%99%90%E4%B8%8E%E8%AE%A8%E8%AE%BA--limitations-and-discussion)。而人工制定更复杂的分类规则又会不可避免地陷入逻辑死板甚至过拟合的问题，而且分类结果的质量好坏也一定程度上与研究者主观判断水平和标签质量相关。相比之下，**机器学习（Machine Learning）** 作为一种让算法从数据自己学习规律的方式，则能较好地解决相应问题。

Our previous research finds that the simple fractal alogirthms based on extrema cannot be equated with the macroscopic "major trends" perceived by the human eye[[1]](https://github.com/Cicero-01/Kaltsit-Swing-Fractal-Regime-Detector#%E5%B1%80%E9%99%90%E4%B8%8E%E8%AE%A8%E8%AE%BA--limitations-and-discussion). Conversely, manually designing more complex classification rules inevitably leads to rigid logic or even over-fitting problem. Moreover, the quality of such classification heavily depends on the researcher's subjective judgment and the quality of manual labels. In contrast, **Machine Learning (ML)**, which empowers algorithms to discover patterns autonomously from data, offers a solution to these challenges.

我们认为，市场状态分类本质上可以看作一个聚类问题（Clustering），因此天然适合采用**无监督学习（Unsupervised Learning）** 的方式进行研究。无监督学习中，机器只负责寻找数据的数学聚集特征，而**训练者仅通过控制输入的因子类型来决定最终状态的金融学意义，并不需要手动为数据打上标签**，这种方式比起监督学习能更好地避免人类交易员对于市场的主观判断对结果造成的影响。

We argue that market regime classification is fundamentally a Clustering problem, making it inherently suitable for **Unsupervised Learning**. In unsupervised learning, the machine focuses solely on identifying mathematical clustering characteristics within the data. **The researcher dictates the financial meaning of the final states exclusively by controlling the types of input factors, completely bypassing the need for manual labeling**. Compared to supervised learning, this approach effectively eliminates the biases introduced by human traders' subjective market judgments.

基于这种思想，我们使用**隐马尔可夫模型（Hidden Markov Model, HMM）** 这一种经典的无监督学习算法构建了一个市场状态分类工具`PRTS`，并开发了配套的训练框架。隐马尔可夫模型的原理可以简单解释为：**HMM通过可观测的值推断背后的隐藏状态，并计算不同状态间转换的概率**。在具体的训练框架中，技术指标和因子（如成交量、K线价格数据、均线和其他因子）作为人类交易员可以观测的表象被输入模型，而市场的状态（多头/空头、趋势/震荡）作为我们希望推知的隐藏状态被模型输出；通过调用训练完成的模型，研究者也可以得到某一时间点市场所处状态。

Based on this philosophy, we developed `PRTS`, a market regime classification tool and its accompanying training framework, leveraging the **Hidden Markov Model (HMM)**—a classic unsupervised learning algorithm. The underlying principle of HMM can be summarized simply: **it infers the underlying hidden states through observable values and calculates the transition probabilities between these states**. Within our specific training framework, technical indicators and factors (e.g., volume, OHLC price data, moving averages) serve as the "observable representations" input into the model. In turn, the market regimes (e.g., Bull/Bear, Trend/Chaos) are output as the "hidden states" we expect to deduce. By utilizing the trained model, researchers can seamlessly pinpoint the specific market state at any given time step.

此外，只需数据遵循标准命名格式`PELOS`，`PRTS`**及其配套的训练框架能支持将不定数量的自定义因子作为输入，并手动划分输出状态的数量**，具有较高的通用性。HMM的训练通过`hmmlearn`包完成调用，对训练者的数学背景、Python能力亦没有较高要求。此外，HMM可以依赖CPU完成训练，相比于需要GPU算力的神经网络对个人研究者更为友好。

Furthermore, provided that the data adheres to the `PELOS` standardized naming schemas, `PRTS` and its training framework can accept an arbitrary number of custom input factors and allow users to manually define the number of output states, ensuring exceptional versatility. The HMM training is executed via the `hmmlearn` library, imposing minimal requirements on the user's mathematical background or Python proficiency. Additionally, unlike neural networks that demand heavy GPU computation, HMM can be trained efficiently on CPU, making it accessible and friendly for independent researchers.

## 2.如何开始 Quick Start
```
├──PRTS_train.py               # run this file / 运行这个
├──feature_store.csv           # (necessary) training data / 必须传入训练数据
├──PRTS_inference.py           # Utilize trained model / 调用成熟模型
	├──PRTS_hmm.pkl
	├──PRTS_mapping.pkl        # (generated) model files / 生成的三个模型权重文件
	└──PRTS_scaler.pkl             
	
```
### 2.1 训练 Train

运行`PRTS_train.py`训练模型。训练完成后会自动生成三个模型权重文件。训练脚本默认读取的文件名为`feature_store.csv`，研究者也可以将`df = pd.read_csv("feature_store.csv")`语句中的文件名修改成自己的文件名。

Run script `PRTS_train.py` to train the model. 3 model weight files will be generated after training. The script recognizes the file with name `feature_store.csv` as input in default setting. And researcher can replace the filename in `df = pd.read_csv("feature_store.csv")` with thier own filename for recognition.

### 2.2 调用模型 Utilize trained model

我们提供了一个简单的实盘调用示例`PRTS_inference.py`，展示如何通过训练得到的权重文件读取新数据进行状态分类，并输出最新一天所在的状态。此外，研究者可以根据自己的需求调用模型文件实现更多功能。

We provide a simple scripy `PRTS_inference.py` of Live Inference as an example, which shows how to utilize the model files to process new data and output the state of the last day. Researcher can utilize the model file to achieve more functions according to their needs.

需要注意，传入的`hmm_featurestore.csv`至少有30行数据，因为模型默认的`context_length=30`。此外调用模型时，变量`FEATURES`中**因子名称和顺序要和训练时一致**。

Notably, the `hmm_featurestore.csv` must cotain more than 30 lines of data as the default `context_length=30`. In variable `FEATURES`, **the factor's name and order ought to be the same as its form in training**.

## 3.如何使用自己的数据 How to use your own data

**3.1 因子数与分类状态数 / Factors and State**

模型训练的因子列名和数量、语义锚定因子、状态数可以通过修改`PRTS_train.py`开头的全局变量进行自定义。

Users can custom input factors, numbers of output state, anchor, sematic labels of state by adjusting several global variable at the beginning of `PRTS_train.py`.

```
FEATURES = ['Factor_A', 'Factor_B']
SEMANTIC_ANCHOR_IDX = 1
N_STATES = 2
SEMANTIC_LABELS = {0: 'Chaos', 1: 'Order'}
```

**输入因子名称 / Input Factor Name:** 

用具体的因子名称替换`Factor_A`和`Factor_B`，确保传入的`feature_store.csv`文件中有对应列名。

Replace `Factor_A` and `Factor_B` with the specific names of factors you want to input. Ensure the same column names exist in `feature_store.csv` file.

**输入因子数量 / Input Factor Number:** 

直接在FEATURES = []中继续添加即可，如`['Factor_A', 'Factor_B', 'Factor_C']`。

To add more factors as input, just continue adding in the "[]", such as `['Factor_A', 'Factor_B', 'Factor_C']`.

**输出状态锚定特征 / Output State Anchored Feature:**

HMM分类出的状态的锚定由`SEMANTIC_ANCHOR_IDX`变量控制，**默认锚定传入的第二个因子**，并将值更高的状态定义为`State 1`(升序)。如需改为锚定第一个因子，只需修改为`SEMANTIC_ANCHOR_IDX = 0`。

The variable `SEMANTIC_ANCHOR_IDX` determines which factor the output state is anchored to. **In default setting, the anchor is the second factor** and state with higher factor value is defined as `State 1`(ascending). To anchor the state to the first factor, modifying the variable as `SEMANTIC_ANCHOR_IDX = 0`.

**输出状态数 / Output State Number:** 

修改`N_STATES`的值，同时修改`SEMANTIC_LABELS`中的语义标签已控制输出状态数。确保状态数和标签数对应，例如：

The number of output state could be changed by adjusting the value of `N_State` and the label in `SEMANTIC_LABELS`. Ensure each state has its corresponding label. For exaample: 

```
N_STATES = 3
SEMANTIC_LABELS = {0: 'Chaos', 1: 'Trend_Up', 2: 'Trend_Down'}
```

**3.2 协方差形式 / Covariance Type**

`PRTS`的**默认协方差形式为对角协方差(Diagonal)**。如需使用全协方差，需要在`get_frozen_regime_history()`函数中的`model = hmm.GaussianHMM()`语句，手动将`covariance_type="diag"`修改为`covariance_type="full"`。

In default setting, `PRTS` use **Diagonal Covariance** as its covariance type. If you want to use Full Covariance for training, please adjust the `covariance_type` to `"full"` in the `model = hmm.GaussianHMM()` part of function `get_frozen_regime_history()`.

**3.3 标准数据格式 / Standard Data Schemas:**

具体的数据格式请见 / For Standard Data Schemas: [PELOS](https://github.com/Cicero-01/PELOS-Standardized-Data-Schemas)

## 4.训练诊断与性能评估 Diagnostics & Performance

为了在训练后验证HMM分类稳定性和聚类有效性，`PRTS`代码中内置一段可视化函数`plot_diagnostics()`，在训练结束后会自动绘制一个四分图诊断报告。在如下示例中，我们展示了一张基于经典双因子的HMM训练结果诊断(Fig 1.)。

To validate the stability and clustering efficacy of training, a visualization function`plot_diagnostics()` has been added to the `PRTS` framework, which could generate a diagnostic figure with four panels automatically. In our example below, we show a diagnostic chart based on a classical two-factor HMM training(Fig 1.).

<img width="1600" height="1200" alt="Result" src="https://github.com/user-attachments/assets/7b979f8c-9c71-4156-a4e6-a0a4239e121b" />

<p align="center">Fig 1. diagnostic report example; factors: "Return", "Kaufman_Efficiency"; Symbol: "BTCUSDT"; Data Source: Binance</p>

**参数**: 在Fig 1.所代表的示例中，我们使用基于一段BTC-USDT交易对现货(Binance Symbol: `BTCUSDT`)的日线数据计算出的2个经典指标的相关因子————收益率`Return`和考夫曼效率`Kaufman_Efficiency`进行训练。状态的语义特征锚定`Kaufman Efficency`，特征值大的状态为`Order`。其他参数保持默认。

**Parameters**: In the example shown in Fig 1., we use two factors related to classcial indicators from the daily Klines of BTC-USDT spot market (Binance Symbol: `BTCUSDT`), `Return` and `Kaufman_Efficiency`, to train our model. The semantic labels of the market states are anchored to factor `Kaufman Efficency`, where states with higher value is defined as `Order`. Other parameters stay at default settings.
 
### 解读诊断报告 Understanding the Diagnostics

**4.1 Top-left: 状态转移概率矩阵 /Transition Probability Matrix** 

该矩阵(Fig 1.a)揭示了市场状态的马尔可夫记忆性(Markovian Memory)。图中每一行代表“当前状态”，每一列代表“下一个状态”，数值分别代表HMM分类出的状态**继续保持当前状态和切换至其他状态的概率**。

This matrix (Fig 1.a) reveals the Markovian Memory of market states. Each line represents the current state while each column represents the next state of market. The values indicate **the probability of current market state remains or switches respectively.**

示例中，`Chaos`状态和`Order`状态分别有95.38%和93.73%的可能继续维持当前状态，这代表`PRTS`较好地抓住了市场的惯性。分类出的市场状态具有很好的黏性，一旦形成就会持续一段时间，没有太强的“状态闪烁问题”。

In our example, the states `Chaos` and `Order` respectively hold a posibility of 95.38% and 93.73% to persist, which indicates `PRTS` successfully captures market inertia. The classified market regimes exhibit strong "stickiness" ——once formed, they tend to sustain for a period, effectively mitigating the "state flickering" issue.

**4.2 Top-right: EM收敛曲线 /EM Convergence Log-Likelihood**

收敛曲线(Fig 1.b)展示期望最大化(Expectation-Maximization)算法在**训练迭代过程中的优化效率**。横轴是迭代次数，纵轴是对数似然。EM算法每迭代一次，都会尝试让似然值更高。

The convergence curve (Fig 1.b) **demonstrates the optimization efficiency of the Expectation-Maximization (EM) algorithm during training iterations**. The x-axis represents the number of iterations, and the y-axis represents the log-likelihood. With each iteration, the EM algorithm attempts to maximize this likelihood value.

在示例中，曲线在初期的快速上升后，在第10次迭代之后基本走平完成收敛，之后也没有剧烈震荡或下降。这说明在示例数据的结构比较干净，模型很容易就找到了最优解。

In this example, after a rapid initial ascent, the curve essentially flattens and converges around the 10th iteration, with no violent oscillations or drops thereafter. This suggests that the structure of the sample data is relatively clean, allowing the model to find the optimal solution swiftly and stably.

**4.3 Bottom-Left: 特征发射分布密度图 /Emission Distribution by State**

特征发射分布密度图(Fig 1.c)体现了不同状态在锚定特征上的分布密度，**展示了模型是如何根据指定的锚定因子在物理层面切割市场的**。

The emission distribution density plot (Fig 1.c) visualizes the distribution of different states across the anchored feature, **illustrating how the model physically segments the market based on the specified anchor factor**.

训练数据中，特征锚定在因子`Kaufman_Efficiency`上，红色区域(State 0: `Chaos`)紧密聚集在较低值区间，而淡蓝色区域(State 1: `Order`)则分布在较高值区间。两座山峰的重叠度较低，证明HMM在完全无监督的条件下，划分出了“趋势”与“震荡”这2个有较高的区分度的状态。同时可以发现蓝色区域分布更宽，说明`Order`状态内部的效率值差异较大。

In the training data, the feature is anchored to factor`Kaufman_Efficiency`. The red area (State 0: `Chaos`) is densely clustered in the lower-value range, while the light blue area (State 1: `Order`) is distributed across the higher-value range. The minimal overlap between these two peaks proves that the HMM, under entirely unsupervised conditions, autonomously discovered a high degree of separation between "Order" and "Chaos". Furthermore, the wider distribution of the blue area indicates a greater variance in efficiency values within the `Order` state.

值得注意的是，大约在0.25-0.35之间存在两个状态都有分布的重叠区间。这说明`Kaufman_Efficiency`作为锚定特征是有效的，但它不是完美的分离标准。

Notably, there is an overlapping region between approximately 0.25 and 0.35 where both states coexist. This implies that while `Kaufman_Efficiency` is effective as an anchoring feature, it is not a perfect separation criterion.

**4.4 Bottom-right: State 1后验概率曲线 /Posterior Probability of State 1**

后验概率曲线(Fig 1.d)是一张**很重要的图，衡量了模型对当前市场状态推断的置信度(Confidence)**。

The posterior probability curve (Fig 1.d) is a crucial chart that **measures the model's confidence in inferring the current market state**.

在展示的数据中，属于State 1(`Order`)的概率曲线在0和1之间硬切换，没有大量中间值，也极少在0.5(50% 决策边界)附近纠结徘徊.这说明`PRTS`的状态判定是极其果断明确、高置信度的。其状态切换频率和持续时间与Fig 1.的前三张图能够相互印证。这是HMM发射分布分离度较好的表现。

In the presented data, the probability curve for State 1 (Order) exhibits hard switching between 0 and 1, with very few intermediate values and rarely lingering around the 0.5 (50%) decision boundary. This demonstrates that state determination of `PRTS` is exceptionally decisive and highly confident. The frequency and duration of its state transitions perfectly corroborate the first three panels of Fig 1, which is a direct reflection of the excellent separation in the HMM's emission distributions.

## ⚠️**免责声明 / Disclaimer:** 

本工具仅供研究学习使用，**不构成任何投资或财务建议**。开发者及贡献者对因使用本软件或其中代码所造成的任何直接或间接财务损失，不承担任何法律责任。金融市场交易具有极高风险，请在真实交易前进行充分测试，并自行承担所有风险 (DYOR)。

This project and its tools are provided for educational and research purposes only and **do not constitute financial or investment advice**. The developers and contributors assume no legal responsibility or liability for any direct or indirect financial losses incurred from the use of this software. Trading in financial markets involves significant risk. Always do your own research (DYOR) and test thoroughly before real trading.
