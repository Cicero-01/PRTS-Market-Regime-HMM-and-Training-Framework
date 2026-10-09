import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler



# 第一个因子：用于观察状态的平均收益特征
# 第二个因子：用于定义“混沌 vs 秩序”的语义锚定（值越大越偏秩序/趋势）
# Input Factors
FEATURES = ['Factor_A', 'Factor_B']
# 语义锚定特征在 FEATURES 中的索引。
# 约定：该特征值越大，越接近“秩序/趋势”状态。
# 如果你的因子含义不同，改这个索引即可。
# State Anchor Feature(Ascending)
SEMANTIC_ANCHOR_IDX = 1

# 状态数 State Number
N_STATES = 2

# 训练窗口与上下文长度 Training Window and Context Length
TRAIN_WINDOW = 700
CONTEXT_LENGTH = 30

# 语义标签名 State Name
SEMANTIC_LABELS = {0: 'Chaos', 1: 'Order'}


# 数据切分与标准化
# Data Split and Normalization
def split_full_data(X, train_window):
    X_train = X.iloc[0:train_window].values
    X_test = X.iloc[train_window:].values
    scaler = StandardScaler()
    # 只允许用训练集的数据来建立标准（计算均值和方差）
    X_train_scaled = scaler.fit_transform(X_train)
    # 测试集绝对不能参与fit，只能被动接受transform
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# 训练 Train
def build_semantic_mapping(model, anchor_idx):

    anchor_means = model.means_[:, anchor_idx]
    order = np.argsort(anchor_means)  # 从小到大排序的内部编号

    # 语义 0 给最小的，语义 1 给最大的（N=2时）
    mapping = {}
    for semantic_id, internal_id in enumerate(order):
        mapping[internal_id] = semantic_id

    return mapping, order


def get_frozen_regime_history(train_data, test_data, full_index,
                              train_window, context_length,
                              anchor_idx=SEMANTIC_ANCHOR_IDX,
                              n_states=N_STATES):
    total_len = len(full_index)
    regimes = np.full(total_len, np.nan)

    print(f"🧠 [阶段 1] 离线学习前 {train_window} 天的市场...")

    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=1000, random_state=42)
    model.fit(train_data)

    # 建立语义映射
    label_mapping, order = build_semantic_mapping(model, anchor_idx)
    print(f"🔗 语义映射: {label_mapping}")

    # In-Sample 推演
    train_states = model.predict(train_data)
    regimes[0:train_window] = [label_mapping[s] for s in train_states]

    print("⚔️ [阶段 2] 训练完成！开始逐日推演测试集...")

    # Out-of-Sample 滚动推演
    test_len = len(test_data)
    for i in range(test_len):
        start_idx = max(0, i - context_length + 1)
        context_data = test_data[start_idx: i + 1]
        raw_states = model.predict(context_data)
        today_state = label_mapping[raw_states[-1]]
        regimes[train_window + i] = today_state

    return pd.Series(regimes, index=full_index), model, label_mapping




def plot_diagnostics(model, df, features, label_mapping, X_scaled,
                     semantic_labels=SEMANTIC_LABELS,
                     anchor_idx=SEMANTIC_ANCHOR_IDX,
                     plot_feature_idx=None):

    if plot_feature_idx is None:
        plot_feature_idx = anchor_idx

    # 反转映射：语义编号 -> 内部编号
    semantic_to_internal = {v: k for k, v in label_mapping.items()}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))


    # 图1.a: 状态转移矩阵
    # Fig 1.a: Transition Probability Matrix
    order_idx = [semantic_to_internal[s] for s in sorted(semantic_labels.keys())]
    transmat = model.transmat_[order_idx, :][:, order_idx]

    tick_labels = [f"State {s} ({semantic_labels[s]})" for s in sorted(semantic_labels.keys())]

    sns.heatmap(transmat, annot=True, cmap="Blues", fmt=".2%",
                xticklabels=tick_labels,
                yticklabels=tick_labels, ax=axes[0, 0])
    axes[0, 0].set_title("Transition Probability Matrix", fontweight='bold')

    print("\n🧮 状态转移矩阵:")
    for i, s_i in enumerate(sorted(semantic_labels.keys())):
        for j, s_j in enumerate(sorted(semantic_labels.keys())):
            print(f"{semantic_labels[s_i]} -> {semantic_labels[s_j]}: {transmat[i, j]:.2%}")


    # 图1.b: EM收敛曲线
    # Fig 1.b: EM Convergence Log-Likelihood
    if hasattr(model, "monitor_") and model.monitor_.history:
        history = model.monitor_.history
        axes[0, 1].plot(history, color='orange', marker='o', markersize=3, linewidth=1.5)
        axes[0, 1].set_title("EM Convergence (Log-Likelihood)", fontweight='bold')
        axes[0, 1].set_xlabel("Iterations")
        axes[0, 1].set_ylabel("Log-Likelihood")
        axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    else:
        axes[0, 1].text(0.5, 0.5, "Convergence History Not Available",
                        ha='center', va='center')


    # 图1.c: 指定特征的分布密度图
    # Fig 1.c: Emission Distribution by State
    plot_feature_name = features[plot_feature_idx]
    default_colors = ['lightcoral', 'lightblue']
    dynamic_palette = {s: default_colors[s] if s < 2 else 'gray' for s in semantic_labels.keys()}

    sns.kdeplot(data=df, x=plot_feature_name, hue='State', fill=True,
                palette=dynamic_palette,
                ax=axes[1, 0], alpha=0.6)
    axes[1, 0].set_title(f"Emission Distribution: {plot_feature_name}",
                         fontweight='bold')


    # 图1.d: State 1(Order)的后验概率曲线
    # Fig 1.d: Posterior Probability of State 1
    probs = model.predict_proba(X_scaled)
    semantic_1 = max(semantic_labels.keys())  # 默认语义1是Order / Default: State 1
    prob_state_1 = probs[:, semantic_to_internal[semantic_1]]

    axes[1, 1].plot(prob_state_1, color='teal', alpha=0.8, linewidth=1)
    axes[1, 1].axhline(0.5, color='gray', linestyle='--', label='50% Boundary')
    axes[1, 1].set_title(f"Posterior Probability of {semantic_labels[semantic_1]}",
                         fontweight='bold')
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()



# 训练主流程
# Training Process

if __name__ == "__main__":
    print("📥加载数据...")
    df = pd.read_csv("feature_store.csv") # Input csv file
    df['Open Time'] = pd.to_datetime(df['Open Time'])

    # 从DataFrame中提取因子
    df_hmm = df[FEATURES]

    # Step 1: 切分+标准化
    X_train_scale, X_test_scale, scaler = split_full_data(df_hmm, train_window=TRAIN_WINDOW)

    # Step 2: 训练+推演
    regimes_series, model, label_mapping = get_frozen_regime_history(
        X_train_scale, X_test_scale, df.index,
        train_window=TRAIN_WINDOW,
        context_length=CONTEXT_LENGTH,
        anchor_idx=SEMANTIC_ANCHOR_IDX,
        n_states=N_STATES
    )

    df['State'] = regimes_series
    df = df.dropna(subset=['State']).reset_index(drop=True)

    # 计算每个状态的持续天数 Duration
    df['State_Change'] = df['State'].diff().ne(0).cumsum()
    duration = df.groupby('State_Change')['State'].agg(['first', 'count'])
    print(duration.groupby('first')['count'].describe())

    # 打印结果 Print Result
    print("\n📝训练结果 Training Result:")
    profile = df.groupby('State').agg({
        FEATURES[0]: ['mean', 'count'],
        FEATURES[1]: 'mean'
    }).rename(columns={f"{FEATURES[0]}_count": 'Days_Count'})
    print(profile)

    # 画图 Draw
    X_full_scale = np.vstack((X_train_scale, X_test_scale))
    plot_diagnostics(
        model=model,
        df=df,
        features=FEATURES,
        label_mapping=label_mapping,
        X_scaled=X_full_scale,
        anchor_idx=SEMANTIC_ANCHOR_IDX,
        plot_feature_idx=None
    )

# 保存模型文件 Save the model
joblib.dump(scaler, 'PRTS_scaler.pkl')

# 保存HMM模型 Save HMM
joblib.dump(model, 'PRTS_hmm.pkl')

# 保存建立的语义映射 Save label
joblib.dump(label_mapping, 'PRTS_mapping.pkl')

print("💾模型已保存 The files have been saved.")