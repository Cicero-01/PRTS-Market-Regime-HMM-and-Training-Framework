import joblib
import pandas as pd

def load_prts_memory(scaler_path='PRTS_scaler.pkl',
                     model_path='PRTS_hmm.pkl',
                     mapping_path='PRTS_mapping.pkl'):

    print("📥 正在加载模型...")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    label_mapping = joblib.load(mapping_path)
    return scaler, model, label_mapping

def predict_today_state(new_df, scaler, model, label_mapping,
                        features, context_length=30):

    X_new = new_df[features].values
    X_new_scaled = scaler.transform(X_new)

    # 只取最后context_length天
    context_data = X_new_scaled[-context_length:]

    raw_states = model.predict(context_data)
    today_state = label_mapping[raw_states[-1]]

    return today_state

if __name__ == "__main__":
    # 传入因子 Input Factors
    FEATURES = ['Factor_A', 'Factor_B']
    scaler, model, label_mapping = load_prts_memory()

    df_live = pd.read_csv("hmm_featurestore.csv")
    # 使用
    today = predict_today_state(df_live, scaler, model, label_mapping, FEATURES)
    print(f"今日时间: {df_live['Open Time'].iloc[-1]}")
    print(f"今日状态: {today} ({'Chaos' if today == 0 else 'Order'})")