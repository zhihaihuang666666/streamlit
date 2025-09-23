# 导入包
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt
# 导入必要的组件以显示SHAP的HTML可视化
import streamlit.components.v1 as components

## ===================== 加载模型 =====================##
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# 查看特征 - 这会显示模型期望的特征顺序
model_feature_names = model.get_booster().feature_names
print("模型训练时的特征名：", model_feature_names)
print("模型训练时的特征数量：", len(model_feature_names))

## ===================== 特征列表与配置 =====================##
# 修正特征顺序，使其与模型训练时的顺序一致
FEATURES = model_feature_names  # 使用模型的特征顺序

# 特征类型配置：区分分类特征（二元）和数值特征
CATEGORICAL_FEATURES = [
    "Hypertension", "Diabetes", "Dyslipidaemia", "Lung_disease", "Liver_disease",
    "Kidney_disease", "Stomach_or_other_digestive_disease", "Asthma", 
    "Memory_related_disease", "frailty"
]
NUMERICAL_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

# 特征映射（提升用户体验）
FEATURE_NAMES = {
    "Hypertension": "Hypertension",
    "Diabetes": "Diabetes",
    "Dyslipidaemia": "Dyslipidaemia",
    "Lung_disease": "Lung disease",
    "Liver_disease": "Liver disease",
    "Kidney_disease": "Kidney disease",
    "Stomach_or_other_digestive_disease": "Stomach or other digestive disease",
    "Asthma": "Asthma",
    "Memory_related_disease": "Memory related disease",
    "frailty": "Frailty",
    "SBP": "SBP",
    "waist": "Waist(cm)",
    "CVFI": "CVFI",
    "Age": "Age"  
}

## ===================== Streamlit 页面配置 =====================##
st.set_page_config(page_title="CVD预测系统", layout="wide")
st.title("🧠 CVD预测系统 (XGBoost模型)")

## ===================== 单样本预测 =============================##
st.header("🔹 单样本预测")

# 创建空字典用于存储用户输入的所有特征值
input_data = {} 
col1, col2 = st.columns(2)

# 遍历特征生成输入组件
for i, feature in enumerate(FEATURES):
    # 按奇偶分配到不同列
    with col1 if i % 2 == 0 else col2:
        feature_name = FEATURE_NAMES.get(feature, feature)
        
        if feature in CATEGORICAL_FEATURES:
            # 分类特征使用选择框（0=否，1=是）
            val = st.selectbox(
                f"{feature_name}",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key=feature
            )
        else:
            # 数值特征使用数字输入，并设置合理范围
            if feature == "Age":
                val = st.number_input(f"{feature_name}", min_value=45, max_value=120, value=60, step=1)
            elif feature == "SBP":
                val = st.number_input(f"{feature_name}", min_value=50, max_value=220, value=100, step=1)
            elif feature == "waist":
                # 将所有参数统一为浮点数，与step=0.5保持一致
                val = st.number_input(f"{feature_name}", min_value=20.0, max_value=150.0, value=80.0, step=0.1)
            elif feature == "CVFI":
                # 将所有参数统一为浮点数，与step=0.5保持一致
                val = st.number_input(f"{feature_name}", min_value=80.0, max_value=200.0, value=100.0, step=0.1)
        # 将用户输入的特征值存储到input_data字典中，键为特征名，值为用户输入值
        input_data[feature] = val

# 预测按钮与逻辑
if st.button("预测单样本"):
    try:
        # 构造输入DataFrame，确保列顺序与模型期望一致
        df_input = pd.DataFrame([input_data], columns=FEATURES)
        
        # 处理可能的分类特征（如果有字符串类型）
        for col in df_input.columns:
            if df_input[col].dtype == object:
                le = LabelEncoder()
                df_input[col] = le.fit_transform(df_input[col].astype(str))
        
        # 标准化数值特征（注意：实际部署应使用训练时的scaler，此处为简化处理）
        #scaler = StandardScaler()
        #X_scaled = scaler.fit_transform(df_input)
        X_scaled = df_input
        
        # 模型预测
        y_pred = model.predict(X_scaled)[0]
        y_proba = model.predict_proba(X_scaled)[0][1]  # 假设1是阳性类别
        
        # 显示结果
        st.success(f"预测结果: CVD = {y_pred} (概率: {y_proba:.4f})")
        
        # 补充解释信息
        if y_pred == 1:
            st.info("提示：模型预测为阳性，建议结合临床进一步检查。")
        else:
            st.info("提示：模型预测为阴性，建议保持健康生活方式并定期监测。")
        


        ## ===================== SHAP分析 =====================##
        st.subheader("📊 模型预测解释 (SHAP)")
        # 初始化SHAP解释器
        explainer = shap.TreeExplainer(model)
        # 计算SHAP值
        shap_values = explainer.shap_values(X_scaled)
        
        # 生成force_plot
        # 对于二分类问题，我们取类别1的SHAP值
        if isinstance(shap_values, list):
            shap_value = shap_values[1]  # 取正类的SHAP值
        else:
            shap_value = shap_values
        
        # 创建force_plot
        force_plot_html = shap.force_plot(
            explainer.expected_value,        # 基础值
            shap_value[0],                   # 第一个样本的SHAP值
            features=df_input.iloc[0],       # 第一个样本的特征值
            feature_names=df_input.columns,  # 特征名称
            matplotlib=False                 # 生成HTML格式（支持交互）
        )
        
        # 将SHAP的force_plot转换为HTML并在Streamlit中显示
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
        components.html(shap_html, height=600)
        
        # 添加SHAP图解释说明
        st.info("""
        上图展示了各特征对预测结果的影响：
        - 红色表示该特征增加了预测为阳性(HL=1)的概率
        - 蓝色表示该特征降低了预测为阳性(HL=1)的概率
        - 特征条的长度表示影响程度的大小
        """)
            
    except Exception as e:
        st.error(f"预测过程出错：{str(e)}")




##打开终端contro+R,再运行streamlit run "C:\Users\HZH\Desktop\streamlit.app\XGBoost\prediction.py"##
