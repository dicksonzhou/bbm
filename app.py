import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(page_title="骨髓转移智能预测平台", layout="wide", page_icon="🏥")

st.title("🏥 骨髓转移智能预测平台 (自定义数据版)")
st.markdown("""
**操作流程说明：**
1. 📂 **上传数据**: 在左侧栏上传您的历史临床数据 (CSV格式)。
2. 🤖 **自动训练**: 系统将自动识别特征并训练 XGBoost 机器学习模型。
3. 🩺 **预测分析**: 输入新患者指标，系统将计算转移概率并解释原因。
""")

# 定义常用字段的中英对照字典（用于优化侧边栏显示）
# 即使CSV是英文表头，侧边栏也能显示中文，方便医生输入
COLUMN_TRANSLATION = {
    'Age': '年龄',
    'Gender': '性别',
    'LDH': '乳酸脱氢酶 (LDH)',
    'ALP': '碱性磷酸酶 (ALP)',
    'HB': '血红蛋白 (HB)',
    'PLT': '血小板 (PLT)',
    'Ca': '血钙 (Ca)',
    'Primary_Cancer': '原发肿瘤部位',
    'Bone_Marrow_Metastasis': '骨髓转移状态',
    'Patient_ID': '患者ID'
}

# ==========================================
# 2. 侧边栏：数据上传与模型训练
# ==========================================
st.sidebar.header("📂 1. 数据上传")

uploaded_file = st.sidebar.file_uploader("请上传训练数据 (CSV文件)", type=["csv"])

@st.cache_resource
def train_model_from_csv(file):
    try:
        # 读取数据
        df = pd.read_csv(file)
        
        # 简单的清洗：删除ID列 (假设包含 ID 字样的列是无关列)
        cols_to_drop = [c for c in df.columns if 'ID' in c.upper()]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            
        # 检查是否包含目标列 (支持中文或英文的目标列名)
        possible_targets = ['Bone_Marrow_Metastasis', '骨髓转移', 'Target', 'Label', '转移状态']
        target_col = next((col for col in possible_targets if col in df.columns), None)
        
        if not target_col:
            return None, None, None, f"❌ 错误：CSV中未找到目标列 (例如: 'Bone_Marrow_Metastasis' 或 '骨髓转移')"
        
        # 分离特征和标签
        X_raw = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 自动识别分类特征和数值特征
        cat_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
        
        # 记录分类变量的原始选项（用于生成UI）
        cat_options = {col: X_raw[col].unique().tolist() for col in cat_cols}
        
        # One-Hot 编码
        X = pd.get_dummies(X_raw, columns=cat_cols)
        
        # 训练模型
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 计算正负样本比例，处理样本不平衡
        pos_ratio = y_train.value_counts().min() / y_train.value_counts().max()
        scale_pos_weight = 1 / pos_ratio if pos_ratio > 0 else 1
        
        model = xgb.XGBClassifier(
            n_estimators=150, 
            max_depth=5, 
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False
        )
        model.fit(X_train, y_train)
        
        # 计算指标
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.0 # 如果测试集只有一个类别，AUC无法计算
        
        metrics = {"acc": acc, "auc": auc}
        
        return model, X_train, cat_options, metrics
        
    except Exception as e:
        return None, None, None, f"❌ 训练出错: {str(e)}"

if uploaded_file is not None:
    with st.spinner('正在解析数据并训练 AI 模型，请稍候...'):
        model, X_train_ref, cat_options, metrics = train_model_from_csv(uploaded_file)
    
    if isinstance(metrics, str): # 报错信息
        st.error(metrics)
        st.stop()
    else:
        st.sidebar.success(f"✅ 模型训练成功!")
        st.sidebar.info(f"📊 模型精度 (Acc): {metrics['acc']:.1%}\n\n📈 AUC 值: {metrics['auc']:.3f}")

else:
    st.warning("👈 请先在左侧侧边栏上传数据文件 (推荐使用 bone_marrow_data.csv)")
    st.stop()

# ==========================================
# 3. 侧边栏：动态生成输入表单
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🩺 2. 患者特征输入")

user_input = {}

# 获取训练数据的原始列名（排除One-Hot生成的列）
# 这里需要一些技巧来恢复原始的数值列名
all_model_cols = X_train_ref.columns.tolist()
input_cols = []

# A. 处理分类变量
if cat_options:
    for col_name, options in cat_options.items():
        # 显示友好的中文标签
        label = f"{COLUMN_TRANSLATION.get(col_name, col_name)}"
        if label != col_name:
            label += f" ({col_name})" # 如果有翻译，保留原名在括号里
            
        user_input[col_name] = st.sidebar.selectbox(label, options)

# B. 处理数值变量
# 排除掉 One-Hot 产生的列 (例如 Primary_Cancer_Lung)
one_hot_prefixes = [f"{col}_" for col in cat_options.keys()]

for col in X_train_ref.columns:
    is_one_hot = any(col.startswith(prefix) for prefix in one_hot_prefixes)
    
    if not is_one_hot:
        # 这是一个数值列
        label = COLUMN_TRANSLATION.get(col, col)
        if label != col:
            label += f" ({col})"
            
        # 根据常用医学指标设置默认范围，提升体验
        if 'Age' in col or '年龄' in col:
            val = st.sidebar.slider(label, 1, 100, 60)
        elif 'Gender' in col and not cat_options: # 如果性别被识别为数值(0/1)
             val = st.sidebar.selectbox(label, [0, 1])
        else:
            # 默认数值输入
            val = st.sidebar.number_input(label, value=0.0)
            
        user_input[col] = val

# ==========================================
# 4. 预测与解释逻辑
# ==========================================

# 构建输入 DataFrame
input_df_raw = pd.DataFrame([user_input])

# 数据对齐：确保输入的列与训练时完全一致
input_df_encoded = pd.get_dummies(input_df_raw)
input_df_final = input_df_encoded.reindex(columns=X_train_ref.columns, fill_value=0)

# 界面主区域
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 当前输入数据")
    # 转置显示更美观
    st.dataframe(input_df_raw.T.rename(columns={0: '数值'}), use_container_width=True)
    
    start_predict = st.button("🚀 开始预测 & 分析", type="primary", use_container_width=True)

if start_predict:
    # A. 预测计算
    prob = model.predict_proba(input_df_final)[0][1]
    
    st.markdown("---")
    st.subheader("🎯 预测结论")
    
    c1, c2, c3 = st.columns(3)
    
    # 根据概率显示不同颜色
    c1.metric("骨髓转移概率", f"{prob:.2%}")
    
    if prob > 0.5:
        risk_text = "高风险 (High Risk)"
        risk_color = "red"
        icon = "⚠️"
    else:
        risk_text = "低风险 (Low Risk)"
        risk_color = "green"
        icon = "✅"
        
    c2.markdown(f"风险等级:<br><span style='color:{risk_color};font-size:20px;font-weight:bold'>{icon} {risk_text}</span>", unsafe_allow_html=True)
    
    st.progress(float(prob))

    # B. SHAP 解释
    st.markdown("---")
    st.subheader("🔍 AI 归因分析 (SHAP)")
    st.info("说明：红色条代表该指标增加了患病风险，蓝色条代表该指标降低了患病风险。")
    
    with st.spinner("正在进行 SHAP 计算，请稍候..."):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_df_final)
            
            # 1. 瀑布图
            st.write("#### 1. 个体决策分析 (Waterfall Plot)")
            st.caption("展示了该患者各项指标如何共同作用，导致了最终的预测概率。")
            fig1, ax1 = plt.subplots()
            # max_display 控制显示多少个重要特征
            shap.plots.waterfall(shap_values[0], show=False, max_display=10)
            st.pyplot(fig1)
            
            # 2. 蜂群图
            st.write("#### 2. 全局特征重要性 (Beeswarm Plot)")
            with st.expander("点击查看模型全局逻辑"):
                st.caption("基于训练集前200个样本的分析：点的颜色越红代表数值越高，位置越靠右代表风险越高。")
                # 为了速度，只取部分样本做背景
                bg_samples = X_train_ref.iloc[:200]
                shap_values_bg = explainer(bg_samples)
                
                fig2, ax2 = plt.subplots()
                shap.plots.beeswarm(shap_values_bg, show=False)
                st.pyplot(fig2)
        except Exception as e:
            st.error(f"SHAP图表生成失败: {str(e)} (通常是因为数据格式兼容性问题)")

st.markdown("---")
st.caption("⚠️ 免责声明：本工具仅供医学科研与辅助教学使用，预测结果不能替代医生的专业临床诊断。")