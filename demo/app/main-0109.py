import streamlit as st
from model_loader import ChurnPredictor

# 设置页面
st.set_page_config(page_title="客户流失风险预测系统", layout="wide")
st.title("📊 客户流失风险预测系统")
st.markdown("输入客户特征，预测其流失风险，并提供业务决策建议。")

# 侧边栏：输入特征
st.sidebar.header("客户特征输入")

# 二元特征
gender = st.sidebar.selectbox("性别", ["Female", "Male"])
senior_citizen = st.sidebar.checkbox("是否为老年人")
partner = st.sidebar.selectbox("是否有伴侣", ["No", "Yes"])
dependents = st.sidebar.selectbox("是否有家属", ["No", "Yes"])
phone_service = st.sidebar.selectbox("是否有电话服务", ["No", "Yes"])
paperless_billing = st.sidebar.selectbox("是否使用无纸化账单", ["No", "Yes"])

# 数值特征
tenure = st.sidebar.slider("在网时长（月）", 0, 72, 24)
monthly_charges = st.sidebar.number_input("月度费用（美元）", 0.0, 200.0, 70.0)

total_charges = st.sidebar.number_input("总费用（美元）", 0.0, 10000.0, 2000.0)

# 分类特征
contract = st.sidebar.selectbox("合同类型", ["Month-to-month", "One year", "Two year"])

# 电话服务相关
multiple_lines = st.sidebar.selectbox("是否有多条线路", ["No phone service", "No", "Yes"])

# 互联网服务相关
internet_service = st.sidebar.selectbox("互联网服务类型", ["No", "DSL", "Fiber optic"])
online_security = st.sidebar.selectbox("是否有在线安全服务", ["No internet service", "No", "Yes"])
online_backup = st.sidebar.selectbox("是否有在线备份服务", ["No internet service", "No", "Yes"])
device_protection = st.sidebar.selectbox("是否有设备保护服务", ["No internet service", "No", "Yes"])
tech_support = st.sidebar.selectbox("是否有技术支持服务", ["No internet service", "No", "Yes"])
streaming_tv = st.sidebar.selectbox("是否有电视流媒体服务", ["No internet service", "No", "Yes"])
streaming_movies = st.sidebar.selectbox("是否有电影流媒体服务", ["No internet service", "No", "Yes"])

# 支付方式
payment_method = st.sidebar.selectbox("支付方式", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# 主区域：显示结果
if st.sidebar.button("预测流失风险"):
    # 1. 组装输入
    input_dict = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # 2. 加载预测器（带缓存，避免每次点击都重复加载）
    @st.cache_resource
    def load_predictor():
        return ChurnPredictor()
    predictor = load_predictor()
    
    # 3. 预测
    with st.spinner('正在分析客户数据...'):
        result = predictor.predict(input_dict)
    
    # 4. 展示结果 - 用清晰、业务化的方式
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("流失风险概率", f"{result['churn_probability']*100:.2f}%")
    with col2:
        st.metric("风险等级", result['risk_level'])
    with col3:
        st.metric("建议行动", "立即挽留" if result['churn_prediction'] else "保持观察")
    
    # 5. （高级）展示SHAP力导向图或特征重要性条形图
    # st.subheader("决策依据")
    # st.bar_chart(data=result['top_factors'])
    
    # 6. 业务建议
    st.subheader("📈 业务决策建议")
    if result['churn_prediction']:
        st.warning(f"该客户流失风险高。建议：")
        st.info("""
        1. **主动联系**：客户服务部门在24小时内进行关怀回访。
        2. **定向优惠**：提供针对{contract}客户的专属续约折扣。
        3. **根本分析**：结合该客户的特征（高月费、短在网时长），检查产品适配性。
        """)
    else:
        st.success(f"该客户当前较为稳定。建议：")
        st.info("""
        1. **维持关系**：纳入常规客户满意度调研名单。
        2. **交叉销售**：根据其使用习惯，推荐高价值附加服务。
        """)