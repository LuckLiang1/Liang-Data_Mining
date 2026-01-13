import streamlit as st
import pandas as pd
from model_loader import ChurnPredictor

# ------------------------- 1. 页面基础配置 -------------------------
st.set_page_config(page_title="客户流失风险预测系统", layout="wide")
st.title("📊 客户流失风险预测系统")
st.markdown("""
**基于XGBoost模型 & 特征重要性分析**  
本系统仅需**5个核心特征**（贡献超80%预测力），即可精准评估流失风险。
""")

# ------------------------- 2. 初始化预测器（关键步骤！）---------------------
# 使用缓存，避免每次交互都重新加载模型
@st.cache_resource
def load_predictor():
    return ChurnPredictor()

# 初始化 predictor 变量（务必在侧边栏和主逻辑之前执行）
try:
    predictor = load_predictor()
    # 可选：在侧边栏顶部显示一个成功加载的小提示
    # st.sidebar.success('模型加载成功！')
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()  # 如果模型加载失败，停止执行

# ------------------------- 3. 回调函数定义 -------------------------
# 定义设置高风险客户的回调函数
def set_high_risk():
    st.session_state.contract = "Month-to-month"
    st.session_state.tenure = 3
    st.session_state.internet_service = "Fiber optic"
    st.session_state.online_security = "No"
    st.session_state.tech_support = "No"

# 定义设置忠诚客户的回调函数
def set_loyal():
    st.session_state.contract = "Two year"
    st.session_state.tenure = 48
    st.session_state.internet_service = "DSL"
    st.session_state.online_security = "Yes"
    st.session_state.tech_support = "Yes"

# ------------------------- 4. 侧边栏：用户输入 -------------------------
st.sidebar.header("🔍 输入核心风险特征 (Top 5)")

# 初始化session_state中的值（如果尚未设置）
if 'contract' not in st.session_state:
    st.session_state.contract = "Month-to-month"
if 'tenure' not in st.session_state:
    st.session_state.tenure = 12
if 'internet_service' not in st.session_state:
    st.session_state.internet_service = "Fiber optic"
if 'online_security' not in st.session_state:
    st.session_state.online_security = "No"
if 'tech_support' not in st.session_state:
    st.session_state.tech_support = "No"

# 核心特征输入控件，其值关联到session_state
# 1. 合同类型
contract = st.sidebar.selectbox(
    "1. 合同类型 *",
    ["Month-to-month", "One year", "Two year"],
    key="contract",  # 关键：设置key参数，使用session_state中的值
    help="最关键的指标！月度合同客户的流失率是长期合同的3-5倍。"
)

# 2. 在网时长
tenure = st.sidebar.slider(
    "2. 在网时长 (月) *",
    min_value=0,
    max_value=72,
    key="tenure",  # 关键：设置key参数，使用session_state中的值
    help="新客户（<12个月）流失风险显著更高，处于不稳定期。"
)

# 3. 互联网服务类型
internet_service = st.sidebar.selectbox(
    "3. 互联网服务类型 *",
    ["Fiber optic", "DSL", "No"],
    key="internet_service",  # 关键：设置key参数，使用session_state中的值
    help="光纤用户对质量要求高且市场竞争激烈，DSL用户相对稳定。"
)

# 4. 在线安全服务
online_security = st.sidebar.selectbox(
    "4. 在线安全服务 *",
    ["No", "Yes", "No internet service"],
    key="online_security",  # 关键：设置key参数，使用session_state中的值
    help="未订阅此服务的客户，可能对增值服务付费意愿低或安全意识不足。"
)

# 5. 技术支持服务
tech_support = st.sidebar.selectbox(
    "5. 技术支持服务 *",
    ["No", "Yes", "No internet service"],
    key="tech_support",  # 关键：设置key参数，使用session_state中的值
    help="缺乏技术支持，遇到问题时容易产生不满并导致流失。"
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 仅需以上5项，系统即可生成80%以上的预测准确度。")

# 一键填充按钮（使用回调函数）
st.sidebar.markdown("**快速体验:**")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🚨 高风险客户", use_container_width=True, on_click=set_high_risk):
        pass  # 回调函数会处理状态更新

with col2:
    if st.button("✅ 忠诚客户", use_container_width=True, on_click=set_loyal):
        pass  # 回调函数会处理状态更新

# 添加颜色变化提示
if 'contract' in st.session_state:
    current_contract = st.session_state.contract
    if current_contract == "Month-to-month":
        st.sidebar.warning("⚠️ 当前设置为高风险客户配置")
    elif current_contract == "Two year":
        st.sidebar.success("✅ 当前设置为忠诚客户配置")

st.sidebar.markdown("---")
predict_button = st.sidebar.button("📊 开始风险评估", type="primary", use_container_width=True)

# 初始化session_state中的预测状态
if 'has_predicted' not in st.session_state:
    st.session_state.has_predicted = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# ------------------------- 5. 主区域：结果显示逻辑 -------------------------
# 只有当用户点击预测按钮时才执行预测和展示结果
if predict_button or st.session_state.has_predicted:
    # 组装输入数据（使用最新的session_state值）
    input_dict = {
        'Contract': st.session_state['contract'],
        'tenure': st.session_state['tenure'],
        'InternetService': st.session_state['internet_service'],
        'OnlineSecurity': st.session_state['online_security'],
        'TechSupport': st.session_state['tech_support']
    }

    # 进行预测（此时predictor变量已定义）
    if predict_button:  # 只有点击按钮时才重新预测
        with st.spinner('正在分析核心风险因素...'):
            st.session_state.prediction_result = predictor.predict(input_dict)  # 这里应该不再报错
            st.session_state.has_predicted = True
    
    # 获取最新的预测结果
    result = st.session_state.prediction_result

    # --- 1. 风险总览仪表板 ---
    st.subheader("📈 风险评估总览")
    with st.container():
        # 第一行：概率和等级并排
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 流失概率卡片 - 改进版
            st.markdown("### 流失概率")
            
            # 创建更清晰的进度条显示
            churn_percentage = result['churn_probability'] * 100
            
            # 使用columns创建更好的布局
            prob_col1, prob_col2 = st.columns([3, 1])
            
            with prob_col1:
                # 主进度条
                st.progress(
                    float(result['churn_probability']),
                    text=f"{churn_percentage:.1f}%"
                )
            
            with prob_col2:
                # 概率数值突出显示
                st.markdown(f"""
                <div style="text-align: center; padding: 8px; background: {'#ffebee' if churn_percentage > 50 else '#fff3e0' if churn_percentage > 20 else '#e8f5e9'}; 
                            border-radius: 8px; border: 2px solid {'#f44336' if churn_percentage > 50 else '#ff9800' if churn_percentage > 20 else '#4caf50'};">
                    <div style="font-size: 20px; font-weight: bold; color: {'#d32f2f' if churn_percentage > 50 else '#f57c00' if churn_percentage > 20 else '#388e3c'}">
                        {churn_percentage:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # 概率解读 - 使用不同的emoji和颜色
            if churn_percentage < 20:
                st.success("✅ 低风险：客户状态稳定")
            elif churn_percentage < 50:
                st.warning("⚠️ 中等风险：需要关注")
            else:
                st.error("🚨 高风险：急需干预")
        
        with col2:
            # 风险等级卡片
            st.markdown("### 风险等级")
            
            # 根据风险级别使用不同样式
            risk_config = {
                "紧急高风险": {"icon": "🚨", "color": "#d32f2f", "bg_color": "#ffebee"},
                "高风险": {"icon": "⚠️", "color": "#f57c00", "bg_color": "#fff3e0"},
                "关注中": {"icon": "🔍", "color": "#ffb300", "bg_color": "#fff8e1"},
                "中等风险": {"icon": "⚠️", "color": "#ffb300", "bg_color": "#fff3e0"},
                "低风险": {"icon": "✅", "color": "#388e3c", "bg_color": "#e8f5e9"}
            }
            
            risk_level = result['risk_level']
            config = risk_config.get(risk_level, {"icon": "📊", "color": "#1976d2", "bg_color": "#e3f2fd"})
            
            risk_html = f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; 
                        background: {config['bg_color']}; 
                        border: 2px solid {config['color']}; 
                        margin: 5px 0;">
                <div style="font-size: 32px; margin-bottom: 10px;">{config['icon']}</div>
                <div style="font-size: 22px; font-weight: bold; color: {config['color']};">
                    {risk_level}
                </div>
            </div>
            """
            st.markdown(risk_html, unsafe_allow_html=True)
        
        # 第二行：建议行动和潜在损失
        st.markdown("---")
        col3, col4 = st.columns([2, 1])
        
        with col3:
            # 建议行动卡片
            st.markdown("### 建议行动")
            
            # 根据风险级别确定行动紧迫性
            if churn_percentage > 50:
                urgency_icon = "⏰ 立即行动"
                urgency_color = "#d32f2f"
            elif churn_percentage > 20:
                urgency_icon = "📅 本周安排"
                urgency_color = "#f57c00"
            else:
                urgency_icon = "📋 常规维护"
                urgency_color = "#388e3c"
            
            action_html = f"""
            <div style="background: #e3f2fd; border-radius: 8px; padding: 16px; margin: 8px 0; 
                        border-left: 5px solid #2196f3;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 20px; margin-right: 10px;">🎯</span>
                    <div>
                        <div style="font-size: 18px; font-weight: bold; color: #1976d2;">
                            {result['recommended_action']}
                        </div>
                        <div style="font-size: 14px; color: #666; margin-top: 4px;">
                            <span style="color: {urgency_color}; font-weight: bold;">{urgency_icon}</span>
                        </div>
                    </div>
                </div>
            </div>
            """
            st.markdown(action_html, unsafe_allow_html=True)
            
            # 如果是高风险，添加紧迫性提示
            if churn_percentage > 50:
                st.error("⏰ 建议在24小时内采取行动")
            elif churn_percentage > 20:
                st.warning("📅 建议在本周内安排跟进")
        
        with col4:
            # 潜在损失/客户价值卡片
            if result['predicted_ltv_loss'] > 0:
                st.markdown("### 潜在损失")
                
                loss_value = result['predicted_ltv_loss']
                loss_html = f"""
                <div style="background: linear-gradient(135deg, #ffebee, #ffcdd2); 
                            border-radius: 8px; padding: 18px; margin: 8px 0; 
                            text-align: center; border: 2px solid #f44336;">
                    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
                        <span style="font-size: 24px; margin-right: 10px;">💸</span>
                        <div>
                            <div style="font-size: 28px; font-weight: bold; color: #d32f2f;">
                                ${loss_value:,.2f}
                            </div>
                            <div style="font-size: 14px; color: #666; margin-top: 4px;">
                                预计LTV损失
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #999; margin-top: 8px;">
                        基于客户历史数据和行业基准估算
                    </div>
                </div>
                """
                st.markdown(loss_html, unsafe_allow_html=True)
            else:
                st.markdown("### 客户价值")
                
                # 估算客户价值
                try:
                    ltv_value = predictor.get_estimated_ltv(st.session_state['tenure'])
                    value_color = "#388e3c" if ltv_value > 1000 else "#ff9800"
                    value_icon = "💰" if ltv_value > 1000 else "💎"
                    
                    value_html = f"""
                    <div style="background: linear-gradient(135deg, #e8f5e9, #c8e6c9); 
                                border-radius: 8px; padding: 18px; margin: 8px 0; 
                                text-align: center; border: 2px solid #4caf50;">
                        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
                            <span style="font-size: 24px; margin-right: 10px;">{value_icon}</span>
                            <div>
                                <div style="font-size: 28px; font-weight: bold; color: {value_color};">
                                    ${ltv_value:,}
                                </div>
                                <div style="font-size: 14px; color: #666; margin-top: 4px;">
                                    预计客户终身价值
                                </div>
                            </div>
                        </div>
                        <div style="font-size: 12px; color: #999; margin-top: 8px;">
                            基于在网时长和平均消费估算
                        </div>
                    </div>
                    """
                    st.markdown(value_html, unsafe_allow_html=True)
                except:
                    st.info("💰 客户价值估算暂不可用")

    # 添加分隔线
    st.markdown("---")

    # --- 2. 深度风险解读 ---
    st.subheader("🔍 风险根源分析")
    if result['key_risk_factors']:
        st.warning(f"**识别到 {len(result['key_risk_factors'])} 个高风险特征：**")
        for detail in result['risk_factor_details']:
            st.markdown(f"- {detail}")

        # 特征当前值展示 - 修复Arrow序列化问题
        # 确保所有值都是字符串类型
        key_features_data = result['key_features']
        
        # 转换所有值为字符串
        display_data = {
            '特征': list(key_features_data.keys()),
            '当前值': [str(value) for value in key_features_data.values()]
        }
        
        importance_data = pd.DataFrame(display_data)
        
        # 使用st.table代替st.dataframe，更稳定
        st.markdown("**核心特征当前值：**")
        st.table(importance_data)

    # --- 3. 动态挽留策略生成 ---
    st.subheader("🎯 个性化挽留策略建议")

    if result['key_risk_factors']:
        # 使用expander展开面板，提供更多空间
        with st.expander("📋 查看详细挽留策略", expanded=True):
            strategy = []
            
            if "月度合同" in result['key_risk_factors']:
                strategy.append("""
                **🔄 合约升级计划**
                - **目标**：将月度合同转为长期合约
                - **行动**：提供"转年约享8折"专属优惠，并赠送1个月**高级技术支持**服务
                - **预期效果**：可将流失概率降低**30-40%**
                """)
            
            if "无在线安全" in result['key_risk_factors'] or "无技术支持" in result['key_risk_factors']:
                strategy.append("""
                **🛡️ 增值服务体验计划**
                - **目标**：让客户体验增值服务价值
                - **行动**：免费开通**30天**全功能套餐（含在线安全+技术支持），到期后按5折续费
                - **预期效果**：提升粘性，流失概率降低**20-25%**
                """)
            
            if "光纤用户" in result['key_risk_factors']:
                strategy.append("""
                **📶 服务质量保障计划**
                - **目标**：巩固高端用户满意度
                - **行动**：进行网络质量专项检测，提供月度用量报告，优先处理故障
                - **预期效果**：提升感知价值，流失概率降低**15-20%**
                """)
            
            # 逐条显示策略，每条之间加分隔线
            for i, s in enumerate(strategy):
                st.markdown(s)
                if i < len(strategy) - 1:
                    st.markdown("---")
    else:
        st.success("该客户无明显高风险特征，建议常规维护与交叉销售。")
        st.info("""
        **常规维护建议：**
        1. **满意度调研**：将其纳入下季度满意度调研名单
        2. **交叉销售**：根据其使用习惯，推荐高价值附加服务
        3. **忠诚度计划**：邀请加入忠诚度计划，累积积分兑换礼品
        """)

    # --- 4. 模拟干预效果分析（可选）---
    st.subheader("🤔 模拟不同挽留策略的效果")

    # 初始化session_state中的干预选择
    if 'intervention' not in st.session_state:
        st.session_state.intervention = "无行动"
    
    # 定义重置回调函数
    def reset_intervention():
        st.session_state.intervention = "无行动"

    # 定义详细的干预策略
    intervention_strategies = {
        "无行动": {
            "desc": "不采取任何特殊行动",
            "cost": "$0",
            "reduction": 0.0,
            "details": ["无额外成本", "保持现状"],
            "recommend_for": "风险极低 (<10%) 或挽留成本高于潜在损失的客户"
        },
        "轻度干预": {
            "desc": "发送优惠券/促销邮件",
            "cost": "$5-10",
            "reduction": 0.15,
            "details": ["发送个性化优惠券", "邮件/短信跟进", "成本较低，可大规模执行"],
            "recommend_for": "中等风险 (10-30%) 客户"
        },
        "中度干预": {
            "desc": "客户专员回访+套餐折扣",
            "cost": "$20-40",
            "reduction": 0.30,
            "details": ["客户经理电话回访", "提供15%套餐折扣", "聆听客户反馈", "解决简单问题"],
            "recommend_for": "中高风险 (30-50%) 客户"
        },
        "重度干预": {
            "desc": "免费升级+专属客户经理",
            "cost": "$50-100",
            "reduction": 0.50,
            "details": ["套餐免费升级1-3个月", "分配专属客户经理", "优先技术支持", "定期满意度回访"],
            "recommend_for": "高风险 (>50%) 的VIP或高价值客户"
        }
    }

    # 创建两个主要列
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**选择干预策略:**")
        
        # 使用radio组件，绑定到session_state
        intervention = st.radio(
            "",
            list(intervention_strategies.keys()),
            format_func=lambda x: f"**{x}** - {intervention_strategies[x]['desc']}",
            key="intervention",
            label_visibility="collapsed"
        )
        
        # 显示选中策略的详细信息
        strategy = intervention_strategies[intervention]
        st.markdown("**策略详情:**")
        
        for detail in strategy['details']:
            st.markdown(f"• {detail}")
        
        st.markdown(f"**适用场景:** {strategy['recommend_for']}")
        st.caption(f"💰 预计成本: {strategy['cost']}")

    with col2:
        st.markdown("**效果模拟:**")
        
        if intervention != "无行动":
            # 计算干预后的流失概率
            original_prob = result['churn_probability']
            reduction = strategy['reduction']
            new_prob = max(0.01, original_prob * (1 - reduction))
            
            # 计算投资回报率
            ltv_loss = result['predicted_ltv_loss']
            if ltv_loss > 0:
                # 简单估算：成本取范围中值
                if "轻度" in intervention:
                    cost = 7.5
                elif "中度" in intervention:
                    cost = 30
                else:
                    cost = 75
                
                saved_value = ltv_loss * reduction
                roi = (saved_value - cost) / cost if cost > 0 else 0
                
                # 显示ROI指标
                if roi > 1:
                    st.success(f"📈 **投资回报率: {roi:.1f}x**")
                    st.caption(f"预计挽回价值: ${saved_value:.0f}, 成本: ${cost}")
                elif roi > 0:
                    st.info(f"📊 **投资回报率: {roi:.1f}x**")
                    st.caption(f"预计挽回价值: ${saved_value:.0f}, 成本: ${cost}")
                else:
                    st.warning(f"⚠️ **投资回报率: {roi:.1f}x**")
                    st.caption("成本可能高于挽回价值")
            
            # 创建对比图表
            prob_data = pd.DataFrame({
                '场景': ['干预前', '干预后'],
                '流失概率': [original_prob * 100, new_prob * 100]
            })
            
            # 使用原生图表
            st.bar_chart(
                prob_data.set_index('场景'),
                color=["#ff6b6b"],  # 只需要一个颜色，因为只有一列数据
                height=200
            )
            
            # 显示具体数值对比
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "干预前流失率",
                    f"{original_prob*100:.1f}%"
                )
            with col_b:
                delta_percent = -reduction * 100
                st.metric(
                    "干预后流失率",
                    f"{new_prob*100:.1f}%",
                    delta=f"{delta_percent:.0f}%",
                    delta_color="inverse"
                )
            
            # 根据ROI给出建议
            if 'roi' in locals():
                if roi > 2:
                    st.success("✅ **强烈推荐**: 该策略投资回报率很高")
                elif roi > 0.5:
                    st.info("🤔 **可以考虑**: 投资回报率适中")
                else:
                    st.warning("⚠️ **谨慎考虑**: 投资回报率较低")
        
        else:
            st.info("选择一种干预策略查看效果模拟")
            
            # 显示各种策略的预期效果对比
            st.markdown("**各策略预期效果:**")
            comparison_data = []
            for name, strat in intervention_strategies.items():
                if name != "无行动":
                    new_prob = max(0.01, result['churn_probability'] * (1 - strat['reduction']))
                    comparison_data.append({
                        '策略': name,
                        '成本': strat['cost'],
                        '预期效果': f"降低{strat['reduction']*100:.0f}%",
                        '预估流失率': f"{new_prob*100:.1f}%"
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.table(df_comparison)

    # 添加一个重置按钮
    if st.button("🔄 重置模拟设置", type="secondary", on_click=reset_intervention):
        st.rerun()

    st.markdown("---")
else:
    # 用户尚未点击预测按钮时显示的引导信息
    st.info("👈 请在左侧边栏输入客户特征，然后点击 **『开始风险评估』** 按钮。")
    st.markdown("### 💡 使用提示")
    st.markdown("""
    1.  您可以直接调整左侧的5个核心特征。
    2.  可以使用 **『高风险客户』** 或 **『忠诚客户』** 按钮快速填充示例数据。
    3.  系统将基于XGBoost模型，结合超过7000个客户样本训练出的规律进行预测。
    """)

# 可选：在底部显示技术信息
with st.expander("ℹ️ 技术信息"):
    st.markdown(f"""
    - **模型类型**: XGBoost (经过网格搜索调优)
    - **召回率 (测试集)**: 87%
    - **精确率 (测试集)**: 48%
    - **核心特征数**: 5个 (从20个原始特征中筛选)
    - **决策阈值**: {predictor.BUSINESS_THRESHOLD if 'predictor' in locals() else 'N/A'}
    """)
