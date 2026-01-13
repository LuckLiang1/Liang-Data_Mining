import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ChurnPredictor:
    def __init__(self, model_path=None):
        """加载优化后的XGBoost模型和所有编码器"""
        if model_path is None:
            # 指向你新保存的XGBoost模型
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'optimized_xgb_churn_model.pkl')
        
        # 加载模型
        self.model = joblib.load(model_path)
        
        # 【关键】业务决策阈值
        self.BUSINESS_THRESHOLD = 0.45
        
        # 【核心修复】为每个分类特征创建并配置LabelEncoder
        # 这些编码必须与训练时完全一致！
        self.label_encoders = {}
        
        # 定义每个分类特征的类别顺序（必须与训练时相同）
        categorical_features_config = {
            'gender': ['Female', 'Male'],
            'SeniorCitizen': [0, 1],  # 注意：已经是数值，不需要编码，但要包含在特征中
            'Partner': ['No', 'Yes'],
            'Dependents': ['No', 'Yes'],
            'PhoneService': ['No', 'Yes'],
            'MultipleLines': ['No phone service', 'No', 'Yes'],
            'InternetService': ['No', 'DSL', 'Fiber optic'],
            'OnlineSecurity': ['No internet service', 'No', 'Yes'],
            'OnlineBackup': ['No internet service', 'No', 'Yes'],
            'DeviceProtection': ['No internet service', 'No', 'Yes'],
            'TechSupport': ['No internet service', 'No', 'Yes'],
            'StreamingTV': ['No internet service', 'No', 'Yes'],
            'StreamingMovies': ['No internet service', 'No', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaperlessBilling': ['No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
        }
        
        # 初始化每个分类特征的LabelEncoder
        for feature, categories in categorical_features_config.items():
            le = LabelEncoder()
            le.fit(categories)  # 用与训练时相同的类别顺序进行fit
            self.label_encoders[feature] = le
        
        # 定义特征默认值（当用户未提供时使用）
        self.feature_defaults = {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 70.0,
            'TotalCharges': 2000.0
        }
        
        # 记录训练时的特征顺序（必须与你的训练数据列顺序完全一致！）
        # 这个顺序可以从训练代码的 X_train.columns 获取
        self.expected_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        # 核心特征（用于界面输入和解释）
        self.core_features = {
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'OnlineSecurity': ['No', 'Yes', 'No internet service'],
            'TechSupport': ['No', 'Yes', 'No internet service'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'tenure': [0, 72]  # 范围
        }

    def _encode_categorical_features(self, input_dict):
        """将分类特征编码为数值（与训练时相同的编码）"""
        encoded_dict = {}
        
        for feature, value in input_dict.items():
            if feature in self.label_encoders:
                # 分类特征：使用LabelEncoder编码
                try:
                    # 确保值在编码器的已知类别中
                    if str(value) in self.label_encoders[feature].classes_:
                        encoded_value = self.label_encoders[feature].transform([str(value)])[0]
                    else:
                        # 如果是不在训练数据中的新类别，使用最常见的类别（索引0）
                        encoded_value = 0
                except Exception as e:
                    print(f"警告：编码特征 {feature} 时出错: {e}")
                    encoded_value = 0
                encoded_dict[feature] = encoded_value
            elif feature in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
                # 数值特征：直接转换为float/int
                try:
                    encoded_dict[feature] = float(value)
                except:
                    encoded_dict[feature] = 0.0
            else:
                # 其他特征（不应该有这种情况）
                encoded_dict[feature] = value
        
        return encoded_dict

    def _prepare_input_dataframe(self, input_dict):
        """准备模型所需的DataFrame（确保正确的特征顺序和类型）"""
        # 1. 用默认值填充缺失的特征
        complete_input = self.feature_defaults.copy()
        complete_input.update(input_dict)
        
        # 2. 编码所有分类特征
        encoded_input = self._encode_categorical_features(complete_input)
        
        # 3. 创建DataFrame（单行）
        input_df = pd.DataFrame([encoded_input])
        
        # 4. 确保所有特征都存在且顺序正确
        for col in self.expected_columns:
            if col not in input_df.columns:
                # 如果特征缺失，使用默认编码值
                if col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                    input_df[col] = 1  # 默认"Yes"的编码
                elif col in ['SeniorCitizen', 'MultipleLines', 'OnlineBackup', 
                           'DeviceProtection', 'StreamingTV', 'StreamingMovies']:
                    input_df[col] = 0  # 默认"No"的编码
                elif col == 'PaymentMethod':
                    input_df[col] = 0  # 默认"Electronic check"
                elif col in ['MonthlyCharges', 'TotalCharges']:
                    input_df[col] = self.feature_defaults[col]
                elif col == 'tenure':
                    input_df[col] = 12
                else:
                    input_df[col] = 0
        
        # 5. 按训练时的特征顺序排列
        input_df = input_df[self.expected_columns]
        
        # 6. 确保数值特征为正确的数据类型
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        for col in numerical_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                # 填充可能的NaN值
                input_df[col] = input_df[col].fillna(self.feature_defaults.get(col, 0))
        
        return input_df, complete_input

    def predict(self, input_data: dict) -> dict:
        """
        核心预测函数。
        输入：一个字典，至少包含核心特征
        输出：包含预测详情和业务解读的字典
        """
        # 1. 准备输入数据
        processed_df, original_features = self._prepare_input_dataframe(input_data)
        
        # 2. 调试：显示处理后的数据（测试时可开启）
        # print("处理后的特征值:")
        # print(processed_df.iloc[0].to_dict())
        
        # 3. 预测概率
        churn_probability = self.model.predict_proba(processed_df)[0][1]  # 流失类的概率
        
        # 4. 使用业务阈值决策
        churn_prediction = churn_probability >= self.BUSINESS_THRESHOLD
        
        # 5. 风险分级与业务解读
        risk_info = self._assess_risk(churn_probability, original_features)
        
        return {
            'churn_prediction': bool(churn_prediction),
            'churn_probability': round(float(churn_probability), 4),
            **risk_info,
            'threshold_used': self.BUSINESS_THRESHOLD
        }
    
    def _assess_risk(self, prob, features):
        """根据概率和特征组合，生成丰富的风险解读"""
        # 风险等级
        if prob >= 0.7:
            level, action, color = "⚡ 紧急高风险", "立即电话挽留", "#dc3545"
        elif prob >= 0.5:
            level, action, color = "⚠️ 高风险", "24小时专员介入", "#fd7e14"
        elif prob >= 0.35:
            level, action, color = "📈 中风险", "推送定向优惠", "#ffc107"
        elif prob >= 0.2:
            level, action, color = "📉 关注中", "下次营销时重点关怀", "#0dcaf0"
        else:
            level, action, color = "✅ 低风险", "常规维护与交叉销售", "#198754"
        
        # 识别关键风险驱动因素
        risk_factors = []
        factor_details = []
        
        if features.get('Contract') == 'Month-to-month':
            risk_factors.append("月度合同")
            factor_details.append("**月度合同**客户的流失率是年度合同的**3-5倍**。这是最大的风险点。")
        
        if features.get('OnlineSecurity') == 'No' and features.get('InternetService') != 'No':
            risk_factors.append("无在线安全")
            factor_details.append("**未订阅在线安全服务**，表明对增值服务付费意愿低，或对安全感知不足。")
        
        if features.get('TechSupport') == 'No' and features.get('InternetService') != 'No':
            risk_factors.append("无技术支持")
            factor_details.append("**缺少技术支持**，遇到问题时容易不满且无处求助，加速流失。")
        
        if features.get('InternetService') == 'Fiber optic':
            risk_factors.append("光纤用户")
            factor_details.append("**光纤用户**对网络质量期望极高，且市场竞争激烈，容易被竞争对手以更高带宽吸引。")
        
        tenure_val = features.get('tenure', 12)
        if isinstance(tenure_val, str):
            try:
                tenure_val = float(tenure_val)
            except:
                tenure_val = 12
        
        if tenure_val < 12:
            risk_factors.append("新客户")
            factor_details.append(f"**在网仅{tenure_val}个月**，处于磨合期，忠诚度尚未建立。")
        
        # 预估客户生命周期价值损失
        ltv_loss = 0
        monthly_charges = features.get('MonthlyCharges', 70)
        if isinstance(monthly_charges, str):
            try:
                monthly_charges = float(monthly_charges)
            except:
                monthly_charges = 70
        
        if prob > 0.35:  # 中风险以上
            ltv_loss = monthly_charges * 12 * prob
        
        return {
            'risk_level': level,
            'recommended_action': action,
            'risk_color': color,
            'predicted_ltv_loss': round(ltv_loss, 2),
            'key_risk_factors': risk_factors,
            'risk_factor_details': factor_details,
            'key_features': {
                'Contract': features.get('Contract', 'Unknown'),
                'tenure': features.get('tenure', 12),
                'InternetService': features.get('InternetService', 'Unknown'),
                'OnlineSecurity': features.get('OnlineSecurity', 'Unknown'),
                'TechSupport': features.get('TechSupport', 'Unknown')
            }
        }

    def get_estimated_ltv(self, tenure):
        """
        根据在网时长估算客户终身价值
        简单估算：平均月费 $70 * 预计剩余生命周期
        """
        # 基础假设：平均月费
        monthly_fee = 70
        
        # 根据合同类型调整（这里简化处理，实际应该基于更多特征）
        # 简单估算：预计剩余生命周期 = 72 - tenure（最大72个月）
        remaining_months = max(1, 72 - tenure)
        
        # 计算LTV
        estimated_ltv = monthly_fee * remaining_months
        
        return int(estimated_ltv)