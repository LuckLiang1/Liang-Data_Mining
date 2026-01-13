import os
import joblib  # 或 pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ChurnPredictor:
    def __init__(self, model_path=None):
        """加载模型和必要的预处理对象（如特征编码器）"""
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_model.pkl')
        self.model = joblib.load(model_path)
        # 创建一个LabelEncoder字典来保存每个特征的编码器
        self.label_encoders = {}
        
        # 定义分类特征及其可能的值（与训练数据一致）
        self.categorical_features = {
            'gender': ['Female', 'Male'],
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
        for feature, values in self.categorical_features.items():
            le = LabelEncoder()
            le.fit(values)
            self.label_encoders[feature] = le

    def _preprocess(self, input_df):
        """对输入数据进行与训练时一致的预处理"""
        # 复制输入数据以避免修改原始数据
        df = input_df.copy()
        
        # 对分类特征进行编码
        for feature in self.categorical_features.keys():
            if feature in df.columns:
                df[feature] = self.label_encoders[feature].transform(df[feature])
        
        # 将TotalCharges转换为数值型
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        return df

    def predict(self, input_data: dict) -> dict:
        """
        核心预测函数。
        输入：一个字典，包含模型所需的所有特征（如：{'tenure': 12, 'MonthlyCharges': 79.99...}
        输出：包含预测结果、概率及业务解读的字典
        """
        # 1. 将输入字典转换为DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 2. 进行与训练时完全一致的特征工程/转换
        processed_df = self._preprocess(input_df)
        
        # 3. 预测
        probability = self.model.predict_proba(processed_df)[0][1]  # 假设索引1是“流失”类
        prediction = self.model.predict(processed_df)[0]
        
        # 4. （高级）生成解释 - 使用SHAP
        # import shap
        # explainer = shap.TreeExplainer(self.model)
        # shap_values = explainer.shap_values(processed_df)
        # 将最重要的特征及影响值整合到结果中

        # 5. 返回结构化结果
        return {
            'churn_prediction': bool(prediction),
            'churn_probability': round(float(probability), 4),
            'risk_level': '高风险' if probability > 0.7 else '中风险' if probability > 0.4 else '低风险',
            # 'top_factors': [{'feature': 'tenure', 'impact': -0.05}, ...] # 可加入SHAP解释
        }