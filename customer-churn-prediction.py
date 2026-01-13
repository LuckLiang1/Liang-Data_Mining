#!/usr/bin/env python
# coding: utf-8

# 

# 1. 介绍

# <a id = "2" ></a>
# #### <b>What is Customer Churn?</b>
# <span style="font-size:16px;">  Customer churn is defined as when customers or subscribers discontinue doing business with a firm or service. </span>
# 
# <span style="font-size:16px;"> Customers in the telecom industry can choose from a variety of service providers and actively switch from one to the next. The telecommunications business has an annual churn rate of 15-25 percent in this highly competitive market.</span>
# 
# <span style="font-size:16px;"> Individualized customer retention is tough because most firms have a large number of customers and can't afford to devote much time to each of them. The costs would be too great, outweighing the additional revenue. However, if a corporation could forecast which customers are likely to leave ahead of time, it could focus customer retention efforts only on these "high risk" clients. The ultimate goal is to expand its coverage area and retrieve more
# customers loyalty. The core to succeed in this market lies in the customer itself. 
# </span>
# 
# <span style="font-size:16px;"> Customer churn is a critical metric because it is much less expensive to retain existing customers than it is to acquire new customers.</span>
# 
# <a id="churn"></a>
# <a id = "3" ></a>
# 
# <span style="font-size:16px;"><b>To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.</b></span> 
# 
# <span style="font-size:16px;"> To detect early signs of potential churn, one must first develop a holistic view of the customers and their interactions across numerous channels, including store/branch visits, product purchase histories, customer service calls, Web-based transactions, and social media interactions, to mention a few. </span> 
# 
# <span style="font-size:16px;">As a result, by addressing churn, these businesses may not only preserve their market position, but also grow and thrive. More customers they have in their network, the lower the cost of initiation and the larger the profit. As a result, the company's key focus for success is reducing client attrition and implementing effective retention strategy. </span> 
# <a id="reduce"></a>
# 
# <a id = "4" ></a>
# #### <b> Objectives</b>
# I will explore the data and try to answer some questions like:
# * What's the % of Churn Customers and customers that keep in with the active services?
# * Is there any patterns in Churn Customers based on the gender?
# * Is there any patterns/preference in Churn Customers based on the type of service provided?
# * What's the most profitable service types?
# * Which features and services are most profitable?
# * Many more questions that will arise during the analysis
# <a id="objective"></a>

# ___

# <a id = "5" ></a>
# # <span style="font-family:serif; font-size:28px;"> 2. Loading libraries and data</span>
# <a id="loading"></a>

# In[3]:


import pandas as pd
import numpy as np
#missingno模块（缺失值可视化）
import missingno as msno  
# Matplotlib 是 Python 的绘图库，可以用来绘制各种静态，动态，交互式的图表，提供多样化的输出格式。
# 通常与 NumPy 和 SciPy（Scientific Python）一起使用， 这种组合广泛用于替代 MatLab
# SciPy 包含的模块有最优化、线性代数、积分、插值、特殊函数、快速傅里叶变换、信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算
import matplotlib.pyplot as plt
# Seaborn integrates closely with Pandas data structures, 
# making it easy to work with dataframes and arrays
# 建立在 Matplotlib 基础之上的 Python 数据可视化库，专注于绘制各种统计图形
import seaborn as sns
# Plotly Express 是一个高级的Python数据可视化库，它是Plotly.py的封装，提供了一个简洁且一致的API来创建复杂的图表。
import plotly.express as px
# Plotly 是一个强大的 Python 数据可视化库，提供了丰富的图表类型和灵活的定制选项。Plotly 的图形对象（Graph Objects）模块（通常导入为 go）包含了一系列自动生成的 Python 类，这些类表示图形的各个部分
import plotly.graph_objects as go
# 绘制子图
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# StandardScaler是sklearn中的一个归一化工具，可以对每个特征维度进行去均值和方差标准化，使数据符合标准正态分布
from sklearn.preprocessing import StandardScaler
# LabelEncoder 是 sklearn 中用于类别标签编码的重要工具，能够将离散的类别型标签转换为模型可识别的数值格式
from sklearn.preprocessing import LabelEncoder
# 决策树分类器
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# 高斯朴素贝叶斯 先验概率priors
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# MLPClassifier 是一个监督学习算法，它是多层感知机（MLP）的一种，也称为人工神经网络（ANN）。MLPClassifier可以处理包括分类问题在内的多种机器学习任务。它通过学习输入和输出之间的映射关系来进行预测。
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# 极端随机树
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# XGBClassifier是基于梯度提升的机器学习算法，它可以处理缺失值、支持并行计算，并且具有内置的交叉验证功能
from xgboost import XGBClassifier
# CatBoost是一个高性能的机器学习库，它基于对称决策树（oblivious trees）作为基学习器，能够有效处理类别型特征。CatBoostClassifier是CatBoost库中用于分类问题的组件，它提供了丰富的参数用于模型的训练和优化
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# In[5]:


#loading data
# df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df=pd.read_csv(r'E:\Code\Python\customer-churn-prediction\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv') 


# ___

# <a id = "6" ></a>
# # <span style="font-family:serif; font-size:28px;"> 3. Undertanding the data</span>
# <a id = "Undertanding the data" ></a>

# Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

# In[6]:


df.head()


# **The data set includes information about:**
# * **Customers who left within the last month** – the column is called Churn
# 
# * **Services that each customer has signed up for** – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# 
# * **Customer account information** - how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# 
# * **Demographic info about customers** – gender, age range, and if they have partners and dependents
# 
# 
# * ***翻译*** - 客户ID,性别,老年公民,伴侣,家属,入网时长,电话服务,多线路,互联网服务,在线安全,在线备份,设备保护,技术支持,电视流媒体,电影流媒体,合同,无纸化账单,付款方式,每月费用,总费用,流失
# 

# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.columns.values


# In[10]:


df.dtypes


# 
# * The target the we will use to guide the exploration is **Churn**

# ***

# <a id = "7" ></a>
# # <span style="font-family:serif; font-size:28px;"> 4. Visualize missing values </span>
# <a id = "missingvalue" ></a>

# In[11]:


# Visualize missing values as a matrix
msno.matrix(df);


# > Using this matrix we can very quickly find the pattern of missingness in the dataset. 
# * From the above visualisation we can observe that it has no peculiar pattern that stands out. In fact there is no missing data.

# ***

# <a id = "8" ></a>
# # <span style="font-family:serif; font-size:28px;"> 5. Data Manipulation </span>
# <a id = "8" ></a>

# In[12]:


df = df.drop(['customerID'], axis = 1)
df.head()


# * On deep analysis, we can find some indirect missingness in our data (which can be in form of blankspaces). Let's see that!

# In[13]:


# pandas.to_numeric() 是 pandas 顶级函数，语法是
# pandas.to_numeric(arg, errors='raise', downcast=None)
# errors : 可传入 {'ignore', 'raise', 'coerce'}, 默认 'raise'，如果无法解析数据的处理方案。
# 'raise', 如果无法解析将引发异常
# 'coerce', 如果无法解析将设置为 NaN
# 'ignore', 然后无效解析将返回输入
# downcast : str, 默认 None，降级处理、向下转换。可传入值有 'integer', 'signed', 'unsigned', 或者 'float'。
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()


# * Here we see that the TotalCharges has 11 missing values. Let's check this data.

# In[14]:


df[np.isnan(df['TotalCharges'])]


# * It can also be noted that the Tenure column is 0 for these entries even though the MonthlyCharges column is not empty.
# 
# Let's see if there are any other 0 values in the tenure column.

# In[15]:


df[df['tenure'] == 0].index


# * There are no additional missing values in the Tenure column. 
# 
# Let's delete the rows with missing values in Tenure columns since there are only 11 rows and deleting them will not affect the data.

# In[16]:


df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
df[df['tenure'] == 0].index
df1=df.copy()
df1.info()
df1['TotalCharges'] = pd.to_numeric(df1.TotalCharges, errors='coerce')
df1.isnull().sum()


# > To solve the problem of missing values in TotalCharges column, I decided to fill it with the mean of TotalCharges values.

# In[17]:


# 原本的df已经删除缺失的11行，并被替换了，为啥还插值？
df.fillna(df["TotalCharges"].mean())


# In[18]:


df.isnull().sum()


# In[19]:


df["SeniorCitizen"]= df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df.head()


# In[20]:


df["InternetService"].describe(include=['object', 'bool'])


# In[21]:


numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols].describe()


# ___

# <a id = "9" ></a>
# # <span style="font-family:serif; font-size:28px;"> 6. Data Visualization </span>
# <a id = "datavisualization" ></a>

# In[22]:


g_labels = ['Male', 'Female']
c_labels = ['No', 'Yes']
# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=g_labels, values=df['gender'].value_counts(), name="Gender"),
              1, 1)
fig.add_trace(go.Pie(labels=c_labels, values=df['Churn'].value_counts(), name="Churn"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)

fig.update_layout(
    title_text="Gender and Churn Distributions",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
                 dict(text='Churn', x=0.84, y=0.5, font_size=20, showarrow=False)])
fig.show()


# * 26.6 % of customers switched to another firm.
# * Customers are 49.5 % female and 50.5 % male.

# In[23]:


df["Churn"][df["Churn"]=="No"].groupby(by=df["gender"]).count()
# df["Churn"][df["Churn"]=="No"].groupby(by=df["gender"]).count()


# In[24]:


df["Churn"][df["Churn"]=="Yes"].groupby(by=df["gender"]).count()


# In[25]:


plt.figure(figsize=(6, 6))
labels =["Churn: Yes","Churn:No"]
values = [1869,5163]
labels_gender = ["F","M","F","M"]
sizes_gender = [939,930 , 2544,2619]
colors = ['#ff6666', '#66b3ff']
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
explode = (0.3,0.3) 
explode_gender = (0.1,0.1,0.1,0.1)
textprops = {"fontsize":15}
#Plot
plt.pie(values, labels=labels,autopct='%1.1f%%',pctdistance=1.08, labeldistance=0.8,colors=colors, startangle=90,frame=True, explode=explode,radius=10, textprops =textprops, counterclock = True, )
plt.pie(sizes_gender,labels=labels_gender,colors=colors_gender,startangle=90, explode=explode_gender,radius=7, textprops =textprops, counterclock = True, )
#Draw circle
centre_circle = plt.Circle((0,0),5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)

# show plot 
 
plt.axis('equal')
plt.tight_layout()
plt.show()


# * There is negligible difference in customer percentage/ count who chnaged the service provider. Both genders behaved in similar fashion when it comes to migrating to another service provider/firm.

# In[26]:


fig = px.histogram(df, x="Churn", color="Contract", barmode="group", title="<b>Customer contract distribution<b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# * About 75% of customer with Month-to-Month Contract opted to move out as compared to 13% of customrs with One Year Contract and 3% with Two Year Contract

# In[27]:


labels = df['PaymentMethod'].unique()
values = df['PaymentMethod'].value_counts()
print(labels)
print(values)

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_layout(title_text="<b>Payment Method Distribution</b>")
fig.show()


# In[28]:


fig = px.histogram(df, x="Churn", color="PaymentMethod", title="<b>Customer Payment Method distribution w.r.t. Churn</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# * Major customers who moved out were having Electronic Check as Payment Method.
# * Customers who opted for Credit-Card automatic transfer or Bank Automatic Transfer and Mailed Check as Payment Method were less likely to move out.  

# In[29]:


df["InternetService"].unique()


# In[30]:


df[df["gender"]=="Male"][["InternetService", "Churn"]].value_counts()


# In[31]:


df[df["gender"]=="Female"][["InternetService", "Churn"]].value_counts()


# In[32]:


fig = go.Figure()

fig.add_trace(go.Bar(
  x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
       ["Female", "Male", "Female", "Male"]],
  y = [965, 992, 219, 240],
  name = 'DSL',
))

fig.add_trace(go.Bar(
  x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
       ["Female", "Male", "Female", "Male"]],
  y = [889, 910, 664, 633],
  name = 'Fiber optic',
))

fig.add_trace(go.Bar(
  x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
       ["Female", "Male", "Female", "Male"]],
  y = [690, 717, 56, 57],
  name = 'No Internet',
))

fig.update_layout(title_text="<b>Churn Distribution w.r.t. Internet Service and Gender</b>")

fig.show()


# * A lot of customers choose the Fiber optic service and it's also evident that the customers who use Fiber optic have high churn rate, this might suggest a dissatisfaction with this type of internet service.
# * Customers having DSL service are majority in number and have less churn rate compared to Fibre optic service.

# In[33]:


color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
fig = px.histogram(df, x="Churn", color="Dependents", barmode="group", title="<b>Dependents distribution</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# * Customers without dependents are more likely to churn

# In[34]:


color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
fig = px.histogram(df, x="Churn", color="Partner", barmode="group", title="<b>Chrun distribution w.r.t. Partners</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# * Customers that doesn't have partners are more likely to churn

# In[35]:


color_map = {"Yes": '#00CC96', "No": '#B6E880'}
fig = px.histogram(df, x="Churn", color="SeniorCitizen", title="<b>Chrun distribution w.r.t. Senior Citizen</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# * It can be observed that the fraction of senior citizen is very less.
# * Most of the senior citizens churn.

# In[36]:


color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
fig = px.histogram(df, x="Churn", color="OnlineSecurity", barmode="group", title="<b>Churn w.r.t Online Security</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# * Most customers churn in the absence of online security, 

# In[37]:


color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
fig = px.histogram(df, x="Churn", color="PaperlessBilling",  title="<b>Chrun distribution w.r.t. Paperless Billing</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# * Customers with Paperless Billing are most likely to churn.

# In[38]:


fig = px.histogram(df, x="Churn", color="TechSupport",barmode="group",  title="<b>Chrun distribution w.r.t. TechSupport</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
try:
    fig.show()
except Exception as e:
    print(f"Error displaying figure: {e}")   


# * Customers with no TechSupport are most likely to migrate to another service provider.

# In[39]:


color_map = {"Yes": '#00CC96', "No": '#B6E880'}
fig = px.histogram(df, x="Churn", color="PhoneService", title="<b>Chrun distribution w.r.t. Phone Service</b>", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# * Very small fraction of customers don't have a phone service and out of that, 1/3rd Customers are more likely to churn. 只有极少数客户没有电话服务，其中 1/3 的客户更有可能流失。

# In[40]:


# 核密度估计（KDE）是一种用于估计随机变量概率密度函数的非参数方法。在seaborn库中，kdeplot函数提供了一种方便的方式来可视化单变量或双变量的分布。
# 这个函数会生成一个连续的概率密度曲线，可以帮助我们理解数据的分布特征。
# 数据分布（Data Distribution） 指的是一组数据中各个值的出现频率或概率，描述了数据在数轴上的分布形态、集中趋势、离散程度等特征。
sns.set_context("paper",font_scale=1.1)
ax = sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 'No') ],
                color="Red", shade = True);
ax = sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True);
ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Monthly Charges');
ax.set_title('Distribution of monthly charges by churn');


# * Customers with higher Monthly Charges are also more likely to churn

# In[41]:


ax = sns.kdeplot(df.TotalCharges[(df["Churn"] == 'No') ],
                color="Gold", shade = True);
ax = sns.kdeplot(df.TotalCharges[(df["Churn"] == 'Yes') ],
                ax =ax, color="Green", shade= True);
ax.legend(["Not Chu0rn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Total Charges');
ax.set_title('Distribution of total charges by churn');


# In[42]:


fig = px.box(df, x='Churn', y = 'tenure')

# Update yaxis properties
fig.update_yaxes(title_text='Tenure (Months)', row=1, col=1)
# Update xaxis properties
fig.update_xaxes(title_text='Churn', row=1, col=1)

# Update size and title
fig.update_layout(autosize=True, width=750, height=600,
    title_font=dict(size=25, family='Courier'),
    title='<b>Tenure vs Churn</b>',
)

fig.show()


# * New customers are more likely to churn

# In[43]:


plt.figure(figsize=(25, 10))

# df.apply(function,axis) 遍历一行axis=1或一列axis=0(默认)
# lambda:函数式编程
# # factorize()  Example array 分类变量转换为整数编码
# arr = np.array(['b', 'b', 'a', 'c', 'b'], dtype="O")
# # Factorize the array
# codes, uniques = pd.factorize(arr)
# print("Codes:", codes) # Output: [0, 0, 1, 2, 0]
# print("Uniques:", uniques) # Output: ['b', 'a', 'c']
# corr通常是一个相关系数矩阵（如通过pandas.DataFrame.corr()计算得到），形状为(n, n)
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()   

# np.ones_like()创建一个与给定数组形状和类型相同的新数组，但新数组的所有元素都是1
# np.triu 是NumPy库中的一个函数，用于提取矩阵的上三角?部分
mask = np.triu(np.ones_like(corr, dtype=bool))
# 使用seaborn.heatmap()传入掩码，避免重复显示对称信息
ax = sns.heatmap(
    corr,                   # 相关系数矩阵（通常是DataFrame或numpy数组）
    mask=mask,              # 掩码：隐藏下三角部分（包括对角线）
    xticklabels=corr.columns,  # X轴标签使用DataFrame的列名
    yticklabels=corr.columns,  # Y轴标签使用DataFrame的列名
    annot=True,             # 在每个单元格中显示相关系数值
    linewidths=.2,          # 单元格之间的分隔线宽度
    cmap='coolwarm',        # 颜色映射：从蓝色（-1）到白色（0）到红色（+1）
    vmin=-1, vmax=1         # 颜色映射的取值范围：相关系数的理论范围是[-1, 1]
)


# ___

# <a id = "10" ></a>
# # <span style="font-family:serif; font-size:28px;"> 7. Data Preprocessing</span>
# <a id = "datapreprocessing" ></a>

# <a id = "1111" ></a>
# #### **Splitting the data into train and test sets**
# <a id = "Split" ></a>

# In[44]:


def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series) # 将字符串编码为整数
    return dataframe_series


# In[45]:


# df.apply(...)：对 DataFrame 的每一列（默认 axis=0）应用自定义函数。
df = df.apply(lambda x: object_to_int(x))  #匿名函数，其中 x 代表 DataFrame 的每一列（即一个 pandas Series）
df.head()


# In[46]:


plt.figure(figsize=(14,7))
# print(df.corr())
df.corr()
# 计算 DataFrame 中所有数值列之间的皮尔逊相关系数（默认方法），返回一个相关系数矩阵
# ['Churn']：从相关系数矩阵中提取与Churn列相关的所有系数，得到一个 Series
df.corr()['Churn'].sort_values(ascending = False)


# In[47]:


X = df.drop(columns = ['Churn'])
y = df['Churn'].values
print(df["Churn"])
print(df['Churn'].values)


# In[48]:


# stratify=y
# 分层抽样：确保训练集和测试集中目标变量 y 的类别比例与原始数据一致。
# 适用场景：处理不平衡数据集（如正类样本占 10%，负类占 90%），防止训练 / 测试集分布偏差导致模型失效。
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40, stratify=y)

# 确保在划分后再进行特征工程（如标准化、特征选择）。若在划分前处理，测试集可能 “偷看” 到训练集的统计信息。


# In[50]:


def distplot(feature, frame, color='r'):
    plt.figure(figsize=(8,3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color= color)


# 

# In[51]:


num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
for feat in num_cols: distplot(feat, df)


# Since the numerical features are distributed over different value ranges, I will use standard scalar to scale them down to the same range.

# <a id = "111" ></a>
# #### **Standardizing numeric attributes**
# <a id = "Standardizing" ></a>

# In[52]:


df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')),
                       columns=num_cols)
for feat in numerical_cols: distplot(feat, df_std, color='c')


# 

# In[53]:


# Divide the columns into 3 categories, one ofor standardisation, one for label encoding and one for one hot encoding
# 手动指定需要独热编码的列（通常是无序分类变量）:
cat_cols_ohe =['PaymentMethod', 'Contract', 'InternetService'] # those that need one-hot encoding
cat_cols_le = list(set(X_train.columns)- set(num_cols) - set(cat_cols_ohe)) #those that need label encoding
print("需要标签编码的列:",cat_cols_le)


# In[54]:


scaler= StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# <a id = "11" ></a>
# # <span style="font-family:serif; font-size:28px;"> 8. Machine Learning Model Evaluations and Predictions</span>
# <a id = "modelprediction" ></a>

# ![AI-Workbench-Predict-propensity-churn-notebook.png](attachment:8fc66a4b-838f-401e-bf6b-4577d1f313ec.png)

# <a id = "101" ></a>
# #### <b> KNN</b>
# <a id = "knn" ></a>

# In[55]:


knn_model = KNeighborsClassifier(n_neighbors = 11, n_jobs=1) 
knn_model.fit(X_train,y_train)
predicted_y = knn_model.predict(X_test)
accuracy_knn = knn_model.score(X_test,y_test)
print("KNN accuracy:",accuracy_knn)


# In[56]:


print(classification_report(y_test, predicted_y))


# <a id = "102" ></a>
# #### <b>SVC</b>
# <a id = "svc" ></a>

# In[57]:


svc_model = SVC(random_state = 1)
svc_model.fit(X_train,y_train)
predict_y = svc_model.predict(X_test)
accuracy_svc = svc_model.score(X_test,y_test)
print("SVM accuracy is :",accuracy_svc)


# In[58]:


print(classification_report(y_test, predict_y))


# <a id = "103" ></a>
# #### <b> Random Forest</b>
# <a id = "rf" ></a>

# In[63]:


model_rf = RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "sqrt",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))
accuracy_rf = model_rf.score(X_test,y_test)
print("RF accuracy is :",accuracy_rf)


# In[64]:


print(classification_report(y_test, prediction_test))


# In[65]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, prediction_test),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title(" RANDOM FOREST CONFUSION MATRIX",fontsize=14)
plt.show()


# In[66]:


y_rfpred_prob = model_rf.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_rfpred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_rf, tpr_rf, label='Random Forest',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve',fontsize=16)
plt.show();


# <a id = "104" ></a>
# #### <b>Logistic Regression</b>
# <a id = "lr" ></a>

# In[67]:


lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
accuracy_lr = lr_model.score(X_test,y_test)
print("Logistic Regression accuracy is :",accuracy_lr)


# In[68]:


lr_pred= lr_model.predict(X_test)
report = classification_report(y_test,lr_pred)
print(report)


# In[69]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, lr_pred),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("LOGISTIC REGRESSION CONFUSION MATRIX",fontsize=14)
plt.show()


# In[70]:


y_pred_prob = lr_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr, tpr, label='Logistic Regression',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve',fontsize=16)
plt.show();


# <a id = "105" ></a>
# #### **Decision Tree Classifier**
# <a id = "dtc" ></a>

# In[71]:


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
predictdt_y = dt_model.predict(X_test)
accuracy_dt = dt_model.score(X_test,y_test)
print("Decision Tree accuracy is :",accuracy_dt)


# 

# Decision tree gives very low score.

# In[72]:


print(classification_report(y_test, predictdt_y))


# <a id = "106" ></a>
# #### **AdaBoost Classifier**
# <a id = "ada" ></a>

# In[73]:


a_model = AdaBoostClassifier()
a_model.fit(X_train,y_train)
a_preds = a_model.predict(X_test)
print("AdaBoost Classifier accuracy")
metrics.accuracy_score(y_test, a_preds)


# In[74]:


print(classification_report(y_test, a_preds))


# In[75]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, a_preds),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("AdaBoost Classifier Confusion Matrix",fontsize=14)
plt.show()


# <a id = "107" ></a>
# #### **Gradient Boosting Classifier**
# <a id = "gb" ></a>

# In[76]:


gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print("Gradient Boosting Classifier", accuracy_score(y_test, gb_pred))


# In[77]:


print(classification_report(y_test, gb_pred))


# In[78]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, gb_pred),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("Gradient Boosting Classifier Confusion Matrix",fontsize=14)
plt.show()


# <a id = "108" ></a>
# #### **Voting Classifier**
# <a id = "vc" ></a>
# Let's now predict the final model based on the highest majority of voting and check it's score.

# In[89]:


from sklearn.ensemble import VotingClassifier
# GradientBoostingClassifier：梯度提升树，适合捕捉数据中的非线性关系
# LogisticRegression：逻辑回归，提供线性分类边界和概率输出
# AdaBoostClassifier：自适应提升算法，通过组合弱分类器提高整体性能
clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()  
clf3 = AdaBoostClassifier()
# estimators 参数是一个元组列表，每个元组包含 (名称，模型)
# voting='soft' 表示使用软投票机制：基于各模型的预测概率进行加权平均
# 软投票要求所有基础模型都能提供概率预测（即具有 predict_proba 方法）
eclf1 = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)
print("Final Accuracy Score ")
print(accuracy_score(y_test, predictions))


# In[80]:


print(classification_report(y_test, predictions))


# In[81]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, predictions),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("FINAL CONFUSION MATRIX",fontsize=14)
plt.show()


# From the confusion matrix we can see that: 
# There are total 1400+149=1549 actual non-churn values and the algorithm predicts 1400 of them as non churn and 149 of them as churn.
# While there are 237+324=561 actual churn values and the algorithm predicts 237 of them as non churn values and 324 of them as churn values.
# 
# 从混淆矩阵中我们可以看到 实际非流失值共有 1400+149=1549 个，算法预测其中 1400 个为非流失值，149 个为流失值。而实际流失值为 237+324=561 个，算法预测其中 237 个为非流失值，324 个为流失值。

# Customer churn is definitely bad to a firm ’s profitability. Various strategies can be implemented to eliminate customer churn. The best way to avoid customer churn is for a company to truly know its customers. This includes identifying customers who are at risk of churning and working to improve their satisfaction. Improving customer service is, of course, at the top of the priority for tackling this issue. Building customer loyalty through relevant experiences and specialized service is another strategy to reduce customer churn. Some firms survey customers who have already churned to understand their reasons for leaving in order to adopt a proactive approach to avoiding future customer churn. 

# <span style="color:crimson;font-family:serif; font-size:20px;">  Please upvote if you liked the kernel! 😀
#     <p style="color:royalblue;font-family:serif; font-size:20px;">KEEP KAGGLING!</p> 
# </span>

# In[96]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

#中文字符正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']

y_pred_prob = eclf1.predict_proba(X_test)[:,1]

print("========== 分类性能详情 ==========")
print(classification_report(y_test, predictions, target_names=['未流失', '流失']))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")

# 绘制混淆矩阵

cm=confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['预测未流失', '预测流失'],
            yticklabels=['实际未流失', '实际流失'])
ax.set_ylabel('实际')
ax.set_xlabel('预测')
ax.set_title('混淆矩阵 - 客户流失预测')
plt.show()

# 计算业务关键指标
tn, fp, fn, tp = cm.ravel()
print(f"========== 业务视角指标 ==========")
print(f"流失客户捕获率 (召回率 Recall): {tp/(tp+fn):.2%}") # 我们抓住了多少“真流失”？
print(f"预警准确率 (精确率 Precision): {tp/(tp+fp):.2%}") # 我们发出的流失预警中，有多少是对的？
print(f"误伤率 (False Alarm Rate): {fp/(fp+tn):.2%}") # 多少好客户被我们错判为流失？


# In[97]:


# 寻找最佳分类阈值，模型默认以0.5为界划分“流失”与“不流失”。我们可以降低这个阈值，让模型变得更“敏感”。
from sklearn.metrics import precision_recall_curve

# 获取测试集的预测概率（属于“流失”类的概率）
y_pred_proba = eclf1.predict_proba(X_test)[:, 1]

# 计算不同阈值下的精确率和召回率
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# 定义一个函数来寻找满足业务需求的最佳阈值
def find_best_threshold(target_recall):
    """找到第一个达到目标召回率的阈值"""
    for i, recall in enumerate(recalls):
        if recall >= target_recall:
            return thresholds[i], precisions[i], recall
    return thresholds[-1], precisions[-1], recalls[-1]

# 业务目标：我们希望至少抓住75%的流失客户（召回率>=0.75）
target_recall = 0.75
best_thresh, prec_at_thresh, rec_at_thresh = find_best_threshold(target_recall)

print(f"\n========== 阈值调优分析 ==========")
print(f"当设定召回率目标为 {target_recall:.0%} 时：")
print(f"  推荐阈值: {best_thresh:.3f}")
print(f"  对应精确率: {prec_at_thresh:.2%}")
print(f"  对应召回率: {rec_at_thresh:.2%}")

# 使用新阈值进行预测
y_pred_new = (y_pred_proba >= best_thresh).astype(int)

# 重新评估
from sklearn.metrics import classification_report
print(f"\n新阈值下的分类报告:")
print(classification_report(y_test, y_pred_new, target_names=['未流失', '流失']))

# 【重要】保存这个最佳阈值，在部署的model_loader.py中要使用
print(f"\n请记录此阈值，并更新到 model_loader.py 中: best_threshold = {best_thresh}")


# # XGBoost Classifier
# 

# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
  
# 使用更关注召回率的评估指标（F2-Score，给予召回率2倍权重于精确率）
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'scale_pos_weight': [1, 2, 3] # 此参数专门处理不平衡数据，增大该值会提升对“流失”（正类）的关注
}
# 使用召回率作为评估标准
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='recall', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"XGBoost 最佳参数: {grid_search.best_params_}")
print(f"XGBoost 最佳召回率: {grid_search.best_score_:.3f}")

# 用最佳模型预测
best_xgb = grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

print(f"\nXGBoost 测试集分类报告:")
print(classification_report(y_test, y_pred_xgb, target_names=['未流失', '流失']))


# ## 1. 调整阈值
# 

# In[59]:


# 寻找最佳分类阈值，模型默认以0.5为界划分“流失”与“不流失”。我们可以降低这个阈值，让模型变得更“敏感”。
from sklearn.metrics import precision_recall_curve

# 获取测试集的预测概率（属于“流失”类的概率）

y_pred_proba = best_xgb.predict_proba(X_test)[:, 1] #

# 计算不同阈值下的精确率和召回率
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# 定义一个函数来寻找满足业务需求的最佳阈值
def find_best_threshold(target_recall):
    """找到第一个达到目标召回率的阈值"""
    for i, recall in enumerate(recalls):
        if recall >= target_recall:
            return thresholds[i], precisions[i], recall
    return thresholds[-1], precisions[-1], recalls[-1]

# 业务目标：我们希望至少抓住75%的流失客户（召回率>=0.75）
# target_recall = 0.75
# best_thresh, prec_at_thresh, rec_at_thresh = find_best_threshold(target_recall)

print(f"\n========== 阈值调优分析 ==========")
print(f"当设定召回率目标为 {target_recall:.0%} 时：")
print(f"  推荐阈值: {best_thresh:.3f}")
print(f"  对应精确率: {prec_at_thresh:.2%}")
print(f"  对应召回率: {rec_at_thresh:.2%}")

new_thresh = 0.42
# 使用新阈值进行预测
y_pred_new = (y_pred_proba >= new_thresh).astype(int)

# 重新评估
from sklearn.metrics import classification_report
print(f"\n新阈值下的xgb分类报告:")
print(classification_report(y_test, y_pred_new, target_names=['未流失', '流失']))

# 【重要】保存这个最佳阈值，在部署的model_loader.py中要使用
print(f"\n请记录此阈值，并更新到 model_loader.py 中: best_threshold = {best_thresh}")


# ## 2. 特征重要性分析
# 

# In[53]:


import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

# 提取XGBoost模型的特征重要性
xgb_feature_importance = best_xgb.feature_importances_

# # 可视化特征重要性
# plt.figure(figsize=(10, 6))
# plt.barh(X_train.columns, xgb_feature_importance)
# plt.title('XGBoost 特征重要性')
# plt.xlabel('特征重要性分数')
# plt.ylabel('特征')
# plt.show()

features = X_train.columns
importance_df = pd.DataFrame({'feature': features, 'importance': xgb_feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)

print("========== 特征重要性 Top 10 ==========")
print(importance_df.head(10))

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.barh(importance_df.head(10)['feature'], importance_df.head(10)['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance (GradientBoosting)')
plt.gca().invert_yaxis()
plt.show()


# In[ ]:


# import pickle
# import joblib
# from tensorflow.keras.models import save_model

# 示例1：
# joblib.dump(eclf1, 'random_forest_model.joblib')  # 保存为joblib文件
# print("ok")
# # 示例2：TensorFlow/Keras模型
# model = Sequential([...])
# model.fit(X_train, y_train)
# model.save('neural_network_model.h5')  # 保存为H5文件
# 'VotingClassifier' object has no attribute 'save',eclf1软投票模型不能用这个.save

# 示例3：通用pickle方法
# pickle.dump(eclf1, open('model.pkl', 'wb'))

# import os
# 查看当前工作目录
# print(os.getcwd())  /kaggle/working
# 上传到Kaggle数据集（需在Notebook中执行）
# !mkdir my_model_dataset
# 查看创建的文件夹路径
# folder_path = os.path.join(os.getcwd(), 'my_model_dataset')
# print(folder_path)  # 输出：/kaggle/working/my_model_dataset

# # 确认文件夹是否存在
# print(os.path.exists(folder_path))  # 输出：True

# !mv *.joblib my_model_dataset/
# !mv *.h5 my_model_dataset/


# In[83]:


import joblib

# 保存模型到文件
model_path = 'voting_classifier_model.joblib'
joblib.dump(eclf1, model_path)
print(f"模型已保存到: {model_path}")

# 在部署环境中加载模型
loaded_model = joblib.load(model_path)
print("模型加载成功")

# 使用加载的模型进行预测
new_predictions = loaded_model.predict(X_test)
print(f"预测结果示例: {new_predictions[:5]}")  


# In[60]:


import joblib
model_path = 'demo/models/optimized_xgb_churn_model.pkl'  # 使用新名字
joblib.dump(best_xgb, model_path)  # best_xgb 是你的 GridSearchCV 最佳模型
print(f"优化后的XGBoost模型已保存至: {model_path}")


# In[84]:


X_test


# In[85]:


import pandas as pd
import numpy as np
import sklearn
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")

