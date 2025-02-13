import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import  KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load data
df = pd.read_csv(r'C:\Users\Admin\Documents\Python\Food_Delivery_Times.csv')
df.describe()
df.info()    

# 2. Kiểm tra dữ liệu bị thiếu
df.isnull().sum()

# Thay thế dữ liệu bị thiếu
# Traffic_Level, Time_of_Day bằng mode của chúng
# Courier_Experience_yrs bằng median
df['Traffic_Level'].fillna(df['Traffic_Level'].mode()[0], inplace=True)
df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0], inplace=True)
df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median(), inplace=True)

# Kiểm tra lại
df.isnull().sum()

# tạo ra dataFrame mới để sử dụng  
df1 = df.copy() 

# tạo file csv mới để tạo visualization trên power bi
# df1.to_csv(r'C:\Users\Admin\Documents\Python\Food_Delivery_Times1.csv', index=False)
df1.drop(columns = ['Order_ID'], inplace=True)

# 3. Vẽ biểu đồ 
bar_cols = ['Weather', 'Traffic_Level','Time_of_Day', 'Vehicle_Type','Courier_Experience_yrs']
fig, axes = plt.subplots(3, 2, figsize=(20, 30))
axes = axes.flatten()
for ax, i in zip(axes, bar_cols):
    sns.countplot(data = df1,x = i,palette = 'tab10',edgecolor = 'black',ax = ax)
    ax.set_title(f'Countplot for {i}')
    for i in ax.get_xticklabels():
        i.set_rotation(45)
plt.tight_layout()
plt.show()

# Whether thì khảo xát Clear nhiều nhất
# Mật độ xe cộ thì thường là Low và Medium
# Thời gian giao hàng phổ biến là Sáng, chiều và tối
# Kinh nghiêm của người giao hàng thì phân bố đều từ 1 đến 10 đều có vẻ same same


histplot = ['Distance_km', 'Preparation_Time_min']
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes = axes.flatten()

for ax, i in zip(axes, histplot):
    sns.histplot(data = df1,x = i,edgecolor = 'black',ax = ax)
    ax.set_title(f'Histogram for {i}')
    ax.set_xticks(np.arange(0,df1[i].max(),4))
    ax.set_xticklabels(np.arange(0,df1[i].max(),4))
plt.tight_layout()
plt.show()

# Ordinal Encoded

traffic = ['Low', 'Medium', 'High']
time = ['Morning', 'Afternoon', 'Evening', 'Night']

ord = OrdinalEncoder(categories=[traffic, time])
df1[['Traffic_Level','Time_of_Day']] = ord.fit_transform(df1[['Traffic_Level','Time_of_Day']])
print(df1.head())

# Label Encoded

weather = LabelEncoder()
vehicle = LabelEncoder()

df1['Weather'] = weather.fit_transform(df1['Weather'])
df1['Vehicle_Type'] = vehicle.fit_transform(df1['Vehicle_Type'])

weather_map = list(zip(weather.classes_, range(len(weather.classes_))))
vehicle_map = list(zip(vehicle.classes_, range(len(vehicle.classes_))))

print("Weather:")
for original, encoded in weather_map:
    print(f"{original} -> {encoded}")
print('---------------------------------')
print("Vehicle_Type:")
for original, encoded in vehicle_map:
    print(f"{original} -> {encoded}")

df1.head()

# 4. Tìm ra mối quan hệ giữa các biến

sns.heatmap(df1.corr(), cmap='coolwarm', annot=True)
plt.title('Correlation between variables')
plt.xticks(rotation=45)
plt.show()

# 5. Tạo mô hình
X = df1.drop(columns = ['Delivery_Time_min']) 
Y = df1['Delivery_Time_min']

models = [
    ('LR', LinearRegression()),
    ('Ridge', Ridge()),
    ('RF', RandomForestRegressor()),
    ('SVR', SVR()),
    ('DT', DecisionTreeRegressor()),
    ('XGB', xgb.XGBRegressor())
]

for i, model in models:
    K = KFold(n_splits=2)
    result = cross_val_score(model, X, Y, cv = K,scoring='r2')
    print(i,': ' ,result)

# LR :  [0.74920459 0.77053888]
# Ridge :  [0.74917755 0.7705803 ]
# RF :  [0.6939443  0.72632045]
# SVR :  [0.66666638 0.69836033]
# DT :  [0.38475657 0.35663278]
# XGB :  [0.66094297 0.65167105]
# -> Linear và Ridge cho kết quả tốt nhất

params = [{'alpha': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]}]
ridge = Ridge()
grid = GridSearchCV(ridge, param_grid=params, scoring='r2', cv=5)
grid.fit(X, Y)
print(grid.best_estimator_)
#  Ridge(alpha=10)

# 7.Huấn luyện mô hình
ridge_model = Ridge(alpha=10)
ridge_model.fit(X, Y)
Y_pred = ridge_model.predict(X)

r2 = r2_score(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)

print(f"R2 Score: {r2:.2f}")        #0.76 
print(f"Mean Squared Error: {mse:.2f}")     #116.08
# -> Mô hình có độ chính xác cao 76% và Mean Squared Error là 116.08
a = ridge_model.coef_
b = ridge_model.intercept_
for coef, col in zip(a, X.columns):
    print(f"{col}: {coef}")
print(f"Intercept: {b}")

# tìm chỉ số coef và ý nghĩa của từng hệ số
from scipy import stats
for i in  X:
    coef, p_value = stats.pearsonr(df1[i], df1['Delivery_Time_min'])
    print(f"{i}: coef: {coef} \t p_value: {p_value}")

# Tuy nhiên trong mô hình chỉ số Time_of_Day có corr với Delivery_Time_min âm   
# nhưng trong công thức nó lại dương
# Hai biến Time_of_Day và Vehicle_Type có p_value > 0.05 nên không ảnh hưởng đến Delivery_Time_min