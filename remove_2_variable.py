import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  train_test_split
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
plt.figure(figsize=(15, 12))
sns.heatmap(df1.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation between variables')
plt.xticks(rotation=90)
plt.show()

# 5. Tạo mô hình
X = df1.drop(columns = ['Delivery_Time_min','Time_of_Day','Vehicle_Type']) 
Y = df1['Delivery_Time_min']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Áp dụng LinearRegression với sử lý đa thức polynomial
orders = [1,2,3,4,5]
lr = LinearRegression()
for i in orders:
    pr = PolynomialFeatures(degree = i)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr = pr.fit_transform(x_test)
    lr.fit(x_train_pr, y_train)
    y_pred_test = lr.predict(x_test_pr)
    print(f'Order {i}: {r2_score(y_test, y_pred_test)}')

# Order = 4 cho kết quả tốt nhất

# Tạo pipeline và StandardScaler
Input = [('scale', StandardScaler()),('polynomial',PolynomialFeatures(degree=4)), ('model', LinearRegression())]
# tạo pipeline thực hiện sacle dể ra độ quan trọng của các biến, tạo polynomial với mũ là 4
# tiếp tục tạo mô hình LinearRegression
pipe = Pipeline(Input)
pipe.fit(x_train, y_train)
y_pred_train = pipe.predict(x_train)

model = pipe.named_steps['model']
a = model.coef_
b = model.intercept_
for coef, col in zip(a, X.columns):
    print(f"{col}: {coef}")
print(f"Intercept: {b}")

# Đánh giá mô hình
y_pred_test = pipe.predict(x_test)
print(f"Train: MSR: {mean_squared_error(y_train, y_pred_train)}")
print(f"Train: R2 Score: {r2_score(y_train, y_pred_train)}")

print(f"Test: MSR: {mean_squared_error(y_test, y_pred_test)}")
print(f"Test: R2 Score: {r2_score(y_test, y_pred_test)}")
mean_r2 = np.mean(np.array([r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)]))
print(f"r2 trung bình là: {mean_r2:.2f}")
# Mô hình có khả năng cao không bị overfitting, R2 Score khá tốt
# tăng 4% chỉ số r2 so với mô hình trước

# tìm chỉ số coef và ý nghĩa của từng hệ số
from scipy import stats
for i in  X:
    coef, p_value = stats.pearsonr(df1[i], df1['Delivery_Time_min'])
    print(f"{i}: coef: {coef} \t p_value: {p_value}")