# %%
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from collections import Counter
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
# Đọc tệp CSV
loan_df = pd.read_csv(r"C:\Users\giabao64033\Downloads\dataset - loan application 3.csv")

print(loan_df.head())

# %%
loan_df.head()

# %%
loan_df.describe()

# %%
len(loan_df[loan_df['Loan_Status']=='N'])

# %%
loan_df.info()

# %%
#1. imputer --> cac thuat toan --> trinh bay
#2. Imbalance --> cac thuat toan, phuong phap xu ly imbalance --> trinh bay
#3. Trinh bay --> EDA --> rut ra dieu j --> anh huong cua cac thuoc tinh toi output
#4. Plan --> giai quyet bai toan (trello)

# %%
sns.set(rc={'figure.figsize':(11.7,8.27)})

# Vẽ các subplot
plt.subplot(231)
sns.countplot(x="Gender", hue='Loan_Status', data=loan_df)
plt.subplot(232)
sns.countplot(x="Married", hue='Loan_Status', data=loan_df)
plt.subplot(233)
sns.countplot(x="Education", hue='Loan_Status', data=loan_df)
plt.subplot(234)
sns.countplot(x="Self_Employed", hue='Loan_Status', data=loan_df)
plt.subplot(235)
sns.countplot(x="Dependents", hue='Loan_Status', data=loan_df)
plt.subplot(236)
sns.countplot(x="Property_Area", hue='Loan_Status', data=loan_df)

# Hiển thị đồ thị
plt.show()

# %%
bins = np.linspace(loan_df.ApplicantIncome.min(), loan_df.ApplicantIncome.max(), 12)
graph = sns.FacetGrid(loan_df, col="Gender", hue="Loan_Status", palette="Set2", col_wrap=2)
graph.map(plt.hist, "ApplicantIncome", bins=bins, ec="k")
graph.axes[-1].legend()
plt.show()

# %%

bins = np.linspace(loan_df.Loan_Amount_Term.min(), loan_df.Loan_Amount_Term.max(), 12)
graph = sns.FacetGrid(loan_df, col="Gender", hue="Loan_Status", palette="Set2", col_wrap=2)
graph.map(plt.hist, "Loan_Amount_Term", bins=bins, ec="k")
graph.axes[-1].legend()


bins = np.linspace(loan_df.CoapplicantIncome.min(), loan_df.CoapplicantIncome.max(), 12)
graph = sns.FacetGrid(loan_df, col="Gender", hue="Loan_Status", palette="Set2", col_wrap=2)
graph.map(plt.hist, "CoapplicantIncome", bins=bins, ec="k")
graph.axes[-1].legend()

plt.show()


# %%
# Loại bỏ các biến phân loại khỏi DataFrame trước khi tính tương quan
numeric_data = loan_df.select_dtypes(include=['number'])


correlation_matrix = numeric_data.corr()


sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, cmap="YlOrRd")
plt.show()


# %%
mask = np.zeros_like(correlation_matrix)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7,6))
    ax = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap="YlOrRd")
plt.show()


# %%
categorical_columns = ['Gender', 'Married', 'Dependents',
                       'Education', 'Self_Employed', 'Property_Area',
                       'Credit_History', 'Loan_Amount_Term']

# Lựa chọn các cột không phải là số
categorical_columns = ['Gender', 'Married', 'Dependents',
                       'Education', 'Self_Employed', 'Property_Area',
                       'Credit_History', 'Loan_Amount_Term']
categorical_df = loan_df[categorical_columns]

# Lựa chọn các cột là số
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income']
numerical_df = loan_df[numerical_columns]


loan_df['ApplicantIncome'] = pd.to_numeric(loan_df['ApplicantIncome'], errors='coerce')
loan_df['CoapplicantIncome'] = pd.to_numeric(loan_df['CoapplicantIncome'], errors='coerce')


loan_df['Total_Income'] = loan_df['ApplicantIncome'] + loan_df['CoapplicantIncome']
for feature in numerical_columns:
    plt.figure(figsize=(10, 4))

    # Before log-transformation
    plt.subplot(1, 2, 1) # (row, column, index)
    plt.boxplot(loan_df[feature]) 
    plt.title(f'Before Log-Transformation: {feature}')
    plt.xlabel('Dataset')
    plt.ylabel(feature)

    # After log-transformation
    plt.subplot(1, 2, 2)
    plt.boxplot(np.log1p(loan_df[feature]))  
    plt.title(f'After Log-Transformation: {feature}')
    plt.xlabel('Dataset')
    plt.ylabel(f'Log({feature} + 1)')

    plt.tight_layout()
    plt.show()

# %%
# Phân loại dữ liệu (Các cột này không biểu thị số liệu)
categorical_columns = ['Gender', 'Married', 'Dependents',
                       'Education', 'Self_Employed', 'Property_Area',
                       'Credit_History', 'Loan_Amount_Term']

# Lựa chọn các cột không phải là số
categorical_columns = ['Gender', 'Married', 'Dependents',
                       'Education', 'Self_Employed', 'Property_Area',
                       'Credit_History', 'Loan_Amount_Term']
categorical_df = loan_df[categorical_columns]

# Lựa chọn các cột là số
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income']
numerical_df = loan_df[numerical_columns]

# Chuyển đổi cột 'ApplicantIncome' và 'CoapplicantIncome' sang dạng số nếu cần
loan_df['ApplicantIncome'] = pd.to_numeric(loan_df['ApplicantIncome'], errors='coerce')
loan_df['CoapplicantIncome'] = pd.to_numeric(loan_df['CoapplicantIncome'], errors='coerce')

# Thêm cột 'Total_Income' bằng cách tính tổng của 'ApplicantIncome' và 'CoapplicantIncome'
loan_df['Total_Income'] = loan_df['ApplicantIncome'] + loan_df['CoapplicantIncome']
loan_transformed = loan_df.copy()  # Sử dụng df thay vì loan nếu bạn đang sử dụng DataFrame df

loan_copy = loan_df.copy()  # Sử dụng df thay vì loan_copy nếu bạn đang sử dụng DataFrame df

for feature in numerical_columns:
    plt.figure(figsize=(10, 4))

    # Before log-transformation
    plt.subplot(1, 2, 1) # (row, column, index)
    plt.plot(range(len(loan_df[feature])), loan_df[feature].values)
    plt.title(f'Before Log-Transformation: {feature}')
    plt.xlabel('Dataset')
    plt.ylabel(feature)

    # After log-transformation
    plt.subplot(1, 2, 2)
    loan_copy['Total_Income'] = np.log1p(loan_copy['Total_Income'])
    plt.plot(range(len(loan_copy['Total_Income'])), loan_copy['Total_Income'].values)
    plt.title('Log-Transformed Total_Income')
    plt.xlabel('Dataset')
    plt.ylabel('Log(Total_Income + 1)')
    plt.tight_layout()  # Adjust layout to prevent overlapping
    
    plt.show()
    

# %%
missing_values = loan_df.isnull().sum()
print(missing_values)
percentage_missing = (loan_df.isnull().sum() / len(loan_df)) * 100
print(percentage_missing)

# Drop rows with any missing values
df_cleaned = loan_df.dropna()

# Fill missing values with mean
df_filled = loan_df.fillna(loan_df.mean(numeric_only=True),inplace=True)

# %%
for feature in categorical_columns:
  loan_transformed[feature] = np.where(loan_transformed[feature].isnull(),
                                       loan_transformed[feature].mode(),
                                       loan_transformed[feature])
# với những cột giá trị dạng số mà có ô bị thiếu thì sẽ fill bằng giá trị median(aka giá trị xuất hiện ở giữa(khác vs mean đấy))
for feature in numerical_columns:
  loan_transformed[feature] = np.where(loan_transformed[feature].isnull(),
                                       int(loan_transformed[feature].median()),
                                       loan_transformed[feature])

# %%
loan_transformed.isnull().sum()

# %%
target= 'Loan_Status'
loan_transformed[target] = np.where(loan_transformed[target] == 'Y', 1, 0)

# %%
loan_transformed = pd.get_dummies(loan_transformed, drop_first = True)
loan_transformed.head()

# %%
loan_df.describe()

# %%
!pip3 install -U imbalanced-learn

# %%
df = pd.read_csv(r"C:\Users\giabao64033\Downloads\dataset - loan application 3.csv")


# %%
print(X.dtypes)


# %%
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r"C:\Users\giabao64033\Downloads\dataset - loan application 3.csv")
categorical_columns = ['Loan_ID', 'Married','Dependents', 'Loan_Amount_Term', 'Gender','Education','Property_Area','Self_Employed','Total_Income']


# %%
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))

# Lấy tên của các đặc trưng đã mã hóa
feature_names = encoder.get_feature_names_out(categorical_columns)

# Gán lại tên cho các cột sau khi mã hóa
X_encoded.columns = feature_names

# Loại bỏ các cột ban đầu chứa biến hạng mục
df = df.drop(categorical_columns, axis=1)

# Kết hợp DataFrame mới đã mã hóa với DataFrame ban đầu
df = pd.concat([df, X_encoded], axis=1)


# %%
# Xem xét trạng thái thiếu dữ liệu trong DataFrame
missing_values = df.isnull().sum()

# Lọc ra các cột chứa giá trị thiếu
columns_with_missing = missing_values[missing_values > 0].index.tolist()
print("Các cột chứa giá trị thiếu:", columns_with_missing)


# %%
from sklearn.impute import SimpleImputer

# Xác định các cột chứa giá trị thiếu
missing_values = df.isnull().sum()
columns_with_missing = missing_values[missing_values > 0].index.tolist()

# Sử dụng SimpleImputer để điền giá trị thiếu bằng mode của cột
imputer = SimpleImputer(strategy='most_frequent')
for col in columns_with_missing:
    df[col] = imputer.fit_transform(df[[col]])

# Tạo DataFrame mới sau khi điền giá trị thiếu
X_imputed = df.copy()

# Xác định X và y
X = X_imputed.drop('Loan_Status', axis=1)
y = X_imputed['Loan_Status']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# %%
from sklearn.ensemble import RandomForestClassifier

# Khởi tạo mô hình RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)


# %%
from sklearn.metrics import accuracy_score, confusion_matrix

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Tạo Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# %%
from sklearn.metrics import classification_report

# Tạo báo cáo đánh giá
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)



