import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from matplotlib import font_manager, rc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from xgboost import XGBClassifier


font_path = '/Library/Fonts/Arial Unicode.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

import warnings
warnings.filterwarnings("ignore")

# KFold evaluation
# input : X, y, k, mode
# output : None(void)
# description : KFold evaluation about each classification model and print average accuracy
def evaluate_Kfold(X, y, k, mode):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    accuracy_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = None

        if mode == 'Decision Tree':
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
            model.fit(X_train, y_train)
        elif mode == 'KNN':
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train, y_train)
        elif mode == 'Random Forest':
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
        elif mode == 'XGBoost':
            model = XGBClassifier(random_state=42)
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        accuracy_scores.append(accuracy)

    average_accuracy = sum(accuracy_scores) / k  # 평균 정확도 계산

    print(f"{mode} Average accuracy: {average_accuracy:.2f}")


# evaluation_classification
# input : y_true, y_pred
# output : None(void)
# description : print accuracy, confusion matrix, precision, recall, f1 score
def evaluation_classification(y_true, y_pred):
    # calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # calculate confusion matrix
    confusion_mtx = confusion_matrix(y_true, y_pred)

    # calculate precision
    precision = precision_score(y_true, y_pred, average='macro')  # multi-class -> average='macro'

    # calculate recall
    recall = recall_score(y_true, y_pred, average='macro')  # multi-class -> average='macro'

    # calculate f1 score
    f1 = f1_score(y_true, y_pred, average='macro')  # multi-class -> average='macro'

    # print
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_mtx)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

print("---------- 단속위치 데이터 ---------")
address = "경기도 성남시_주정차 위반 단속 위치 현황_20220927.csv"
df1 = pd.read_csv(address, encoding = 'cp949')


words_to_check = ["창곡", "양지", "은행", "산성", "단대", "금광", "상대원", "갈현", "도촌", "여수", "하대원", "성남", "수진", "신흥", "복정", "태평", "신촌", "오야", "심곡", "둔전", "고등", "시흥", "사송", "상적", "금토", "야탑", "이매", "율", "서현", "분당", "수내", "정자", "구미", "금곡", "동원", "궁내", "백현", "판교", "삼평", "하산운", "대장", "운중", "석운"]

df1['단속장소'] = df1['단속장소'].str.extract(r'(\w+[동])')


# 시군명, 관리기관명, 단속일시정보, 단속방법 column을 제거
df1 = df1.drop(['시군명', '관리기관명', '단속일시정보', '단속방법'], axis = 1).reset_index()
df1 = df1.drop(['index'], axis = 1)


#------------------- words_to_check 리스트 안에 있는 단어들이 있는 row들을 통일시킴 -------------------
pattern = r'\b(\w+동)\b'

mask = df1['단속장소'].str.contains('|'.join(words_to_check), case=False, na=False)
df1.loc[mask, '단속장소'] = df1.loc[mask, '단속장소'].str.extract(pattern, expand=False).fillna(df1.loc[mask, '단속장소'])


#------------------- words_to_check 리스트 안에 있는 단어들이 있는 row들을 해당 단어가 포함된 부분만으로 대체 ---------------
df1 = df1.dropna(how='any').reset_index()
df1 = df1.drop(['index'], axis = 1)

# calculate the number of rows in the dataframe
df_grouped = df1.groupby(['집계년도', '단속장소']).size().reset_index(name='단속횟수')

# calculate the mean of the number of rows in the dataframe
df_counts = df_grouped.groupby('단속장소')['단속횟수'].mean().reset_index(name='단속횟수')

# filter out the rows that have less than 10 counts <Outlier>
df_counts = df_counts[df_counts['단속횟수'] > 10]

# sort the rows by the number of counts in descending order
df_counts = df_counts.sort_values(by=['단속횟수'], axis=0, ascending=False).reset_index()
df_counts = df_counts.drop(['index'], axis = 1)

# create a new dataframe
df_new = pd.DataFrame({'단속장소': df_counts['단속장소'], '단속횟수': df_counts['단속횟수']})

# rename the columns
df_new = df_new.rename(columns={'단속장소': '단속동'})

# trim
df_new['단속동'] = df_new['단속동'].str.strip()

df_new.to_excel('단속장소_단속횟수_final.xlsx', index = False)

print(df_new)

print("---------- 인구수 ---------")
address = "경기도 성남시_인구및세대_현황_20230430.csv"
df_population = pd.read_csv(address, encoding = 'utf-8')
print(df_population)

# renew column
df_population = df_population[['동', '인구수_계']]

# rename column
df_population = df_population.rename(columns={'동': '단속동', '인구수_계': '인구 (명)'})

# trim
df_population['단속동'] = df_population['단속동'].str.strip()

print(df_population)

print("---------- 성남시 인구수 + 동 합치기 ---------")
# merge two dataframes
df_sn = pd.merge(df_new, df_population, how='inner')
print(df_sn)

print("---------- 서울시 동별 단속건수 ---------")
address = "서울시_동별_단속현황.xlsx"
df_seoul_data = pd.read_excel(address)
print(df_seoul_data)

print("---------- 데이터셋 merge ---------")

# rename column
df_sn = df_sn.rename(columns={'단속횟수': '단속건수'})

# merge two dataframes
df = pd.concat([df_sn, df_seoul_data], axis=0)
df = df.reset_index(drop=True)

print(df)

print("---------- 경기도 상권정보 ---------")

address = "소상공인시장진흥공단_상가(상권)정보_경기_202209.csv"
df_gyeonggi = pd.read_csv(address, encoding = 'utf-8')
print(df_gyeonggi)

# '상권업종대분류명', '행정동명', '행정동명', '동정보' column
df_gyeonggi = df_gyeonggi[['상권업종대분류명', '행정동명']]

# remove rows that have '중앙동' in '행정동명' column <Unusable Data>
df_gyeonggi = df_gyeonggi[df_gyeonggi['행정동명'] != '중앙동']

print(df_gyeonggi)

print("---------- 서울 상권정보 ---------")

address = "소상공인시장진흥공단_상가(상권)정보_서울_202209.csv"

df_seoul = pd.read_csv(address, encoding = 'utf-8')
print(df_seoul)
# '상권업종대분류명', '행정동명', '행정동명', '동정보' column
df_seoul = df_seoul[['상권업종대분류명', '행정동명']]
print(df_seoul)

print("------------------- 서울 + 경기도 상권정보 합치기 -------------------")
df_store = pd.concat([df_gyeonggi, df_seoul], axis=0)
df_store = df_store.reset_index(drop=True)



# '상권업종대분류명', '행정동명', '행정동명', '동정보' column
df_store = df_store[['상권업종대분류명', '행정동명']]

# 상권업종대분류명 column <one-hot encoding>
df_store = pd.get_dummies(df_store, columns=['상권업종대분류명'])

# groupby '행정동명' and sum
df_combined = df_store.groupby('행정동명').sum().reset_index()

# print(df_combined)

# rename column
df_combined = df_combined.rename(columns={'행정동명': '단속동'})

# merge two dataframes
df_merged = pd.merge(df, df_combined, on='단속동', how='outer')

print("---------- 결측치 제거 ---------")
# drop rows that have NaN
print(df_merged.isnull().sum())
df_merged.dropna(inplace=True)

print("------------------- MinMaxScaler -------------------")

# MinMaxScaler
minmax = MinMaxScaler()

# scale the data
columns_to_scale = ['단속건수', '인구 (명)', '단속동']
df_scaled = minmax.fit_transform(df_merged.drop(columns_to_scale, axis=1))

# reassign the scaled data to its original dataframe
df_merged[df_merged.columns.drop(columns_to_scale)] = df_scaled

print(df_merged)




print("------------------------------ Data Analysis -------------------------------")

#------------------- 학습시킬 데이터를 제작 -------------------
minmax = MinMaxScaler()
X = df_merged.drop(['단속건수', '단속동'], axis=1)
X = minmax.fit_transform(X)
y = df_merged['단속건수']


#------------------- Split data into training and validation sets -------------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

ridgeModel = Ridge()
ridgeModel.fit(X_train, y_train)

lassoModel = Lasso()
lassoModel.fit(X_train, y_train)

linearModel = LinearRegression()
linearModel.fit(X_train, y_train)

# #------------------- 사용자로부터 미래에 대한 가정 정보를 받아옴 -------------------
area = input("예상하고자 하는 지역을 적어주세요 : ")
rate_pol, year = input("인구규모 변화율, 예상할 연도를 적어주세요 : ").split()
rate_pol = int(rate_pol)
year = int(year)

#------------------- 가정을 반영한 새로운 row array를 만듦 -------------------
# 단속동이 area인 row를 찾음
predict_df = df_merged[df_merged['단속동'] == area]
print(predict_df)
area_pol = predict_df['인구 (명)']

#------------------- 가정을 반영함 -------------------
for i in range(year - 2023):

    if rate_pol >= 0:
        area_pol = area_pol + (area_pol * (rate_pol / 100))
    else :
        area_pol = area_pol - (abs(area_pol) * (rate_pol / 100))

predict_df['인구 (명)'] = area_pol

#------------------- X데이터와 함께 scaling하고 그 결과를 저장, scaling으로 인해 변한 자료구조를 다시 dataframe으로 변환함 -------------------
copy_X = df_merged.drop(['단속건수', '단속동'], axis=1)
minmax.fit(copy_X)

predict_df = predict_df.drop(['단속건수', '단속동'], axis=1)

predict_df = minmax.transform(predict_df)
predict_df = pd.DataFrame(predict_df)

#------------------- 사용자의 가정 정보를 토대로 불법주정차 수를 예측 -> Ridge Regression -------------------
print("\nResult of Regression -> Ridge Regression(user input)")
y_pred = ridgeModel.predict(predict_df.values.reshape(1, -1))
print(str(year) + "년 " + str(area) + " 지역의 예상 불법주정차 수는 " + str(y_pred) + " 대 입니다.")


#------------------- 이 부분은 test셋으로 split하였던 부분에 대한 예측값 -------------------
print("\nResult of Regression -> Ridge Regression(test set)")
y_pred = ridgeModel.predict(X_valid)

print(y_pred)

print("\nErrors of Ridge Regression")
mse = mean_squared_error(y_valid, y_pred)
print(f"Mean Squared Error: {mse}")

# Evaluate the model using mean absolute error (MAE)
mae = mean_absolute_error(y_valid, y_pred)
print(f"Mean Absolute Error: {mae}")

# Evaluate the model using mean absolute percentage error (MAPE)
mape = mean_absolute_percentage_error(y_valid, y_pred)
print(f"Mean Absolute Percentage Error: {mape}")

print("\nResult of Regression -> Linear Regression(test set)")
y_pred = linearModel.predict(X_valid)
print(y_pred)

print("\nErrors of Linear Regression")
mse = mean_squared_error(y_valid, y_pred)
print(f"Mean Squared Error: {mse}")

# Evaluate the model using mean absolute error (MAE)
mae = mean_absolute_error(y_valid, y_pred)
print(f"Mean Absolute Error: {mae}")

# Evaluate the model using mean absolute percentage error (MAPE)
mape = mean_absolute_percentage_error(y_valid, y_pred)
print(f"Mean Absolute Percentage Error: {mape}")

print("\nResult of Regression -> Lasso Regression(test set)")
y_pred = lassoModel.predict(X_valid)

print(y_pred)

print("\nErrors of Lasso Regression")
mse = mean_squared_error(y_valid, y_pred)
print(f"Mean Squared Error: {mse}")

# Evaluate the model using mean absolute error (MAE)
mae = mean_absolute_error(y_valid, y_pred)
print(f"Mean Absolute Error: {mae}")

# Evaluate the model using mean absolute percentage error (MAPE)
mape = mean_absolute_percentage_error(y_valid, y_pred)
print(f"Mean Absolute Percentage Error: {mape}")


import seaborn as sns
import matplotlib.pyplot as plt

print(df_merged.columns)

df_plot = df_merged.copy()

# scale
columns_to_scale = ['인구 (명)', '단속동']
df_scaled = minmax.fit_transform(df_plot.drop(columns_to_scale, axis=1))

# insert scaled columns
df_plot[df_plot.columns.drop(columns_to_scale)] = df_scaled

# visualize
sns.pairplot(df_plot, x_vars=['인구 (명)', '상권업종대분류명_관광/여가/오락', '상권업종대분류명_부동산', '상권업종대분류명_생활서비스', '상권업종대분류명_소매'],
             y_vars='단속건수', kind='reg', height=4)
plt.show()

sns.pairplot(df_plot, x_vars=['상권업종대분류명_숙박', '상권업종대분류명_스포츠', '상권업종대분류명_음식', '상권업종대분류명_학문/교육'],
             y_vars='단속건수', kind='reg', height=4)
plt.show()


X_candidates = ['인구 (명)', '상권업종대분류명_관광/여가/오락', '상권업종대분류명_부동산', '상권업종대분류명_생활서비스', '상권업종대분류명_소매', '상권업종대분류명_숙박', '상권업종대분류명_스포츠', '상권업종대분류명_음식', '상권업종대분류명_학문/교육']

for x in X_candidates :
    X_1 = df_merged[x].values.reshape(-1, 1)
    y_1 = df_merged['단속건수']

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

    ridgeModel = Ridge()
    ridgeModel.fit(X_train_1, y_train_1)

    y_pred = ridgeModel.predict(X_test_1)

    mse = mean_squared_error(y_test_1, y_pred)
    mae = mean_absolute_error(y_test_1, y_pred)
    mape = mean_absolute_percentage_error(y_test_1, y_pred)

    print("\n" + x + "에 대한 결과")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}")


# Evaluate the model using mean squared error (MSE)
#mse = mean_squared_error(y_valid, y_pred)
#print(f"Mean Squared Error: {mse}")

# Step 7: Model Evaluation and Optimization


# classification model--------------------------------------------
print("\n--------------------- Classification Analysis -----------------------")
# Find mean of '단속횟수'
count_mean = df_merged['단속건수'].mean()

# category function
def categorize_count(x):
    if x < count_mean:
        return '적음'
    else:
        return '많음'

# add category column
df_merged['단속건수_범주'] = df_merged['단속건수'].apply(categorize_count)

# check category
print(df_merged['단속건수_범주'].value_counts())

print(df_merged)

print("--------------------- Decision Tree -----------------------")

# Decision Tree
# Split data into training and test sets
X = df_merged.drop(['단속건수', '단속동', '단속건수_범주'], axis=1)
y = df_merged['단속건수_범주']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Decision Tree classifier
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)

# Evaluate test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Test set accuracy: {accuracy:.2f}")

evaluate_Kfold(X,y,5, "Decision Tree")
evaluation_classification(y_test,y_pred)
classification_report_result = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report_result)

# visulization
import graphviz
from sklearn.tree import export_graphviz


dot_data = export_graphviz(dt, out_file=None, feature_names=X.columns, class_names=['적음', '많음'], filled=True, rounded=True)

# visualize
graph = graphviz.Source(dot_data)

# save PNG
graph.format = 'png'
graph.render(filename='final_decision_tree', cleanup=True)


print("--------------------- KNN -----------------------")
from sklearn.neighbors import KNeighborsClassifier

# split data into training and test sets
# X = df_merged.drop(['단속횟수', '소속구역명', '단속횟수_범주', '시장규모', '관공서 수', '교육시설 수', '초중고교 수'], axis=1)
X = df_merged.drop(['단속건수', '단속동', '단속건수_범주'], axis=1)
y = df_merged['단속건수_범주']

# scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # K값은 변경 가능
knn.fit(X_train, y_train)

# predict the test set
y_pred = knn.predict(X_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy}")
evaluation_classification(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report_result)

# Define the number of folds
k = 5

# Perform K-fold cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)
accuracy_scores = []

for train_index, val_index in kf.split(X_scaled):
    X_train_k, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train_k, y_val = y[train_index], y[val_index]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_k, y_train_k)
    y_pred = knn.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)

average_accuracy = sum(accuracy_scores) / k

print(f"KNN Average accuracy: {average_accuracy:.2f}")
print("--------------------- Random Forest -----------------------")

# Random Forest
# Split data into training and test sets
X = df_merged.drop(['단속건수', '단속동', '단속건수_범주'], axis=1)
y = df_merged['단속건수_범주']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict test set labels
y_pred = rf.predict(X_test)

# Evaluate test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Test set accuracy: {accuracy:.2f}")

evaluate_Kfold(X, y, 5, "Random Forest")
evaluation_classification(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report_result)


print("--------------------- XGBoost -----------------------")

# XGBoost
# Split data into training and test sets
X = df_merged.drop(['단속건수', '단속동', '단속건수_범주'], axis=1)
y = df_merged['단속건수_범주'].map({'많음': 1, '적음': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the XGBoost classifier
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)

# Predict test set labels
y_pred = xgb.predict(X_test)

# Evaluate test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Test set accuracy: {accuracy:.2f}")

evaluate_Kfold(X, y, 5, "XGBoost")
evaluation_classification(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report_result)