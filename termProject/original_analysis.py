import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from matplotlib import font_manager, rc # 폰트 세팅을 위한 모듈 추가

font_path = '/Library/Fonts/Arial Unicode.ttf'  # 사용하고자 하는 한글 폰트 파일 경로로 변경해주세요.
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


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

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        accuracy_scores.append(accuracy)

    average_accuracy = sum(accuracy_scores) / k  # 평균 정확도 계산

    print(f"{mode} Average accuracy: {average_accuracy:.2f}")


def evaluation_classification(y_true, y_pred):
    # 정확도 계산
    accuracy = accuracy_score(y_true, y_pred)

    # 혼동 행렬 계산
    confusion_mtx = confusion_matrix(y_true, y_pred)

    # 정밀도 계산
    precision = precision_score(y_true, y_pred, average='macro')  # 다중 클래스인 경우 average='macro' 사용

    # 재현율 계산
    recall = recall_score(y_true, y_pred, average='macro')  # 다중 클래스인 경우 average='macro' 사용

    # F1 점수 계산
    f1 = f1_score(y_true, y_pred, average='macro')  # 다중 클래스인 경우 average='macro' 사용


    # 출력
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_mtx)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


address = "경기도 성남시_주정차 위반 단속 위치 현황_20220927.csv"
df1 = pd.read_csv(address, encoding = 'cp949')


words_to_check = ["창곡", "양지", "은행", "산성", "단대", "금광", "상대원", "갈현", "도촌", "여수", "하대원", "성남", "수진", "신흥", "복정", "태평", "신촌", "오야", "심곡", "둔전", "고등", "시흥", "사송", "상적", "금토", "야탑", "이매", "율", "서현", "분당", "수내", "정자", "구미", "금곡", "동원", "궁내", "백현", "판교", "삼평", "하산운", "대장", "운중", "석운"]

df1['단속장소'] = df1['단속장소'].str.extract(r'(\w+[동로])')


# 시군명, 관리기관명, 단속일시정보, 단속방법 column을 제거
df1 = df1.drop(['시군명', '관리기관명', '단속일시정보', '단속방법'], axis = 1).reset_index()
df1 = df1.drop(['index'], axis = 1)


#------------------- words_to_check 리스트 안에 있는 단어들이 있는 row들을 통일시킴 -------------------
mask = df1['단속장소'].str.contains('|'.join(words_to_check), case=False, na=False)

df1.loc[mask, '단속장소'] = df1.loc[mask, '단속장소'].apply(lambda x: next((word for word in words_to_check if word.lower() in x.lower()), x))

df1 = df1.dropna(how='any').reset_index()
df1 = df1.drop(['index'], axis = 1)

# Update the 'place' column where the condition is met
# 단속장소 별 단속횟수 계산
# 연도별로 단속동에 대한 단속건수 계산
df_grouped = df1.groupby(['집계년도', '단속장소']).size().reset_index(name='단속횟수')


df_counts = df_grouped.groupby('단속장소')['단속횟수'].mean().reset_index(name='단속횟수')

# 단속횟수가 10 이상인 데이터만 필터링
df_counts = df_counts[df_counts['단속횟수'] > 10]

# 단속횟수 내림차순으로 정렬
df_counts = df_counts.sort_values(by=['단속횟수'], axis=0, ascending=False).reset_index()
df_counts = df_counts.drop(['index'], axis = 1)


address = "경기도 성남시_성남시 전통시장_발달상권_골목상권 업종별 시장규모 현황_20211201.csv"
df2 = pd.read_csv(address, encoding = 'cp949')

df2['소속구역명'] = df2['소속구역명'].str.extract(r'(\w+[동로])')


#------------------- words_to_check 리스트 안에 있는 단어들이 있는 row들을 통일시킴 -------------------
mask = df2['소속구역명'].str.contains('|'.join(words_to_check), case=False, na=False)

df2.loc[mask, '소속구역명'] = df2.loc[mask, '소속구역명'].apply(lambda x: next((word for word in words_to_check if word.lower() in x.lower()), x))

df2 = df2.dropna(how='any').reset_index()
df2 = df2.drop(['index'], axis = 1)

df_sum = df2.groupby('소속구역명')['시장규모'].sum().reset_index()
df_sum.columns = ['소속구역명', '시장규모']

df_sum.dropna(inplace=True)


#-------------------Data merging-------------------
df_merged = pd.merge(df_counts, df_sum, left_on='단속장소', right_on='소속구역명', how='inner')
df_merged = df_merged.drop('단속장소', axis = 1)




address = "경기도 성남시_성남시 전통시장_발달상권_골목상권 기본상권정보 현황_20211201.csv"
df3 = pd.read_csv(address, encoding = 'cp949')

#------------------- words_to_check 리스트 안에 있는 단어들이 있는 row들을 통일시킴 -------------------

mask = df3['소속구역명'].str.contains('|'.join(words_to_check), case=False, na=False)

df3.loc[mask, '소속구역명'] = df3.loc[mask, '소속구역명'].apply(lambda x: next((word for word in words_to_check if word.lower() in x.lower()), x))
df3 = df3[df3['소속구역 ( 소속구역 한정 1 소속구역 반경 500m 2 소속구역 반경 1000m 3)'] == 3].reset_index()
print(df3)

#------------------- Data merging -------------------

df_pol = df3.groupby('소속구역명')['주거인구 수'].sum().reset_index()
df_merged = pd.merge(df_merged, df_pol, how='inner')

df_pub = df3.groupby('소속구역명')['관공서 수'].sum().reset_index()
df_merged = pd.merge(df_merged, df_pub, how='inner')

df_bank = df3.groupby('소속구역명')['금융기관 수'].sum().reset_index()
df_merged = pd.merge(df_merged, df_bank, how='inner')

df_edu = df3.groupby('소속구역명')['교육시설 수'].sum().reset_index()
df_merged = pd.merge(df_merged, df_edu, how='inner')

df_dis = df3.groupby('소속구역명')['유통점 수 '].sum().reset_index()
df_merged = pd.merge(df_merged, df_dis, how='inner')

df_sch = df3.groupby('소속구역명')['초중고교 수'].sum().reset_index()
df_merged = pd.merge(df_merged, df_sch, how='inner')
print(df_merged)

# 단속횟수 평균
print(df_merged['단속횟수'].mean())

#------------------- 학습시킬 데이터를 제작 및 0과 1 사이로 정규화 -------------------
minmax = MinMaxScaler()
X = df_merged.drop(['단속횟수', '소속구역명'], axis=1)
X = minmax.fit_transform(X)
y = df_merged['단속횟수']

#------------------- Split data into training and validation sets-------------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

linearModel = LinearRegression()
linearModel.fit(X_train, y_train)


#------------------- 사용자로부터 미래에 대한 가정 정보를 받아옴 -------------------
area = input("예상하고자 하는 지역을 적어주세요 : ")
rate_mark, rate_pol, year = input("연간 시장규모 변화율, 인구규모 변화율, 예상할 연도를 적어주세요 : ").split()
rate_mark = int(rate_mark)
rate_pol = int(rate_pol)
year = int(year)


#------------------- 가정을 반영하기 위해 예상하고자 하는 지역의 row를 따로 추출함 -------------------
area_row = df_merged[df_merged['소속구역명'] == area]
area_mark = area_row['시장규모']
area_pol = area_row['주거인구 수']


#------------------- 가정을 반영함 -------------------

for i in range(year - 2023):
    if rate_mark >= 0:
        area_mark = area_mark + (area_mark * (rate_mark / 100))
    else :
        area_mark = area_mark - (abs(area_mark) * (rate_mark / 100))

    if rate_pol >= 0:
        area_pol = area_pol + (area_pol * (rate_pol / 100))
    else :
        area_pol = area_pol - (abs(area_pol) * (rate_pol / 100))

area_row['시장규모'] = area_mark
area_row['주거인구 수'] = area_pol


#------------------- 가정을 반영한 새로운 row array를 만듦 -------------------
predict_array = {'시장규모' : [area_row['시장규모']],
                 '주거인구 수' : [area_row['주거인구 수']],
                 '관공서 수' : [area_row['관공서 수']],
                 '금융기관 수' : [area_row['금융기관 수']],
                 '교육시설 수' : [area_row['교육시설 수']],
                 '유통점 수' : [area_row['유통점 수 ']],
                 '초중고교 수' : [area_row['초중고교 수']]}

predict_df = pd.DataFrame(predict_array)


#------------------- X데이터와 함께 scaling하고 그 결과를 저장, scaling으로 인해 변한 자료구조를 다시 dataframe으로 변환함 -------------------
copy_X = df_merged.drop(['단속횟수', '소속구역명'], axis=1)
minmax.fit(copy_X)

predict_df = minmax.transform(predict_df)
predict_df = pd.DataFrame(predict_df)


#------------------- 사용자의 가정 정보를 토대로 불법주정차 수를 예측 -------------------
print("\nResult of Regression(user input)")
y_pred = linearModel.predict(predict_df.values.reshape(1, -1))
print(str(year) + "년 " + str(area) + " 지역의 예상 불법주정차 수는 " + str(y_pred) + " 대 입니다.")


#------------------- 이 부분은 test셋으로 split하였던 부분에 대한 예측값 -------------------
print("\nResult of Regression(test set)")
y_pred = linearModel.predict(X_valid)

print(y_pred)



# Step 7: Model Evaluation and Optimization

# Evaluate the model using mean squared error (MSE)
mse = mean_squared_error(y_valid, y_pred)
print(f"Mean Squared Error: {mse}")

# Evaluate the model using mean absolute error (MAE)
mae = mean_absolute_error(y_valid, y_pred)
print(f"Mean Absolute Error: {mae}")

# Evaluate the model using mean absolute percentage error (MAPE)
mape = mean_absolute_percentage_error(y_valid, y_pred)
print(f"Mean Absolute Percentage Error: {mape}")

# -> Result is not good enough. We need to optimize the model.

# 다중 선형 회귀 결과 시각화
sns.pairplot(df_merged, x_vars=['시장규모', '주거인구 수', '관공서 수', '금융기관 수', '교육시설 수', '유통점 수 ', '초중고교 수'],
             y_vars='단속횟수', kind='reg', height=4)
plt.show()

X_candidates = ['시장규모', '주거인구 수', '관공서 수', '금융기관 수', '교육시설 수', '유통점 수 ', '초중고교 수']

for x in X_candidates :
    X_1 = df_merged[x].values.reshape(-1, 1)
    y_1 = df_merged['단속횟수']

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

    linearModel = LinearRegression()
    linearModel.fit(X_train_1, y_train_1)

    y_pred = linearModel.predict(X_test_1)

    mse = mean_squared_error(y_test_1, y_pred)
    mae = mean_absolute_error(y_test_1, y_pred)
    mape = mean_absolute_percentage_error(y_test_1, y_pred)

    print("\n" + x + "에 대한 결과")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}")


# classification model--------------------------------------------
print("\n--------------------- Classification Analysis -----------------------")

# Find mean of '단속횟수'
count_mean = df_merged['단속횟수'].mean()

# # 범주화 함수 정의
# def categorize_count(x):
#     if x < count_mean / 5:
#         return '매우 적음'
#     elif x < count_mean * 2 / 5:
#         return '적음'
#     elif x < count_mean * 3 / 5:
#         return '중간'
#     elif x < count_mean * 4 / 5:
#         return '많음'
#     else:
#         return '매우 많음'

# 범주화 함수 정의
def categorize_count(x):
    if x < 5000:
        return '적음'
    else:
        return '많음'

# '단속횟수'를 범주화하여 '단속횟수_범주' 컬럼에 추가
df_merged['단속횟수_범주'] = df_merged['단속횟수'].apply(categorize_count)

# 모든 row 보이게
pd.set_option('display.max_rows', None)

# 범주화 결과 확인
print(df_merged['단속횟수_범주'].value_counts())

print(df_merged)

print("--------------------- Decision Tree -----------------------")


# Decision Tree
# Split data into training and test sets
# X = df_merged.drop(['단속횟수', '소속구역명', '단속횟수_범주', '시장규모', '관공서 수', '교육시설 수', '초중고교 수'], axis=1)
X = df_merged.drop(['단속횟수', '소속구역명', '단속횟수_범주'], axis=1)
y = df_merged['단속횟수_범주']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Decision Tree classifier
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)

# 위의 의사결정트리를 시각화
# 시각화를 위한 dot 데이터 생성
import graphviz
from sklearn.tree import export_graphviz

# 시각화를 위한 dot 데이터 생성
dot_data = export_graphviz(dt, out_file=None, feature_names=X.columns, class_names=['적음', '많음'], filled=True, rounded=True)

# graphviz로 의사결정 트리 시각화
graph = graphviz.Source(dot_data)

# PNG 파일로 저장
graph.format = 'png'
graph.render(filename='decision_tree', cleanup=True)

# Evaluate test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Test set accuracy: {accuracy:.2f}")



evaluate_Kfold(X,y,5, "Decision Tree")
evaluation_classification(y_test,y_pred)
classification_report_result = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report_result)



print("--------------------- KNN -----------------------")
from sklearn.neighbors import KNeighborsClassifier

# 학습 데이터와 타겟 변수 분리
# X = df_merged.drop(['단속횟수', '소속구역명', '단속횟수_범주', '시장규모', '관공서 수', '교육시설 수', '초중고교 수'], axis=1)
X = df_merged.drop(['단속횟수', '소속구역명', '단속횟수_범주'], axis=1)
y = df_merged['단속횟수_범주']

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN 모델 학습
knn = KNeighborsClassifier(n_neighbors=5)  # K값은 변경 가능
knn.fit(X_train, y_train)

# 예측
y_pred = knn.predict(X_test)

# 정확도 평가
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

    knn = KNeighborsClassifier(n_neighbors=3)  # K값은 변경 가능
    knn.fit(X_train_k, y_train_k)
    y_pred = knn.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)

average_accuracy = sum(accuracy_scores) / k  # 평균 정확도 계산

print(f"KNN Average accuracy: {average_accuracy:.2f}")








