import pandas as pd # for data analysis
from matplotlib import font_manager, rc # for plotting Korean font

font_path = '/Library/Fonts/Arial Unicode.ttf'

# 경기도 성남시_주정차 위반 단속 위치 현황
# https://www.data.go.kr/data/15037104/fileData.do
#
# 서울특별시 강동구_동별 불법주정타 단속현황
# https://www.data.go.kr/data/15081659/fileData.do
#
# 서울특별시 강서구_불법주정차 단속 현황
# https://www.data.go.kr/data/15083768/fileData.do
#
# 서울특별시 구로구_주정차단속현황
# https://www.data.go.kr/data/15034492/fileData.do
#
# 서울특별시 서초구_주정차 단속 현황
# https://www.data.go.kr/data/15087185/fileData.do
#
# 서울특별시 성북구_불법주정차 동별 데이터
# https://www.data.go.kr/data/15113658/fileData.do
#
# 서울특별시 송파구_주정차단속건수정보
# https://www.data.go.kr/data/15048835/fileData.do
#
# 서울특별시 영등포구_주정차단속현황
# https://www.data.go.kr/data/15034483/fileData.do
#
# 서울특별시 용산구_불법주정차단속현황
# https://www.data.go.kr/data/15084175/fileData.do
#
# 서울특별시 종로구_불법주정차 통계
# https://www.data.go.kr/data/15100293/fileData.do
#
# 서울특별시_강남구_불법주정차단속현황
# https://www.data.go.kr/data/15048827/fileData.do
#
# 서울특별시_서대문구_주정차 단속 현황
# https://www.data.go.kr/data/15034465/fileData.do
#
# 소상공인시장진흥공단_상가(상권)정보
# https://www.data.go.kr/data/15083033/fileData.do
#
# 경기도 성남시_인구및세대_현황
# https://www.data.go.kr/data/15007386/fileData.do
#
# 서울 인구수 데이터
# https://www.data.go.kr/data/15046938/fileData.do





# font setting
font_name = font_manager.FontProperties(fname=font_path).get_name()

rc('font', family=font_name)



# ---------------------강동구 불러오기---------------------
print("---------------------강동구 불러오기---------------------")
address = "추가데이터/서울특별시 강동구_동별 불법주정차 단속현황_20201231.csv"
df1 = pd.read_csv(address, encoding = 'cp949')
print(df1.describe())

# remain = ['단속동', '단속건수']
df1 = df1.drop(['기준일자'], axis=1)
print(df1)

# ---------------------서초구 불러오기---------------------
print("---------------------서초구 불러오기---------------------")
address = "추가데이터/서울특별시 서초구_주정차 단속 현황_20210831.csv"
df2 = pd.read_csv(address, encoding = 'cp949')
print(df2.describe())

# '날짜' column convert to datetime
df2['단속일시'] = pd.to_datetime(df2['단속일시'])

# add '연도' column
df2['연도'] = df2['단속일시'].dt.year

# preprocessing for 단속동 column ex 서초1동 -> 서초동
df2['단속동'] = df2['단속동'].str.extract(r'(\w+[동])')


# calculate mean of 단속건수 by 연도, 단속동
df_grouped = df2.groupby(['연도', '단속동']).size().reset_index(name='단속건수')

df2 = df_grouped.groupby('단속동')['단속건수'].mean().reset_index(name='단속건수')
print(df2)


# ---------------------송파구 불러오기---------------------
print("---------------------송파구 불러오기---------------------")
address = "추가데이터/서울특별시 송파구_주정차단속건수정보_20200313..csv"
df3 = pd.read_csv(address, encoding = 'cp949')
print(df3.describe())

# preprocessing for 단속동 column
df3['단속동'] = df3['단속동'].str.extract(r'(\w+[동])')

# calculate mean of 단속건수 by 단속동
df3 = df3.groupby('단속동')['단속건수'].mean().reset_index(name='단속건수')

print(df3)

# ---------------------강남구 불러오기---------------------
print("---------------------강남구 불러오기---------------------")
address = "추가데이터/서울특별시_강남구_불법주정차단속현황_20220207.csv"
df4 = pd.read_csv(address, encoding = 'cp949')
print(df4.describe())

# trim space in 동명 column
df4['동명'] = df4['동명'].str.replace(' ', '')

# calculate mean of 단속건수 by 동명
df4 = df4.groupby('동명')['부과건수'].mean().reset_index(name='단속건수')

# rename column name
df4.rename(columns={'동명': '단속동'}, inplace=True)
print(df4)

# ---------------------강서구 불러오기---------------------
print("---------------------강서구 불러오기---------------------")
address = "추가데이터/서울특별시 강서구_불법주정차 단속 현황_20201231.csv"
df5 = pd.read_csv(address, encoding = 'cp949')
print(df5.describe())

# calculate mean of 단속건수 by 행정동명
df5 = df5.groupby('행정동명')[' 단속건수 '].mean().reset_index(name=' 단속건수 ')

# rename column name
df5.rename(columns={'행정동명': '단속동'}, inplace=True)
df5.rename(columns={' 단속건수 ': '단속건수'}, inplace=True)

print(df5)

# ---------------------구로구 불러오기---------------------
print("---------------------구로구 불러오기---------------------")
address = "추가데이터/서울특별시 구로구_주정차단속현황_20230127.csv"
df6 = pd.read_csv(address, encoding = 'cp949')
print(df6.describe())

# calculate mean of 단속건수 by 단속동
df6 = df6.groupby('단속동')['단속건수'].mean().reset_index(name='단속건수')

print(df6)

# ---------------------영등포구 불러오기---------------------
print("---------------------영등포구 불러오기---------------------")
address = "추가데이터/서울특별시 영등포구_주정차단속현황_20230504.csv"
df7 = pd.read_csv(address, encoding = 'cp949')
print(df7.describe())

# preprocessing for 단속동 column
df7['단속동'] = df7['단속동'].str.extract(r'(\w+[동])')

# preprocessing for 단속년도 column 2022년 -> 2022
df7['단속년도'] = df7['단속년도'].str.extract(r'(\d{4})')

# calculate mean of 단속건수 by 단속동
df7 = df7.groupby('단속동')['단속건수'].mean().reset_index(name='단속건수')

print(df7)

# ---------------------용산구 불러오기---------------------
print("---------------------용산구 불러오기---------------------")
address = "추가데이터/서울특별시 용산구_불법주정차단속현황_10_25_2021.csv"
df8 = pd.read_csv(address, encoding = 'cp949')
print(df8.describe())

# convert 단속일시 column to datetime type
df8['단속일시'] = pd.to_datetime(df8['단속일시'])

# add 연도 column
df8['연도'] = df8['단속일시'].dt.year

# preprocessing for 단속동 column
df8['단속동'] = df8['단속동'].str.extract(r'(\w+[동])')

# calculate mean of 단속건수 by 단속동
df_grouped = df8.groupby(['연도', '단속동']).size().reset_index(name='단속건수')
df8 = df_grouped.groupby('단속동')['단속건수'].mean().reset_index(name='단속건수')

print(df8)

# ---------------------서대문구 불러오기---------------------
print("---------------------서대문구 불러오기---------------------")
address = "추가데이터/서울특별시_서대문구_주정차 단속 현황_20220809.csv"
df9 = pd.read_csv(address, encoding = 'cp949')
print(df9.describe())

# preprocessing for 단속동 column
df9['단속동'] = df9['단속동'].str.extract(r'(\w+[동])')

# calculate mean of 단속건수 by 단속동
df9 = df9.groupby(['단속동']).size().reset_index(name='단속건수')

print(df9)

# ---------------------종로구 불러오기---------------------
print("---------------------종로구 불러오기---------------------")
address = "추가데이터/서울특별시 종로구_불법주정차 통계_20211231.csv"
df10 = pd.read_csv(address, encoding = 'cp949')
print(df10.describe())
print(df10.info)

# drop 기간 column
df10 = df10.drop(['기간'], axis=1)

# rename column name
df10.rename(columns={'행정동': '단속동'}, inplace=True)
df10.rename(columns={'단속건': '단속건수'}, inplace=True)

print(df10)

# ---------------------성북구 불러오기---------------------
print("---------------------성북구 불러오기---------------------")
address = "추가데이터/서울특별시 성북구_불법주정차 동별 데이터_20230504.csv"
df11 = pd.read_csv(address, encoding = 'cp949')
print(df11.describe())

# preprocessing for 단속동 column
df11['단속동'] = df11['단속동'].str.extract(r'(\w+[동])')

# calculate sum of 단속건수 by 단속동
df11 = df11.groupby('단속동')['단속건수'].sum().reset_index(name='단속건수')

print(df11)

# merge all dataframes
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11], axis=0, ignore_index=True)
df = df.groupby('단속동')['단속건수'].sum().reset_index(name='단속건수')


print("---------------------결과 불러오기---------------------")
print(df)
print(df.describe())

# ---------------------서울 인구수 데이터 불러오기---------------------
print("---------------------서울 인구수 데이터 불러오기---------------------")
address = "추가데이터/인구밀도_20230603023951.csv"
df_sum = pd.read_csv(address, encoding = 'utf-8')
print(df_sum)

# create new dataframe with only '동별(3)' and '인구 (명)' columns
df_sum = df_sum[['동별(3)', '인구 (명)']]

# rename column name
df_sum.rename(columns={'동별(3)': '단속동'}, inplace=True)

# calculate sum of 인구 (명) by 단속동
df_sum = df_sum.groupby('단속동')['인구 (명)'].sum().reset_index(name='인구 (명)')

print(df_sum)


# ---------------------서울시_동별_단속현황.xlsx 파일로 저장---------------------
print("---------------------서울시_동별_단속현황.xlsx 파일로 저장---------------------")
# df와 df_sum를 하나의 데이터 프레임으로
df_preprocessing = pd.merge(df, df_sum, on='단속동', how='inner')
print(df_preprocessing)

df_preprocessing.to_excel('서울시_동별_단속현황.xlsx', index = False)

# # '인구' column명 변경
# df_preprocessing.rename(columns={'인구 (명)': '인구'}, inplace=True)
#
# # 단속동이 청담동, 삼성동, 신사동 row는 삭제
# df_preprocessing = df_preprocessing.drop(df_preprocessing[df_preprocessing['단속동'] == '청담동'].index)
# df_preprocessing = df_preprocessing.drop(df_preprocessing[df_preprocessing['단속동'] == '삼성동'].index)
# df_preprocessing = df_preprocessing.drop(df_preprocessing[df_preprocessing['단속동'] == '신사동'].index)
