import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csvArray = ['경기도 성남시_주정차 위반 단속 위치 현황_20220927.csv',
            '교통단속카메라.csv',
            '경기도 성남시_공영주차장 월별 입출차 현황_20221122.csv',
            '경기도 성남시_성남시 전통시장_발달상권_골목상권 기본상권정보 현황_20211201.csv',
            '경기도 성남시_성남시 전통시장_발달상권_골목상권 업종별 시장규모 현황_20211201.csv',
            '경기도 성남시_성남시 전통시장_발달상권_골목상권별 이용고객 소비비중 현황_20211201.csv',
            '경기도 성남시_대규모 점포시장현황_20220531.csv',
            ]
pd.set_option('display.max_columns', None)

def statistical_description(df):
    print('Features types : \n', df.dtypes)
    print('Dataset describe : \n', df.describe(include='all'))
    print('Dataset head : \n', df.head())
    for col in df.columns:
        if df[col].dtype == 'object':  # 문자열 데이터에 대해서만 처리
            unique_values = df[col].unique()
            value_counts = df[col].value_counts()
     #       print(f"[{col}] - Unique values: \n {unique_values}")
            print(f"[{col}] - Value counts: \n {value_counts}\n")


for csv in csvArray:
    df = pd.read_csv(csv, encoding='cp949')
    print('Dataset Title : ', csv)
    statistical_description(df)



