import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')  # 윈도우
# plt.rc('font', family='AppleGothic')  # 맥
# plt.rc('font', family='NanumGothic')  # 리눅스/코랩
plt.rcParams['axes.unicode_minus'] = False


df2016 = pd.read_csv("PDQCSV_26_20250520193620928.csv",encoding="CP949") 
df2017 = pd.read_csv("PDQCSV_26_20250520193704993.csv",encoding="CP949") 
df2018 = pd.read_csv("PDQCSV_26_20250520193745778.csv",encoding="CP949") 
df2019 = pd.read_csv("PDQCSV_26_20250520193817155.csv",encoding="CP949") 
df2020 = pd.read_csv("PDQCSV_26_20250520193848790.csv",encoding="CP949") 
df2021 = pd.read_csv("PDQCSV_26_20250520193920240.csv",encoding="CP949")
df2022 = pd.read_csv("PDQCSV_26_20250520193951631.csv",encoding="CP949")
df2023 = pd.read_csv("PDQCSV_26_20250520194024678.csv",encoding="CP949") 
df2024 = pd.read_csv("PDQCSV_26_20250520194108628.csv",encoding="CP949") 

df2016 = df2016[df2016['통계시도명'] == '서울특별시']
df2016 = df2016.query('1 <= 연령 <= 7')
sum2016 = df2016['등록장애인수'].sum()
sum2016

df2017 = df2017[df2017['통계시도명'] == '서울특별시']
df2017 = df2017.query('1 <= 연령 <= 7')
sum2017 = df2017['등록장애인수'].sum()
sum2017

df2018 = df2018[df2018['통계시도명'] == '서울특별시']
df2018 = df2018.query('1 <= 연령 <= 7')
sum2018 = df2018['등록장애인수'].sum()
sum2018

df2019 = df2019[df2019['통계시도명'] == '서울특별시']
df2019 = df2019.query('1 <= 연령 <= 7')
sum2019 = df2019['등록장애인수'].sum()
sum2019

df2020 = df2020[df2020['통계시도명'] == '서울특별시']
df2020 = df2020.query('1 <= 연령 <= 7')
sum2020 = df2020['등록장애인수'].sum()
sum2020

df2021 = df2021[df2021['통계시도명'] == '서울특별시']
df2021 = df2021.query('1 <= 연령 <= 7')
sum2021 = df2021['등록장애인수'].sum()
sum2021

df2022 = df2022[df2022['통계시도명'] == '서울특별시']
df2022 = df2022.query('1 <= 연령 <= 7')
sum2022 = df2022['등록장애인수'].sum()
sum2022

df2023 = df2023[df2023['통계시도명'] == '서울특별시']
df2023 = df2023.query('1 <= 연령 <= 7')
sum2023 = df2023['등록장애인수'].sum()
sum2023

df2023 = df2023[df2023['통계시도명'] == '서울특별시']
df2023 = df2023.query('1 <= 연령 <= 7')
sum2023 = df2023['등록장애인수'].sum()
sum2023

df2024 = df2024[df2024['통계시도명'] == '서울특별시']
df2024 = df2024.query('1 <= 연령 <= 7')
sum2024 = df2024['등록장애인수'].sum()
sum2024


temp =[2675,2920,3011,3081,3046,2990,2993,3089,3109]

years = list(range(2016, 2025))  # 2016 ~ 2025
years
# 데이터프레임 생성
df = pd.DataFrame({
    '연도': years,
    '장애인수': temp
})

# 시각화
(
    so.Plot(df, x='연도', y='장애인수')
    .add(so.Line())
    .add(so.Dots())  # ← 여기서 Dots 사용
    .label(title='연도별 장애인 수 변화', x='연도', y='장애인 수')
    .scale(x=so.Continuous(), y=so.Continuous())
    .show()
)
