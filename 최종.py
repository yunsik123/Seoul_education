import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import openpyxl
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS
from tqdm import tqdm
import xml.etree.ElementTree as ET
import requests
import time
from urllib.parse import quote
import scipy
import rasterio
import rasterstats
from scipy.interpolate import griddata
from scipy import ndimage
from pyproj import Transformer
import folium
from folium.plugins import HeatMap


#1. 서울시 학교별 특수학급수
df1 = pd.read_csv("data/서울시_학교별_학급별_학생수_현황.csv",encoding="CP949")
#2. 전국초중등위치데이터
df2 = pd.read_csv("data/전국초중등학교위치표준데이터.csv",encoding="CP949")
#3. 행정구역별특수학급및학생수
df3 = pd.read_excel("data/2024_행정구역별_학년별_학급수_학생수.xlsx")
#4. 행정구역별교원수
df4 = pd.read_excel("data/2024_행정구역별_직위별_교원수.xlsx")
#5. 구별 장애유형관련 인원수(수요예측용용)
df5 = pd.read_csv("data/구별장애유형별인구수.csv",encoding="UTF-8")
#6. 학교별 장애인 편의시설 현황
df6 = pd.read_csv("data/서울시_학교별_장애인_편의시설_현황.csv",encoding="CP949")
#7. 서울시 전철역사위치정보(교통1)
df7 = pd.read_csv("data/서울교통공사_1_8호선_역사_좌표(위경도)_정보_20241031.csv",encoding="CP949")
#8. 서울시 버스정류장위치정보보(교통2)
df8 = pd.read_csv("data/국토교통부_전국 버스정류장 위치정보_20241028.csv",encoding="CP949")
#9. 경사도정보
#10. 주변 좋은시설
#공공도서관
#주요공원
#사회복지시설
#경찰서
df10 = pd.read_csv("data/서울시 공공도서관 현황정보.csv",encoding="CP949")
df11 = pd.read_csv("data/서울시 주요 공원현황.csv",encoding="CP949")
df12 = pd.read_csv("data/보건복지부_장애인복지관 현황_20240425.csv",encoding="CP949")
df13 = pd.read_csv("data/seoul_police_stations.csv",encoding="CP949")
df14 = pd.read_csv("data/서울시 병의원 위치 정보.csv",encoding="CP949")
#11. 주변 나쁜시설(어떤것들을 중요하게 생각할지)
df15 = pd.read_csv("data/서울시 유흥주점영업 인허가 정보.csv",encoding="CP949")




df1.info()# 5421개
df1.describe()

df1 = df1[df1['공시연도'] ==2024]
df1['교육지원청'].value_counts()
df1 = df1[['학교명','특수학급 학급수','특수학급 학급당 학생수','전공과 학급수','정보공시 학교코드']]
#https://www.nise.go.kr/onmam/front/M0000088/agency/list.do
df1['전공과 학급수'] = df1['전공과 학급수'].fillna(0)#전공과 학급수 NULL로 표시되니 처리해줄것
df1.info()#1345
df1.loc[df1['학교명'] == '이화여자대학교사범대학부속이화?금란중학교', '학교명'] = '이화여자대학교사범대학부속이화·금란중학교'
df1.loc[df1['학교명'] == '동국대학교사범대학부속여자중학교', '학교명'] = '동국대학교사범대학부속가람중학교'
df1.info()#1345



df2.info()#11985
df2.describe()
df2 = df2[df2['시도교육청명']=='서울특별시교육청']
df2 = df2[['학교ID','학교명','학교급구분','설립형태','운영상태','소재지지번주소','시도교육청명','위도','경도']]
df2 = df2.query("(학교급구분 == '초등학교') or (학교급구분 == '중학교')")
df2['운영상태'].value_counts()#운영밖에없음
df2['시도교육청명'].value_counts()
df2 = df2[df2['시도교육청명']=='서울특별시교육청']
df2['학교급구분'].value_counts()
df2['행정구역'] = df2['소재지지번주소'].str.extract(r'서울특별시\s+(\S+구)')
df2['행정구역'].value_counts()
df2['행정구역'].nunique()
df2 = df2[['행정구역','학교명','학교급구분','설립형태','위도','경도']]
df2 = df2.rename(columns={'학교급구분':'학제'})
df2['설립형태'].value_counts() #공립 사립 국립
df2.info()#992
df2



mdf= pd.merge(df1, df2, on=['학교명'], how='inner')
mdf.info()#988 에서 990

####개수 차이와 관련한 코드 전부 고등학교임
#only_in_df1 = df1[~df1['학교명'].isin(mdf['학교명'])]
#print("df1에만 있는 학교들:")
#print(only_in_df1)

#여기는 4개정도가 나오는데 이름 처리해줘야함
#only_in_df2 = df2[~df2['학교명'].isin(mdf['학교명'])]
#print("df2에만 있는 학교들:")
#print(only_in_df2)
#강동구 서울위례초등학교 #공사중
#강동구 서울둔촌초등학교 #공사중
#서대문구 이화여자대학교사범대학부속이화·금란중학교    ·대신?가쓰였음 고쳐주기
#광진구 동국대학교사범대학부속가람중학교 동국대학교사범대학부속여자중학교 가람으로바뀜
#df1을 바꿔줘야함


df3 = df3[['시도','행정구역','학제','편성 학급수.15','학급구분별 학생수.16']]
df3 = df3[(df3['시도'] == '서울')]
df3 = df3.query("(학제 == '초등학교') or (학제 == '중학교')")
df3 = df3.rename(columns={'편성 학급수.15': '구별및학제별특수학급수','학급구분별 학생수.16':'구별학제별특수학급학생수'})
df3 = df3.drop(columns=['시도'])
df3.info()
df3
dfl = df3.copy()

mdf2= pd.merge(mdf, df3, on=['행정구역', '학제'], how='inner')
mdf2

#테스트
temp = mdf.query("(행정구역 == '서초구') and(학제=='초등학교')")
temp.info() #23 #3개빠짐 위에서 4개관련 채워야할듯


#구별장애인수다보니 10~19세에서 16세미만이 고려되기어려워짐..ㅠㅠㅠ
df5
df5 = df5.drop(columns=['특수학교수','특수학급수','비율'])
df5 = df5.rename(columns={'자치구별': '행정구역'})



mdf3= pd.merge(mdf, df5, on=['행정구역'], how='inner')



df6
df6 = df6[df6['공시연도']==2024]
df6= df6[['정보공시 학교코드','주 출입구 접근로 설치여부','장애인 전용 주차구역 지정여부','주 출입구 높이차이 제거여부','출입구(문) 설치유무','계단/승강기/경사로/휠체어리프트 유무','장애인용 대변기 설치여부','장애인용 소변기 설치여부','점자블록 설치여부','유도 및 안내설비 설치여부','경보 및 피난설비 설치여부']]
mdf4= pd.merge(mdf, df6, on=['정보공시 학교코드'], how='inner')
mdf4=mdf4.dropna()


df7
df8 = df8[df8['도시명'] == "서울특별시"]
bus_coords = df8[['위도', '경도']].values.tolist()


#통학소요
df7.info()
df7 = df7[['위도','경도']];df7['장소'] = '전철'
df8 = df8[['위도','경도']];df8['장소'] = '버스정류장'
df9 = pd.concat([df7,df8],ignore_index=True)
df9




# 정류장(버스 + 전철) 위도·경도 리스트
station_coords = df9[['위도', '경도']].values.tolist()

# 거리 구간별 점수 매핑
distance_scores = [
    (0, 10, 5),
    (10, 50, 4),
    (50, 200, 3),
    (200, 400, 2),
    (800, float('inf'), 1)
]

# 가장 가까운 정류장 거리 기반 점수 계산 함수
def accessibility_score(school_lat, school_lon):
    school_coord = (school_lat, school_lon)
    min_distance = float('inf')

    for station_lat, station_lon in station_coords:
        station_coord = (station_lat, station_lon)
        distance = geodesic(school_coord, station_coord).meters
        if distance < min_distance:
            min_distance = distance

    for r_min, r_max, score in distance_scores:
        if r_min <= min_distance < r_max:
            return score
    return 1  # 예외 방지용

# 접근성 점수 컬럼 생성
mdf4['통학접근성점수'] = mdf4.apply(
    lambda row: accessibility_score(row['위도'], row['경도']),
    axis=1
)


# 1. 표고 데이터 불러오기 (crs 자동 인식)
elevation = gpd.read_file("data/표고_5000/N3P_F002.shp")

# 2. 고도 컬럼 이름 맞추기
elevation = elevation.rename(columns={'HEIGHT': 'elv'})

# 3. md4에서 위도/경도 → Point 생성
mdf4['geometry'] = mdf4.apply(lambda row: Point(row['경도'], row['위도']), axis=1)

# 4. GeoDataFrame으로 만들기 (위도/경도 → EPSG:4326)
md4_gdf = gpd.GeoDataFrame(mdf4, geometry='geometry', crs='EPSG:4326')

# 5. elevation 데이터와 좌표계 맞추기 (EPSG:5179인 경우 대부분)
md4_gdf = md4_gdf.to_crs(elevation.crs)

# 6. 경사도 계산 함수 정의
def calculate_slope(school_point, elevation, buffer_radius=150):
    buffer = school_point.buffer(buffer_radius)
    nearby = elevation[elevation.geometry.within(buffer)]
    
    if nearby.empty:
        return np.nan  # 반경 내 표고점 없음
    
    distances = nearby.geometry.distance(school_point)
    heights = nearby['elv']
    slopes = np.degrees(np.arctan(np.abs(heights - heights.mean()) / distances))
    
    return slopes.mean()

# 7. 모든 학교에 대해 경사도 계산
slopes = []
for geom in tqdm(md4_gdf.geometry):
    slope = calculate_slope(geom, elevation)
    slopes.append(slope)

# 8. 결과 저장
mdf4['경사도'] = slopes
mdf4['경사도'] = mdf4['경사도'].fillna(mdf4['경사도'].mean()) #결측치4개
#####결측치관련논의#######








#공공도서관
#주요공원
#사회복지시설
#경찰서
#유흥주점
#당구장

df10 = df10[['위도','경도']];df10['장소'] = '공공도서관'
df10 = df10.dropna()

df11 = df11[['X좌표(WGS84)','Y좌표(WGS84)']];df11['장소'] = '주요공원'
df11 = df11.rename(columns={'X좌표(WGS84)': '경도','Y좌표(WGS84)':'위도'})
df11 = df11.dropna()

df12 = df12[['엑스(X)좌표','와이(Y)좌표']];df12['장소'] = '장애인인사회복지시설'
df12 = df12.rename(columns={'엑스(X)좌표': '경도','와이(Y)좌표':'위도'})
df12 = df12.dropna()

df13 = df13[['위도','경도']];df13['장소'] = '경찰서'
df13 = df13.dropna()

df14 = df14[['병원위도','병원경도']];df14['장소'] = '병원'
df14 = df14.rename(columns={'병원위도': '위도','병원경도':'경도'})

df15 = df15[['좌표정보(X)','좌표정보(Y)']];df15['장소'] = '유흥주점'
df15 = df15.rename(columns={'좌표정보(X)': '경도','좌표정보(Y)':'위도'})
df15 = df15.dropna()

# EPSG:5174 (중부원점 TM) → EPSG:4326 (WGS84)
transformer = Transformer.from_crs("epsg:5174", "epsg:4326", always_xy=True)
# 변환 함수 정의
def convert_coords(x, y):
    lon, lat = transformer.transform(x, y)
    return lon, lat

# df15에 변환 적용
df15['경도'], df15['위도'] = zip(*df15.apply(lambda row: convert_coords(row['경도'], row['위도']), axis=1))





mdf4.to_csv('temp.csv', index=False, encoding='utf-8-sig')
#############여기까지 일단 돌리기############################
#가져옴
mdf4 = pd.read_csv('temp.csv', encoding='utf-8-sig')


#10개의 학교시설을 1~5점으로 점수 매겨보자~
temp = mdf4[['주 출입구 접근로 설치여부','장애인 전용 주차구역 지정여부','주 출입구 높이차이 제거여부','출입구(문) 설치유무','계단/승강기/경사로/휠체어리프트 유무','장애인용 대변기 설치여부','장애인용 소변기 설치여부','점자블록 설치여부','유도 및 안내설비 설치여부','경보 및 피난설비 설치여부']]
unique_values = pd.unique(temp.values.ravel())
print(unique_values)#3종류


def calc_score(row):
    base_score = 5
    단순설치_count = (row == '단순설치').sum()
    미설치_count = (row == '미설치').sum()

    score = base_score - (단순설치_count * 0.5) - (미설치_count * 1)
    if score < 1:
        score = 1  # 최소 점수 1점으로 제한
    return score

# 점수 계산해서 새로운 열 만들기
mdf4['설치점수'] = mdf4[['주 출입구 접근로 설치여부','장애인 전용 주차구역 지정여부','주 출입구 높이차이 제거여부','출입구(문) 설치유무','계단/승강기/경사로/휠체어리프트 유무','장애인용 대변기 설치여부','장애인용 소변기 설치여부','점자블록 설치여부','유도 및 안내설비 설치여부','경보 및 피난설비 설치여부']].apply(calc_score, axis=1)



#df10
# 거리 구간별 점수 매핑
distance_scores = [
    (0, 10, 5),
    (10, 50, 4),
    (50, 200, 3),
    (200, 400, 2),
    (800, float('inf'), 1)
]

station_coords = df10[['위도', '경도']].values.tolist()

# 가장 가까운 정류장 거리 기반 점수 계산 함수
def accessibility_score(school_lat, school_lon):
    school_coord = (school_lat, school_lon)
    min_distance = float('inf')

    for station_lat, station_lon in station_coords:
        station_coord = (station_lat, station_lon)
        distance = geodesic(school_coord, station_coord).meters
        if distance < min_distance:
            min_distance = distance

    for r_min, r_max, score in distance_scores:
        if r_min <= min_distance < r_max:
            return score
    return 1  # 예외 방지용

# 접근성 점수 컬럼 생성
mdf4['공공도서관점수'] = mdf4.apply(
    lambda row: accessibility_score(row['위도'], row['경도']),
    axis=1
)



#df11
# 거리 구간별 점수 매핑
distance_scores = [
    (0, 10, 5),
    (10, 50, 4),
    (50, 200, 3),
    (200, 400, 2),
    (800, float('inf'), 1)
]
df11
station_coords = df11[['위도', '경도']].values.tolist()

# 가장 가까운 정류장 거리 기반 점수 계산 함수
def accessibility_score(school_lat, school_lon):
    school_coord = (school_lat, school_lon)
    min_distance = float('inf')

    for station_lat, station_lon in station_coords:
        station_coord = (station_lat, station_lon)
        distance = geodesic(school_coord, station_coord).meters
        if distance < min_distance:
            min_distance = distance

    for r_min, r_max, score in distance_scores:
        if r_min <= min_distance < r_max:
            return score
    return 1  # 예외 방지용

# 접근성 점수 컬럼 생성
mdf4['공원점수'] = mdf4.apply(
    lambda row: accessibility_score(row['위도'], row['경도']),
    axis=1
)



#df12
distance_scores = [
    (0, 10, 5),
    (10, 50, 4),
    (50, 200, 3),
    (200, 400, 2),
    (800, float('inf'), 1)
]

station_coords = df12[['위도', '경도']].values.tolist()

# 가장 가까운 정류장 거리 기반 점수 계산 함수
def accessibility_score(school_lat, school_lon):
    school_coord = (school_lat, school_lon)
    min_distance = float('inf')

    for station_lat, station_lon in station_coords:
        station_coord = (station_lat, station_lon)
        distance = geodesic(school_coord, station_coord).meters
        if distance < min_distance:
            min_distance = distance

    for r_min, r_max, score in distance_scores:
        if r_min <= min_distance < r_max:
            return score
    return 1  # 예외 방지용

# 접근성 점수 컬럼 생성
mdf4['복지관점수'] = mdf4.apply(
    lambda row: accessibility_score(row['위도'], row['경도']),
    axis=1
)


#df13
distance_scores = [
    (0, 10, 5),
    (10, 50, 4),
    (50, 200, 3),
    (200, 400, 2),
    (800, float('inf'), 1)
]

station_coords = df13[['위도', '경도']].values.tolist()

# 가장 가까운 정류장 거리 기반 점수 계산 함수
def accessibility_score(school_lat, school_lon):
    school_coord = (school_lat, school_lon)
    min_distance = float('inf')

    for station_lat, station_lon in station_coords:
        station_coord = (station_lat, station_lon)
        distance = geodesic(school_coord, station_coord).meters
        if distance < min_distance:
            min_distance = distance

    for r_min, r_max, score in distance_scores:
        if r_min <= min_distance < r_max:
            return score
    return 1  # 예외 방지용

# 접근성 점수 컬럼 생성
mdf4['경찰서점수']= mdf4.apply(
    lambda row: accessibility_score(row['위도'], row['경도']),
    axis=1
)


#df14
distance_scores = [
    (0, 10, 5),
    (10, 50, 4),
    (50, 200, 3),
    (200, 400, 2),
    (800, float('inf'), 1)
]

station_coords = df14[['위도', '경도']].values.tolist()

# 가장 가까운 정류장 거리 기반 점수 계산 함수
def accessibility_score(school_lat, school_lon):
    school_coord = (school_lat, school_lon)
    min_distance = float('inf')

    for station_lat, station_lon in station_coords:
        station_coord = (station_lat, station_lon)
        distance = geodesic(school_coord, station_coord).meters
        if distance < min_distance:
            min_distance = distance

    for r_min, r_max, score in distance_scores:
        if r_min <= min_distance < r_max:
            return score
    return 1  # 예외 방지용

# 접근성 점수 컬럼 생성
mdf4['병원점수']= mdf4.apply(
    lambda row: accessibility_score(row['위도'], row['경도']),
    axis=1
)



#df15
distance_scores = [
    (0, 50, 1),
    (50, 100, 2),
    (100, 200, 3),
    (200, 300, 4),
    (300, float('inf'), 5)
]

station_coords = df15[['위도', '경도']].values.tolist()

# 가장 가까운 정류장 거리 기반 점수 계산 함수
def accessibility_score(school_lat, school_lon):
    school_coord = (school_lat, school_lon)
    min_distance = float('inf')

    for station_lat, station_lon in station_coords:
        station_coord = (station_lat, station_lon)
        distance = geodesic(school_coord, station_coord).meters
        if distance < min_distance:
            min_distance = distance

    for r_min, r_max, score in distance_scores:
        if r_min <= min_distance < r_max:
            return score
    return 1  # 예외 방지용

# 접근성 점수 컬럼 생성
mdf4['유흥시설점수']= mdf4.apply(
    lambda row: accessibility_score(row['위도'], row['경도']),
    axis=1
)




#경사도 점수부여
# 기준 각도 (°)
angle_18 = np.degrees(np.arctan(1/18))  # 약 3.18
angle_12 = np.degrees(np.arctan(1/12))  # 약 4.76
angle_10 = np.degrees(np.arctan(1/10))  # 약 5.71
angle_8  = np.degrees(np.arctan(1/8))   # 약 7.13

# 점수 부여 함수
def assign_score(slope_deg):
    if pd.isna(slope_deg):
        return np.nan
    elif slope_deg < angle_18:
        return 5
    elif slope_deg < angle_12:
        return 4
    elif slope_deg < angle_10:
        return 3
    elif slope_deg < angle_8:
        return 2
    else:
        return 1

# 새 컬럼 생성
mdf4['경사도점수'] = mdf4['경사도'].apply(assign_score)




mdf4.to_csv('temp2.csv', index=False, encoding='utf-8-sig')
mdf4 = pd.read_csv('temp2.csv', encoding='utf-8-sig')



#수요는? 
#근처 구의 10세미만 장애인수 총합을 그 초등학교의 수요로보는거죠?
#2025 초6
#2026 초5
#2027 초4
#2028 초3
#2029 초2
#2030 초1

# 구 전체 수요를 보고 설치 위치 후보를 뽑은 뒤 #
#1. 현재 구별 초등학교 장애학생수 파악
#2. 구별 중학교 특수학급수확인
#3. 구별 특수학급 수요지표를 확인한뒤
#4. 그 비율대로 중학교 특수학급설치수를 나눔
#5. 구별로 몇개씩 설치할지 정함.
#6. 구별로 AHP가 높은 것들을 선정함.
#인도 초·중등교육에서 학년별 인구, 진급률, 중도탈락률, 반복률 등을 반영해 미래 학급/학생 수를 예측하는 다양한 모델(ARIMA, Cohort, Logistic 등) 비교



#AHP분석 
# AHP 가중치 정의
weights = {
    '경사도점수': 0.159,
    '통학접근성점수': 0.250,
    '공공도서관점수': 0.026,
    '공원점수': 0.033,
    '복지관점수': 0.049,
    '경찰서점수': 0.027,
    '병원점수': 0.044,
    '유흥시설점수': 0.224,
    '설치점수': 0.186  
}

# AHP점수 계산 열 추가
mdf4['AHP점수'] = mdf4[list(weights.keys())].apply(lambda row: sum(row[col] * weights[col] for col in weights), axis=1)
#마지막에 사립공립관련확인








mdf4.to_csv('temp3.csv', index=False, encoding='utf-8-sig')
mdf4 = pd.read_csv('temp3.csv', encoding='utf-8-sig')

# 후보지 주변 2~3개 초등학교 학생 수를 기반으로 수요 타당성 평가하는 2단계 접근이 가장 합리적입니다.



#53개씩 짓는다. 초등학교!!!!기준
import pandas as pd
#구별 1~7세 사이의 장애인수
dff = pd.read_csv("data/PDQCSV_26_20250520194151405.csv", encoding="CP949")
dff = dff[dff['통계시도명'] == '서울특별시']
dff = dff.query('1 <= 연령 <= 7')
dff = dff.groupby('통계시군구명')['등록장애인수'].sum().reset_index()
dff = dff.rename(columns={'통계시군구명': '행정구역'})

# 특수학급 수 필터링
dfl =df3.copy()
dfl = dfl[dfl['학제'] == '초등학교']
dfl


# 병합
dfkl = pd.merge(dff, dfl, on='행정구역', how='inner')
dfkl['구별학급당학생수요'] = dfkl['등록장애인수'] / dfkl['구별및학제별특수학급수']


# 전체 수요 대비 소수 배정
total_demand = dfkl['구별학급당학생수요'].sum()   #      나눈다. /4.5 5.4  다더한거에 54를곱해서서
dfkl['배정학급(소수)'] = dfkl['구별학급당학생수요'] / total_demand * 53

# 소수점 버림 + 정수합계 보정
dfkl['배정학급(정수)'] = np.floor(dfkl['배정학급(소수)']).astype(int)
diff = 53 - dfkl['배정학급(정수)'].sum()

# 소수점 큰 순서대로 diff만큼 1씩 추가
dfkl['소수점'] = dfkl['배정학급(소수)'] - dfkl['배정학급(정수)']
dfkl = dfkl.sort_values(by='소수점', ascending=False).reset_index(drop=True)
dfkl.loc[:diff-1, '배정학급(정수)'] += 1

# 최종 확인
dfkl = dfkl.sort_values(by='행정구역').reset_index(drop=True)
print("배정 학급 총합:", dfkl['배정학급(정수)'].sum())
print(dfkl[['행정구역', '등록장애인수', '구별및학제별특수학급수', '구별학급당학생수요', '배정학급(정수)']])

dfkl = dfkl[['행정구역','배정학급(정수)']]








#초등학교 가져온다음에 
mdf4 = mdf4[mdf4['학제'] == '초등학교']
temp = mdf4[['학교명','행정구역','AHP점수','경도','위도']]


# 결과를 저장할 리스트
selected_rows = []

# 각 행정구역별로 반복
for _, row in dfkl.iterrows():
    district = row['행정구역']
    n = row['배정학급(정수)']
    
    # 해당 행정구역의 학교 중 AHP점수 기준으로 내림차순 정렬, 상위 n개 선택
    top_schools = temp[temp['행정구역'] == district].sort_values(by='AHP점수', ascending=False).head(n)
    selected_rows.append(top_schools)

# 리스트를 하나의 DataFrame으로 결합
final_df = pd.concat(selected_rows).reset_index(drop=True)

# 결과 출력
print(final_df)



#1. 이미있는 특수학급수는 어떻게할거임?
#2. 장애유형..... 중도포기자의 85%가 중도포기다 그렇다면 중증장애인수는 85%를 곱해서 계산->이걸 고려해야되나
final_df.to_csv('temp4.csv', index=False, encoding='utf-8-sig')


# 연령별 등록장애인수 집계
age_counts = dff.groupby('연령')['등록장애인수'].sum().reset_index()

# 시각화
plt.figure(figsize=(10,6))
plt.bar(age_counts['연령'], age_counts['등록장애인수'], color='skyblue')
plt.xlabel('연령')
plt.ylabel('등록장애인수')
plt.title('연령별 등록장애인수')
plt.xticks(age_counts['연령'])
plt.tight_layout()
plt.show()




#학교시각화 orange
mdf4 = mdf4[mdf4['학제'] == '초등학교']
seoul_center = [37.5665, 126.9780]

m = folium.Map(location=seoul_center, zoom_start=12)

for _, row in mdf4.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,  # 픽셀 기준이므로 적절한 값으로 고정
        color='orange',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)
m





import folium

# 서울 중심 좌표
seoul_center = [37.5665, 126.9780]

# 지도 생성
m = folium.Map(location=seoul_center, zoom_start=12)

# 학교 위치 시각화
for _, row in mdf4[mdf4['학제'] == '초등학교'].iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,
        color='orange',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

# 클릭 시 위도/경도 팝업 표시 기능 추가
m.add_child(folium.LatLngPopup())

# 지도 출력
m








#1.df9 전철버스
import folium

# 중심 좌표 설정 (서울시 중심)
seoul_center = [37.5665, 126.9780]

# 지도 객체 생성
m = folium.Map(location=seoul_center, zoom_start=12)

for _, row in df9.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,  # 픽셀 기준이므로 적절한 값으로 고정
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)
m

#2.공공도서관 df10
import folium

# 중심 좌표 설정 (서울시 중심)
seoul_center = [37.5665, 126.9780]

# 지도 객체 생성
m = folium.Map(location=seoul_center, zoom_start=12)

for _, row in df10.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,  # 픽셀 기준이므로 적절한 값으로 고정
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)
m


#3. 주요공원 df11

import folium

# 중심 좌표 설정 (서울시 중심)
seoul_center = [37.5665, 126.9780]

# 지도 객체 생성
m = folium.Map(location=seoul_center, zoom_start=12)

for _, row in df11.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,  # 픽셀 기준이므로 적절한 값으로 고정
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)
m

#4. 장애인인사회복지시설 df12
m = folium.Map(location=seoul_center, zoom_start=12)

for _, row in df12.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,  # 픽셀 기준이므로 적절한 값으로 고정
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)
m

#5. 경찰서  df13
m = folium.Map(location=seoul_center, zoom_start=12)

for _, row in df13.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,  # 픽셀 기준이므로 적절한 값으로 고정
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)
m

#6. 병원    df14
m = folium.Map(location=seoul_center, zoom_start=12)

for _, row in df14.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,  # 픽셀 기준이므로 적절한 값으로 고정
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)
m

#7. 유흥주점  df15
m = folium.Map(location=seoul_center, zoom_start=12)

for _, row in df15.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,  # 픽셀 기준이므로 적절한 값으로 고정
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)
m


# 1. 데이터프레임 예시 (실제 데이터에 맞게 수정 가능)


# 2. 서울시 구별 GeoJSON 불러오기
url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_municipalities_geo_simple.json"
geo_data = requests.get(url).json()

# 3. 지도 생성
m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

# 4. 히트맵 추가
folium.Choropleth(
    geo_data=geo_data,
    name='choropleth',
    data=dfkl,
    columns=['행정구역', '구별학급당학생수요'],
    key_on='feature.properties.name',
    fill_color='Reds',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='구별 특수학급당 학생 수요'
).add_to(m)

# 5. 지도 출력



dfkl.to_csv('asdf.csv', index=False, encoding='utf-8-sig')
# 막대그래프 시각화
plt.figure(figsize=(8,5))
plt.bar(dfkl['행정구역'], dfkl['구별학급당학생수요'], color='skyblue')
plt.title('구별학급당학생수요')
plt.xlabel('행정구역')
plt.ylabel('학급당학생수요')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# 데이터 준비
data = {
    '교통수단': ['통학버스', '자가용', '대중교통', '도보', '기숙사', '기타'],
    '학생수': [2193, 1432, 311, 243, 117, 2]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 막대그래프 시각화
plt.figure(figsize=(8,5))
plt.bar(df['교통수단'], df['학생수'], color='skyblue')
plt.title('교통수단별 학생수')
plt.xlabel('교통수단')
plt.ylabel('학생수')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()






#경사도 시각화
# 서울 중심
center_lat, center_lon = 37.5665, 126.9780

# folium 지도 생성 (배경: Vworld 지도 - 한국어 포함)
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)

# ✅ 한국어 지명 포함된 타일 추가 (Vworld)
folium.TileLayer(
    tiles='http://xdworld.vworld.kr:8080/2d/Base/202002/{z}/{x}/{y}.png',
    attr='VWorld Map',
    name='Vworld Base Map',
    overlay=False,
    control=True
).add_to(m)

# HeatMap 데이터 준비
heat_data = [
    [row['위도'], row['경도'], row['경사도']]
    for idx, row in mdf4.iterrows()
    if not pd.isnull(row['경사도'])
]

# 보라색 gradient 설정
purple_gradient = {
    0.2: '#e0d3f5',
    0.4: '#c59df2',
    0.6: '#a365e5',
    0.8: '#722fb7',
    1.0: '#4b0082'
}

# 히트맵 추가
HeatMap(
    heat_data,
    radius=15,
    blur=25,
    min_opacity=0.3,
    max_zoom=1,
    gradient=purple_gradient
).add_to(m)

# 지도 출력
m





m = folium.Map(location=seoul_center, zoom_start=12)

# 학교 위치 시각화
for _, row in final_df.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=3,
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

# 클릭 시 위도/경도 팝업 표시 기능 추가
m.add_child(folium.LatLngPopup())

# 지도 출력
m
# 3. 데이터 불러오기
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
temp3 = pd.read_csv("/content/gdrive/MyDrive/temp3.csv",encoding="UTF-8")

# 4-1. 행정구역별 학교 수 막대그래프
plt.figure(figsize=(12, 6))
school_counts = temp3['행정구역'].value_counts().sort_values(ascending=False)
sns.barplot(x=school_counts.index, y=school_counts.values, palette="Blues_d")
plt.title("행정구역별 선정된 학교 수")
plt.xlabel("행정구역")
plt.ylabel("학교 수")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4-2. 행정구역별 평균 AHP 점수 막대그래프
plt.figure(figsize=(12, 6))
mean_ahp = temp3.groupby('행정구역')['AHP점수'].mean().sort_values(ascending=False)
sns.barplot(x=mean_ahp.index, y=mean_ahp.values, palette="Oranges_d")
plt.title("행정구역별 평균 AHP 점수 (높은 순)")
plt.xlabel("행정구역")
plt.ylabel("평균 AHP 점수")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()