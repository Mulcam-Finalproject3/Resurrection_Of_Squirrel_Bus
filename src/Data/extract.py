
import requests
import json
import pandas as pd

# 함수화
# MT1	대형마트
# CS2	편의점
# PS3	어린이집, 유치원
# SC4	학교
# AC5	학원
# PK6	주차장
# OL7	주유소, 충전소
# SW8	지하철역
# BK9	은행
# CT1	문화시설
# AG2	중개업소
# PO3	공공기관
# AT4	관광명소
# AD5	숙박
# FD6	음식점
# CE7	카페
# HP8	병원
# PM9	약국

# category_lst=  ['MT1','CS2','PS3','SC4','AC5','SW8','BK9','CT1','PO3','AT4','CE7','HP8']


def kakao_infra(key, category_code_list):
    bus_station_XY = pd.read_csv('../data/bus_station_XY_final.csv')
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    kakao_key = key

    # 유치원, 마트, 식당, 학교, 대학, 지하철, 투어, 카페, 병원, 
    # 문화시설, 대학병원, 공공기관, 


    for cat in category_code_list:
        bus_station_XY[cat]=''

        for i in range(len(bus_station_XY[:10])):
            params = {'category_group_code': cat,
                        'x': bus_station_XY.loc[i,'X좌표'],
                        'y': bus_station_XY.loc[i,'Y좌표'],
                        'page': 7,
                        'radius':500
                    }
            header = {'Authorization': f'KakaoAK {kakao_key}'}
            bus = requests.get(url=url,
                            params= params,
                            headers = header).json()
            print(bus)
            lst = []
            bus_station_XY.loc[i, cat] = bus['meta']['total_count']

    return bus_station_XY