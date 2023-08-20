import requests
import json
import pandas as pd
import xmltodict
import os
# 버스 데이터 수집

    # 버스 승객수 월별 시간별별
def get_bus_passenger_time(year_month):
    import requests
    import json
    import pandas as pd
    import xmltodict
    import os
    num=1
    bus_df=pd.DataFrame()    
        
    while  True:
        
        url = f'http://openapi.seoul.go.kr:8088/737a486d62636d7937327365785962/xml/CardBusTimeNew/{num}/{num+999}/{year_month}/'
        try:
            response = requests.get(url)
            dict_type = xmltodict.parse(response.text)
            bus_df= pd.DataFrame(dict_type['CardBusTimeNew']['row'])
            if not os.path.exists('버스_월별_시간별.csv'):
                bus_df.to_csv('버스_월별_시간별.csv', index=False, mode='w', encoding='utf-8-sig')
            else:
                bus_df.to_csv('버스_월별_시간별.csv', index=False,  mode='a', encoding='utf-8-sig', header=False)

            # 확인용
            print(f'try {year_month}, {num} 결과완료')
            
            num += 1000
        except KeyError:
                if not os.path.exists('버스_월별_시간별.csv'):
                    bus_df.to_csv('버스_월별_시간별.csv', index=False, mode='w', encoding='utf-8-sig')
                else:
                    bus_df.to_csv('버스_월별_시간별.csv', index=False,  mode='a', encoding='utf-8-sig', header=False)
                print(f'error {year_month} 끝')
                break

    return 

    # 날짜 계산


def date_range(start, end):
    from datetime import datetime, timedelta
    start = datetime.strptime(start, "%Y%m%d")
    end = datetime.strptime(end, "%Y%m%d")
    dates = [(start + timedelta(days=i)).strftime("%Y%m%d") for i in range((end-start).days+1)]
    return dates

    # 일별 버스 승객 총 승하차수 수집
def get_bus_passenger_day(year_month):
    import requests
    import json
    import pandas as pd
    import xmltodict
    import os
    num=1
    bus_station_df=pd.DataFrame()    
        
    while  True:
        url = f'http://openapi.seoul.go.kr:8088/737a486d62636d7937327365785962/xml/CardBusStatisticsServiceNew/{num}/{num+999}/{year_month}/'
        
        try:
            response = requests.get(url)
            dict_type = xmltodict.parse(response.text)
            bus_station_df= pd.DataFrame(dict_type['CardBusStatisticsServiceNew']['row'])
            print(f'{num}')
            if not os.path.exists('충모_정류장별_승하차.csv'):
                bus_station_df.to_csv('충모_정류장별_승하차.csv', index=False, mode='w', encoding='utf-8-sig')
            else:
                bus_station_df.to_csv('충모_정류장별_승하차.csv', index=False,  mode='a', encoding='utf-8-sig', header=False)

            # 확인용
            print(f'try {year_month}, {num} 결과완료')
            
            num += 1000
        except KeyError:   # 마지막 몇개 안남았을때
                if not os.path.exists('충모_정류장별_승하차_3.csv'):
                    bus_station_df.to_csv('충모_정류장별_승하차.csv', index=False, mode='w', encoding='utf-8-sig')
                else:
                    bus_station_df.to_csv('충모_정류장별_승하차.csv', index=False,  mode='a', encoding='utf-8-sig', header=False)
                print(f'error {year_month} 끝')
                break

    return 


# 카카오 API에서 버스 정류장 행정동-> 법정동 주소 수집
def busstation_XY_dong(Bus_Location_df,list):
    from urllib.parse import urlparse
    num=0
    for i in list:
        # list에서 X좌표와 Y좌표 가져오기
        x= i[0]
        y= i[1]

        url = f"https://dapi.kakao.com/v2/local/geo/coord2regioncode?x={x}&y={y}"
        API_KEY = '45635cd5acc8c9d86c84895fcfd0e313'
        headers = {'Authorization': 'KakaoAK {}'.format(API_KEY)}
        api_test = requests.get(url,headers=headers)
        data = requests.get(urlparse(url).geturl(),headers=headers).json()

        # X,Y좌표에 해당하는 법정동과 행정동 데이터를 데이터프레임에 한행씩 추가
        Bus_Location_df.loc[num,'법정동']= data['documents'][0]['region_3depth_name']
        Bus_Location_df.loc[num,'행정동']= data['documents'][1]['region_3depth_name']

        # 행 번호 1씩 추가
        num+=1
        print(f'{num}번째')  # 확인용
    return Bus_Location_df




# kakao api에서 버스 정류장 별 infra 추출-> dataframe 반환
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