import requests
import json
import pandas as pd
import xmltodict
import os
import glob

# 버스 데이터 수집

# 버스 승객수 월별 시간별별
def get_bus_passenger_time(year_month):
    import requests
    import json
    import pandas as pd
    import xmltodict
    import os

    num = 1
    bus_df = pd.DataFrame()

    while True:

        url = f"http://openapi.seoul.go.kr:8088/737a486d62636d7937327365785962/xml/CardBusTimeNew/{num}/{num+999}/{year_month}/"
        try:
            response = requests.get(url)
            dict_type = xmltodict.parse(response.text)
            bus_df = pd.DataFrame(dict_type["CardBusTimeNew"]["row"])
            if not os.path.exists("버스_월별_시간별.csv"):
                bus_df.to_csv(
                    "버스_월별_시간별.csv", index=False, mode="w", encoding="utf-8-sig"
                )
            else:
                bus_df.to_csv(
                    "버스_월별_시간별.csv",
                    index=False,
                    mode="a",
                    encoding="utf-8-sig",
                    header=False,
                )

            # 확인용
            print(f"try {year_month}, {num} 결과완료")

            num += 1000
        except KeyError:
            if not os.path.exists("버스_월별_시간별.csv"):
                bus_df.to_csv(
                    "버스_월별_시간별.csv", index=False, mode="w", encoding="utf-8-sig"
                )
            else:
                bus_df.to_csv(
                    "버스_월별_시간별.csv",
                    index=False,
                    mode="a",
                    encoding="utf-8-sig",
                    header=False,
                )
            print(f"error {year_month} 끝")
            break

    return

    # 날짜 계산


def date_range(start, end):
    from datetime import datetime, timedelta

    start = datetime.strptime(start, "%Y%m%d")
    end = datetime.strptime(end, "%Y%m%d")
    dates = [
        (start + timedelta(days=i)).strftime("%Y%m%d")
        for i in range((end - start).days + 1)
    ]
    return dates

    # 일별 버스 승객 총 승하차수 수집


def get_bus_passenger_day(year_month):
    import requests
    import json
    import pandas as pd
    import xmltodict
    import os

    num = 1
    bus_station_df = pd.DataFrame()

    while True:
        url = f"http://openapi.seoul.go.kr:8088/737a486d62636d7937327365785962/xml/CardBusStatisticsServiceNew/{num}/{num+999}/{year_month}/"

        try:
            response = requests.get(url)
            dict_type = xmltodict.parse(response.text)
            bus_station_df = pd.DataFrame(
                dict_type["CardBusStatisticsServiceNew"]["row"]
            )
            print(f"{num}")
            if not os.path.exists("충모_정류장별_승하차.csv"):
                bus_station_df.to_csv(
                    "충모_정류장별_승하차.csv", index=False, mode="w", encoding="utf-8-sig"
                )
            else:
                bus_station_df.to_csv(
                    "충모_정류장별_승하차.csv",
                    index=False,
                    mode="a",
                    encoding="utf-8-sig",
                    header=False,
                )

            # 확인용
            print(f"try {year_month}, {num} 결과완료")

            num += 1000
        except KeyError:  # 마지막 몇개 안남았을때
            if not os.path.exists("충모_정류장별_승하차_3.csv"):
                bus_station_df.to_csv(
                    "충모_정류장별_승하차.csv", index=False, mode="w", encoding="utf-8-sig"
                )
            else:
                bus_station_df.to_csv(
                    "충모_정류장별_승하차.csv",
                    index=False,
                    mode="a",
                    encoding="utf-8-sig",
                    header=False,
                )
            print(f"error {year_month} 끝")
            break

    return


# 카카오 API에서 버스 정류장 행정동-> 법정동 주소 수집
def busstation_XY_dong(Bus_Location_df, list):
    from urllib.parse import urlparse

    num = 0
    for i in list:
        # list에서 X좌표와 Y좌표 가져오기
        x = i[0]
        y = i[1]

        url = f"https://dapi.kakao.com/v2/local/geo/coord2regioncode?x={x}&y={y}"
        API_KEY = "45635cd5acc8c9d86c84895fcfd0e313"
        headers = {"Authorization": "KakaoAK {}".format(API_KEY)}
        api_test = requests.get(url, headers=headers)
        data = requests.get(urlparse(url).geturl(), headers=headers).json()

        # X,Y좌표에 해당하는 법정동과 행정동 데이터를 데이터프레임에 한행씩 추가
        Bus_Location_df.loc[num, "법정동"] = data["documents"][0]["region_3depth_name"]
        Bus_Location_df.loc[num, "행정동"] = data["documents"][1]["region_3depth_name"]

        # 행 번호 1씩 추가
        num += 1
        print(f"{num}번째")  # 확인용
    return Bus_Location_df


# kakao api에서 버스 정류장 별 infra 추출-> dataframe 반환
def kakao_infra(key, category_code_list):
    bus_station_XY = pd.read_csv(glob.glob('./csv/bus_station_XY_final.csv')[0])
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    kakao_key = key

    # 유치원, 마트, 식당, 학교, 대학, 지하철, 투어, 카페, 병원,
    # 문화시설, 대학병원, 공공기관,

    for cat in category_code_list:
        bus_station_XY[cat] = ""

        for i in range(len(bus_station_XY)):
            params = {
                "category_group_code": cat,
                "x": bus_station_XY.loc[i, "X좌표"],
                "y": bus_station_XY.loc[i, "Y좌표"],
                "page": 7,
                "radius": 500,
            }
            header = {"Authorization": f"KakaoAK {kakao_key}"}
            bus = requests.get(url=url, params=params, headers=header).json()
            print(cat, i)
            lst = []
            bus_station_XY.loc[i, cat] = bus["meta"]["total_count"]
    
    col_rename = {'AC5': 'academy_cnt',
                    'PS3': 'kindergarten_cnt',
                    'MT1': 'mart_cnt',
                    'FD6':'restaurant_cnt',
                    'SC4': 'school_cnt',
                    'SW8': 'subway_cnt',
                    'AT4': 'tour_cnt',
                    'CE7' :'cafe_cnt',
                    'HP8': 'hospital_cnt',
                    'CT1': 'culture_cnt',
                    'PO3': 'public_office_cnt',
                    'CS2': 'convenience_cnt',
                    'PK6':'park_cnt',
                    'OL7':'gas_cnt',
                    'BK9':'bank_cnt',
                    'AG2':'estate_agent_cnt',
                    'AD5':'accommodation_cnt',
                    }


    for old_name, new_name in col_rename.items():
        if old_name in bus_station_XY.columns:
            bus_station_XY.rename(columns={old_name: new_name}, inplace=True)


    bus_station_XY.rename(columns=col_rename, inplace=True)

    return bus_station_XY


# 기상청 API에서 기상 데이터 수집


# 시간 및 날짜
def date_time():
    """
    기상청API에 적용할 시간을 정합니다.
    기본은 오늘 날짜이며, 23시 10분이 지나지 않았으면 전날 23:10분 예보로 지정됩니다.
    hour: 시간
    minute: 분
    """
    from datetime import date, time, datetime, timedelta

    now = datetime.now()
    today = datetime.today().strftime("%Y%m%d")
    y = date.today() - timedelta(days=1)
    yesterday = y.strftime("%Y%m%d")

    if now.hour >= 23 and now.minute > 10:  # base_time와 base_date 구하는 함수
        base_time = "2310"
        base_date = today
    else:
        base_time = "2310"
        base_date = yesterday

    return base_time, base_date

    # 좌표 설정


def xy():
    """
    기상을 측정할 좌표를 설정
    """
    nx = 60  # 위도와 경도를 x,y좌표로 변경
    ny = 127
    return nx, ny

    # 기상데이터 전처리 파이프라인


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WeatherTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.training_features = ["TMP", "VEC", "WSD", "PCP", "REH"]

    def fit(self, item):
        return self

    def transform(self, item):
        item_df = pd.DataFrame(item)
        grouped_data = (
            item_df.groupby(["fcstTime", "category"])[["fcstValue"]]
            .aggregate("first")
            .reset_index()
        )
        result_data = (
            grouped_data.groupby(["fcstTime", "category"])["fcstValue"]
            .aggregate("first")
            .unstack()
        )
        result_data["PCP"][result_data["PCP"] == "강수없음"] = 0
        result_data = result_data.loc[:, self.training_features].astype(float)

        return result_data

    # 최종 기상데이터 호출 API


def weather_api():
    import requests
    from datetime import date, time, datetime, timedelta
    import time
    from urllib.parse import unquote
    from urllib.parse import quote_plus
    from urllib.parse import unquote_plus
    from urllib.parse import urlencode

    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"

    serviceKey = "GdenAOBzMnIcPEJ6tyQqFbbSrAB%2B81vv7NDRBKe%2FbjomnxP5MpRkqe877BlZoLFm%2FqF35QGJcXvzteALmxDurg%3D%3D"
    serviceKeyDecoded = unquote(serviceKey, "UTF-8")

    base_time, base_date = date_time()

    nx, ny = xy()

    queryParams = "?" + urlencode(
        {
            quote_plus("serviceKey"): serviceKeyDecoded,
            quote_plus("base_date"): base_date,
            quote_plus("base_time"): base_time,
            quote_plus("nx"): nx,
            quote_plus("ny"): ny,
            quote_plus("dataType"): "json",
            quote_plus("numOfRows"): "289",
        }
    )  # 페이지로 안나누고 한번에 받아오기 위해 numOfRows=120으로 설정해주었다
    # 기상데이터는 12개로 12단위로 곱해서 row값을 정할 것, 24시간으로 하면 +1(최고기온떄문에)

    # 값 요청 (웹 브라우저 서버에서 요청 - url주소와 파라미터)
    res = requests.get(url + queryParams, verify=False)  # verify=False이거 안 넣으면 에러남ㅜㅜ
    items = res.json().get("response").get("body").get("items")  # 데이터들 아이템에 저장

    transformer_test = WeatherTransformer().fit_transform(items["item"])

    return transformer_test
def get_sgis_accessToken():
    """_summary_
    SGIS 인증 토큰 가져오는 코드

    Returns:
        _type_: _description_
    """
  SGIS_API = "https://sgisapi.kostat.go.kr/OpenAPI3/auth/authentication.json?consumer_key={}&consumer_secret={}"
  C_KEY = "5a856ac23eac44689cb4" # 서비스 ID
  C_SECRET = "3f8798ac490345dca660" # 보안
  response = requests.get(REAL_TIME_API.format(C_KEY, C_SECRET))
  response.status_code
  data = response.json()
  token = data['result']['accessToken'] # 인증 토큰

  return token


def get_population_data():
    """_summary_
    인구, 가구수 데이터 뽑기

    Returns:
        _type_: _description_
    """
    
    POPULATION_TOTAL_API = 'https://sgisapi.kostat.go.kr/OpenAPI3/stats/population.json'
    POPULATION_15to64_API = "https://sgisapi.kostat.go.kr/OpenAPI3/stats/searchpopulation.json"
    token = getAccessToken()

    total_population_api_url = f"{API}?accessToken={token}&year=2020&adm_cd=11&low_search=2"
    population_15to64_api_url = f"{POPULATION_15to64_API}?accessToken={token}&year=2020&adm_cd=11&low_search=2&age_type=23"
    
    response = requests.get(api_url)
    population_data = response.json()
    population_data_json = json_normalize(population_data['result'])

    df_population_result = pd.DataFrame(population_data_json)
    df_population_result = df_population_result.rename(columns={"population": f"population_code_15to64"}).drop(axis = 1, columns = ["adm_nm", "avg_age"])

    df_population_result = pd.DataFrame(population_data_json)
    df_population_result = df_population_result.loc[:, ['adm_cd', 'adm_nm', 'tot_family', 'tot_ppltn', 'corp_cnt', 'employee_cnt']]
    
    df_total_population = pd.merge(df_total_population, df_population_age, how='left', on=['adm_cd'])
    df_total_population

    return df_population_result