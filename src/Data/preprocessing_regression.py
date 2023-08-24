import pandas as pd

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, isnan, when, count, hour, to_timestamp, dayofweek
from datetime import datetime

spark = SparkSession.builder.master("local").appName("MyApp")\
    .config("spark.executor.memory", "16g").config("spark.executor.cores", "4")\
    .getOrCreate()

path = './Data/csv'

# 최종 선정 노선
ky_df = spark.read.option("encoding", "UTF-8").csv(f'{path}/가양.csv', inferSchema=True, header=True)
ky_list = ky_df.collect()
ky_node_list = [row["NODE_ID"] for row in ky_list]

# 시간대 별 승하차 데이터
bus_06_07_df = spark.read.csv(f'{path}/AAA_06_07.csv', inferSchema=True, header=True)
bus_07_08_df = spark.read.csv(f'{path}/AAA_07_08.csv', inferSchema=True, header=True)
bus_08_09_df = spark.read.csv(f'{path}/AAA_08_09.csv', inferSchema=True, header=True)
bus_09_10_df = spark.read.csv(f'{path}/AAA_09_10.csv', inferSchema=True, header=True)

bus_06_07_df = bus_06_07_df.withColumnRenamed("ride_06_07", "RIDE").withColumnRenamed("alight_06_07", "ALIGHT").withColumn("Hour", hour("USE_DT"))
bus_07_08_df = bus_07_08_df.withColumnRenamed("ride_07_08", "RIDE").withColumnRenamed("alight_07_08", "ALIGHT").withColumn("Hour", hour("USE_DT"))
bus_08_09_df = bus_08_09_df.withColumnRenamed("ride_08_09", "RIDE").withColumnRenamed("alight_08_09", "ALIGHT").withColumn("Hour", hour("USE_DT"))
bus_09_10_df = bus_09_10_df.withColumnRenamed("ride_09_10", "RIDE").withColumnRenamed("alight_09_10", "ALIGHT").withColumn("Hour", hour("USE_DT"))

bus_06_07_df.createOrReplaceTempView('bus_06_07_df')
bus_07_08_df.createOrReplaceTempView('bus_07_08_df')
bus_08_09_df.createOrReplaceTempView('bus_08_09_df')
bus_09_10_df.createOrReplaceTempView('bus_09_10_df')

union_query = """
SELECT *
FROM bus_06_07_df
UNION ALL
SELECT *
FROM bus_07_08_df
UNION ALL
SELECT *
FROM bus_08_09_df
UNION ALL
SELECT *
FROM bus_09_10_df
"""
merged_df = spark.sql(union_query)

def get_holiday():

    import requests
    from datetime import datetime


    years = [2020, 2021]
    SERVICE_KEY = "n8NfLiNxxjBIMps28fLbd259sMTpTGn%2BMHumLN0llqi8EMYDYnwhZQU4yoIEymUZ%2FP4qbe34kP3yPcEbzZUlRw%3D%3D"

    # 공휴일 API
    API = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo?solYear={}&ServiceKey={}&_type=json&numOfRows=100"

    date = []
    res = []

    for year in years:
        res_year = requests.get(API.format(year, SERVICE_KEY)).json()["response"]["body"]["items"]["item"]
        res += res_year

    for i in range(len(res)):
        date.append(res[i]['locdate'])

    date_list = [datetime.strptime(str(d), "%Y%m%d").strftime("%Y-%m-%d") for d in date]

    return date_list

def delete_weekend_and_holiday(dataframe):
    dataframe = dataframe.withColumn("date", to_date("USE_DT"))\
                        .withColumn("week", dayofweek("date"))
    
    dataframe = dataframe.withColumn("dayofweek", dayofweek(col("date")))
    
    values_to_remove = ['1', '7']
    dataframe = dataframe.filter(~col("dayofweek").isin(values_to_remove))
    
    date_list = get_holiday()
    dataframe = dataframe.filter(~col("date").isin(date_list))
    
    return dataframe

def get_final_route_list(dataframe):
    dataframe = dataframe.filter(col("STND_BSST_ID").isin(ky_node_list))

def rename_and_drop_columns(dataframe, final_route_dataframe):
    drop_columns = ["_c0", "Unnamed: 0", "USE_DT", "BUS_ROUTE_ID", "BUS_ROUTE_NO","BUS_ROUTE_NM", "BUS_STA_NM", "NODE_ID", "date", "week", "dayofweek", "풍향(deg)", "습도(%)"]
    
    dataframe = dataframe.drop(*drop_columns)
    
    rename_df = dataframe.withColumnRenamed("기온(°C)", "tmp").withColumnRenamed("풍속(m/s)", "wsd").withColumnRenamed("강수량(mm)", "pcp")\
                    .withColumnRenamed("위도", "lat").withColumnRenamed("경도", "lon")
    
    rename_df = rename_df.na.drop(subset=["lat", "lon", "tmp", "wsd", "pcp"])
    
    rename_df = rename_df.join(final_route_dataframe, rename_df.STND_BSST_ID == final_route_dataframe.NODE_ID, "inner")

    drop_cols = ["STND_BSST_ID", "법정동_구",  "BSST_ARS_NO", "법정동", "법정동코드", "lat", "lon", "행정동", "행정동코드", "ARS_ID", "정류소명"]
    
    rename_df = rename_df.drop(*drop_cols)
    rename_df = rename_df.withColumnRenamed("Y좌표", "lat").withColumnRenamed("X좌표", "lon")
    rename_df.withColumn("NODE_ID", col("NODE_ID").cast("Integer"))

    ride_df = rename_df.select(["lon", "lat", "tmp", "wsd", "pcp", "NODE_ID", "Hour", "RIDE"])\
            .groupBy(["lon", "lat", "tmp", "wsd", "pcp", "Hour", "NODE_ID"])\
            .sum("RIDE")
    alight_df = rename_df.select(["lon", "lat", "tmp", "wsd", "pcp", "Hour", "NODE_ID", "ALIGHT"])\
            .groupBy(["lon", "lat", "tmp", "wsd", "pcp", "Hour", "NODE_ID"])\
            .sum("ALIGHT")
    
    return ride_df, alight_df

# 전처리 완료 데이터 저장 (parquet 형식)
def save_data(ride_df, alight_df):
    ride_df.write.format("parquet").save(f"{path}/ride_df/")
    alight_df.write.format("parquet").save(f"{path}/alight_df/")

# 전처리 완료 데이터 불러오기
def load_data(file_name):
    import pandas as pd

    dataframe = pd.read_parquet(f"{path}/{file_name}/")

    return dataframe

spark.stop()