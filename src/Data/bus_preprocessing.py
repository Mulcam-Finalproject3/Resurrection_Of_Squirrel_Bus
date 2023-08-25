import pandas as pd
# 월별 시간별 버스 승하차 승객수 데이터 전처리 함수   

def bus_month_time_passenger(bus_station_time_df):


    # 공통된 컬럼들 뽑기
    basic_cloumns=bus_station_time_df.columns[0:6].tolist()

    # 컬럼을 리스트로 
    total_columns= bus_station_time_df.columns.tolist()

    # 승하차와 관련된 컬럼들만 추출 
    ride_columns=[i for i in total_columns if 'RIDE' in i]
    alight_columns=[i for i in total_columns if 'ALIGHT' in i]

    # 시간별 승하차 승객 수 데이터 프레임
    ride_df =bus_station_time_df[ride_columns] 
    alight_df =bus_station_time_df[alight_columns]

    # 전체 승하차 승객 수 
    ride_df['승차인원 수']=ride_df.sum(axis=1)
    alight_df['하차인원 수']=alight_df.sum(axis=1)

    ride_rate=['ride_rate_06_07','ride_rate_07_08','ride_rate_08_09','ride_rate_09_10']
    alight_rate=['alight_rate_06_07','alight_rate_07_08','alight_rate_08_09','alight_rate_09_10']
    ride_time=ride_df.columns[6:10].tolist()
    alight_time=alight_df.columns[6:10].tolist()

    for ride_rate,alight_rate, ride_time, alight_time in zip(ride_rate,alight_rate,ride_time,alight_time):
        ride_df[ride_rate]=ride_df[ride_time]/ ride_df['승차인원 수']
        alight_df[alight_rate]=alight_df[alight_time]/ alight_df['하차인원 수']

    # concat할 컬럼들 뽑기
    ride_A_list=ride_df.columns[6:10].tolist()
    ride_B_list=ride_df.columns[24:].tolist()
    alight_A_list=alight_df.columns[6:10].tolist()
    alight_B_list=alight_df.columns[24:].tolist()

    total_bus_time_df=pd.concat([bus_station_time_df[basic_cloumns],ride_df[ride_A_list], ride_df[ride_B_list],alight_df[alight_A_list], alight_df[alight_B_list]],axis=1)

    # 06~10시까지 승하차 인원과 승하차 비율
    total_bus_time_df.sort_values(by="USE_MON")
    total_bus_time_df['월별_노선_정류장_ID'] =total_bus_time_df['USE_MON'].astype(str)+'_'+ total_bus_time_df['BUS_ROUTE_NO'].astype(str) + '_' + total_bus_time_df['STND_BSST_ID'].astype(str)

    # 불필요한 컬럼 제거 
    ride_drop_list=total_bus_time_df.columns[6:10].tolist()
    alight_drop_list=total_bus_time_df.columns[15:19].tolist()
    drop_list = ride_drop_list + alight_drop_list
    total_bus_time_df=total_bus_time_df.drop(drop_list,axis=1)
    return total_bus_time_df


# 일별 버스승객 승하차 데이터에서 merge할 컬럼생성 및 필요한 컬럼만 추출
def daily_bus_passenger(daily_bus_passenger_data):
    daily_bus_passenger_data['월별_노선_정류장_ID']=daily_bus_passenger_data['USE_DT'].astype(str).str[:-2]+'_'\
    +daily_bus_passenger_data['BUS_ROUTE_NO'].astype(str)+'_'+daily_bus_passenger_data['STND_BSST_ID'].astype(str)

    contain_list=['RIDE_PASGR_NUM','ALIGHT_PASGR_NUM','월별_노선_정류장_ID']

    daily_bus_passenger_data=daily_bus_passenger_data[contain_list]
    return daily_bus_passenger_data

# 일별 시간별 승객 승하차수 추정치 구하기 
def calculate_passenger(daily_time_bus_passenger):
    daily_time_bus_passenger['ride_06_07'] = daily_time_bus_passenger['RIDE_PASGR_NUM']*daily_time_bus_passenger['ride_rate_06_07']
    daily_time_bus_passenger['ride_07_08'] = daily_time_bus_passenger['RIDE_PASGR_NUM']*daily_time_bus_passenger['ride_rate_07_08']
    daily_time_bus_passenger['ride_08_09'] = daily_time_bus_passenger['RIDE_PASGR_NUM']*daily_time_bus_passenger['ride_rate_08_09']
    daily_time_bus_passenger['ride_09_10'] = daily_time_bus_passenger['RIDE_PASGR_NUM']*daily_time_bus_passenger['ride_rate_09_10']

    daily_time_bus_passenger['alight_06_07'] = daily_time_bus_passenger['ALIGHT_PASGR_NUM']*daily_time_bus_passenger['alight_rate_06_07']
    daily_time_bus_passenger['alight_07_08'] = daily_time_bus_passenger['ALIGHT_PASGR_NUM']*daily_time_bus_passenger['alight_rate_07_08']
    daily_time_bus_passenger['alight_08_09'] = daily_time_bus_passenger['ALIGHT_PASGR_NUM']*daily_time_bus_passenger['alight_rate_08_09']
    daily_time_bus_passenger['aligth_09_10'] = daily_time_bus_passenger['ALIGHT_PASGR_NUM']*daily_time_bus_passenger['alight_rate_09_10']

    drop_col = ['RIDE_PASGR_NUM',
    'ALIGHT_PASGR_NUM', 'ride_rate_06_07', 'ride_rate_07_08',
    'ride_rate_08_09', 'ride_rate_09_10', 'alight_rate_06_07',
    'alight_rate_07_08', 'alight_rate_08_09', 'alight_rate_09_10']

    daily_time_bus_passenger_final = daily_time_bus_passenger.drop(drop_col, axis=1)
    return daily_time_bus_passenger_final