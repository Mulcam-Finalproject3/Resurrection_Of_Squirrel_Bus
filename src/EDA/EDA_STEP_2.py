def haversine_distance(lat1, lon1, lat2, lon2):
    import pandas as pd
    import math

    # 지구 반지름 (단위: km)
    radius = 6371.0

    # 각도를 라디안으로 변환
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 위도와 경도의 차이 계산
    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad

    # Haversine 공식 적용
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 실제 거리 계산 (단위: km)
    distance = radius * c

    return distance


def congestion_hour_standard(재차인원_df, per_bus_num: int, standard: int = None):
    a = round(
        재차인원_df[["6시재차인원", "7시재차인원", "8시재차인원", "9시재차인원"]].max() / (per_bus_num * 30), 1
    )
    return a[a > standard]


def reduce_congestion_bus_num(재차인원_df, hour_coulmns: list, per_bus_num: int):
    for bus_num in range(20):
        max_value = 재차인원_df[hour_coulmns].max().max()
        if round(max_value / ((int(bus_num) + per_bus_num) * 30), 1) < 21:
            return bus_num


def max_passenger(재차인원_df, per_bus_num, month_operation_day: int = 30):
    return round(
        재차인원_df[["6시재차인원", "7시재차인원", "8시재차인원", "9시재차인원"]].max()
        / (per_bus_num * month_operation_day),
        1,
    )
