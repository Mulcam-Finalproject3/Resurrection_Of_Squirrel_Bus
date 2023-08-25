

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
    a = math.sin(d_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 실제 거리 계산 (단위: km)
    distance = radius * c

    return distance