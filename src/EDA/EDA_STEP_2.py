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
