import pandas as pd


def get_clustering_folium(df, X_col, Y_col, label_column=None):
    import branca.colormap as cm
    import folium

    seoul_center = [37.5665, 126.9780]
    seoul_map = folium.Map(location=seoul_center, zoom_start=12)

    label_colors = {
        0: "#FFC0CB",  # pink
        1: "#0000FF",  # blue
        2: "#008000",  # green
        3: "#FFA500",  # orange
        4: "#800080",  # purple
        5: "#FF0000",  # red
        6: "#008080",  # skyblue
        7: "#000080",  # navy
        8: "#00FF00",  # lime
        9: "#A9A9A9",  # darkgray
        10: "#A52A2A",
        11: "#FFFF00",
        12: "#D3D3D3",  # Light Gray:
        # Dark Gray: #A9A9A9
        # Brown: #A52A2A
        # Yellow: #FFFF00
    }

    if label_column != None:
        folium_data = df[[X_col, Y_col, label_column]]
        for index, rows in folium_data.iterrows():

            X, Y, label = rows[X_col], rows[Y_col], rows[label_column]
            fill_color = label_colors.get(label, "#FF0000")  # 지정되지 않은 라벨은 red로 설정
            folium.Circle(
                location=[Y, X],
                color=fill_color,
                fill=True,
                fill_opacity=0.4,
            ).add_to(seoul_map)
    else:
        folium_data = df[[X_col, Y_col]]
        for index, rows in folium_data.iterrows():

            X, Y = rows[X_col], rows[Y_col]

            # fill_color = label_colors.get(label, '#FF0000')  # 지정되지 않은 라벨은 red로 설정
            folium.Circle(
                location=[Y, X],
                # color=fill_color,
                fill=True,
                fill_opacity=0.4,
            ).add_to(seoul_map)

    return seoul_map


# folium DF 전처리 함수
def bus_info(df, node_name):
    """_summary_

    Args:
        df : ex) b
        node_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    return df[df["노선번호"] == node_name][
        [
            "노선번호",
            "표준버스정류장ID",
            "역명",
            "6시승차총승객수",
            "6시하차총승객수",
            "7시승차총승객수",
            "7시하차총승객수",
            "8시승차총승객수",
            "8시하차총승객수",
            "9시승차총승객수",
            "9시하차총승객수",
            "6_9시_승차",
            "6_9시_하차",
            "X좌표",
            "Y좌표",
            "순번",
        ]
    ]


def bus_재차인원(
    total_df, node_num: str or dict, start_idx: int = None, end_idx: int = None
) -> int:
    """
    _summary_

    parameters:
        total_df : 전체 정류장의 데이터를 넣는다
        node_num : 노선이름을 쓴다. 또는 재차인원 계산을 원하는 df를 넣는다
        start_idx : 해당 노선에서 원하는 정류장 시작점 인덱스를 적는다
        end_idx : 해당 노선에서 원하는 정류장 종점 인덱스를 적는다
    Returns:
        _type_: _description_
    """
    if isinstance(node_num, str):
        if node_num in bus_info(total_df, node_num)["노선번호"].tolist():
            if start_idx or end_idx:
                bus_selected = bus_info(total_df, node_num)[
                    start_idx:end_idx
                ].reset_index()
            else:
                bus_selected = bus_info(total_df, node_num).reset_index()

            bus_selected["6시승하차_sub"] = (
                bus_selected.loc[:, "6시승차총승객수"] - bus_selected.loc[:, "6시하차총승객수"]
            )
            재차인원 = []
            for i in range(len(bus_selected["6시승하차_sub"])):
                if i == 0:
                    재차인원.append(bus_selected.loc[0, "6시승하차_sub"])
                else:
                    재차인원.append(bus_selected.loc[i, "6시승하차_sub"] + 재차인원[i - 1])
            bus_selected["6시재차인원"] = 재차인원

            bus_selected["7시승하차_sub"] = (
                bus_selected.loc[:, "7시승차총승객수"] - bus_selected.loc[:, "7시하차총승객수"]
            )
            재차인원 = []
            for i in range(len(bus_selected["7시승하차_sub"])):
                if i == 0:
                    재차인원.append(bus_selected.loc[0, "7시승하차_sub"])
                else:
                    재차인원.append(bus_selected.loc[i, "7시승하차_sub"] + 재차인원[i - 1])
            bus_selected["7시재차인원"] = 재차인원

            bus_selected["8시승하차_sub"] = (
                bus_selected.loc[:, "8시승차총승객수"] - bus_selected.loc[:, "8시하차총승객수"]
            )
            재차인원 = []
            for i in range(len(bus_selected["8시승하차_sub"])):
                if i == 0:
                    재차인원.append(bus_selected.loc[0, "8시승하차_sub"])
                else:
                    재차인원.append(bus_selected.loc[i, "8시승하차_sub"] + 재차인원[i - 1])
            bus_selected["8시재차인원"] = 재차인원

            bus_selected["9시승하차_sub"] = (
                bus_selected.loc[:, "9시승차총승객수"] - bus_selected.loc[:, "9시하차총승객수"]
            )
            재차인원 = []
            for i in range(len(bus_selected["9시승하차_sub"])):
                if i == 0:
                    재차인원.append(bus_selected.loc[0, "9시승하차_sub"])
                else:
                    재차인원.append(bus_selected.loc[i, "9시승하차_sub"] + 재차인원[i - 1])
            bus_selected["9시재차인원"] = 재차인원
            # 6-9시 재차인원 합
            bus_selected["6_9시재차인원합"] = (
                bus_selected["6시재차인원"]
                + bus_selected["7시재차인원"]
                + bus_selected["8시재차인원"]
                + bus_selected["9시재차인원"]
            )

            return bus_selected[
                [
                    "노선번호",
                    "표준버스정류장ID",
                    "순번",
                    "역명",
                    "6시승차총승객수",
                    "6시하차총승객수",
                    "6시승하차_sub",
                    "7시승차총승객수",
                    "7시하차총승객수",
                    "7시승하차_sub",
                    "8시승차총승객수",
                    "8시하차총승객수",
                    "8시승하차_sub",
                    "9시승차총승객수",
                    "9시하차총승객수",
                    "9시승하차_sub",
                    "6시재차인원",
                    "7시재차인원",
                    "8시재차인원",
                    "9시재차인원",
                    "X좌표",
                    "Y좌표",
                ]
            ]

        else:
            bus_selected = eval(node_num).reset_index()
            bus_selected["6_9시승하차_sub"] = (
                bus_selected.loc[:, "6_9시_승차"] - bus_selected.loc[:, "6_9시_하차"]
            )

            재차인원 = []
            for i in range(len(bus_selected["6_9시_승차"])):
                if i == 0:
                    재차인원.append(bus_selected.loc[0, "6_9시승하차_sub"])
                else:
                    재차인원.append(bus_selected.loc[i, "6_9시승하차_sub"] + 재차인원[i - 1])
            bus_selected["6_9_재차인원_합"] = 재차인원

            return bus_selected[
                [
                    "노선번호",
                    "표준버스정류장ID",
                    "역명",
                    "6_9시_승차",
                    "6_9시_하차",
                    "6_9시승하차_sub",
                    "6_9_재차인원_합",
                    "X좌표",
                    "Y좌표",
                    "순번",
                ]
            ]

    elif isinstance(node_num, pd.DataFrame):
        bus_selected = node_num

        bus_selected["승하차_sub"] = (
            bus_selected.loc[:, "승차총승객수"] - bus_selected.loc[:, "하차총승객수"]
        )
        재차인원 = []
        for i in range(len(bus_selected["승하차_sub"])):
            if i == 0:
                재차인원.append(bus_selected.loc[0, "승하차_sub"])
            else:
                재차인원.append(bus_selected.loc[i, "승하차_sub"] + 재차인원[i - 1])
        bus_selected["재차인원"] = 재차인원

        return bus_selected


# 해당 정류장을 지나는 노선 정보 추출(좌표 포함)
def station_info(df, station_name):
    """_summary_

    Args:
        df (_type_): ex)b
        station_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    selected_station = df[df["역명"].str.contains(station_name)][
        [
            "노선번호",
            "노선명_x",
            "순번",
            "표준버스정류장ID",
            "버스정류장ARS번호",
            "역명",
            "X좌표",
            "Y좌표",
            "6_9시_승차",
            "6_9시_하차",
        ]
    ]
    return selected_station


# 해당 노선에 해당하는 정류장들 데이터 프레임
def select_node_num_info(df, node_name):
    """_summary_

    Args:
        df (_type_): ex)b
        node_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    return df[df["노선번호"] == node_name][
        [
            "노선번호",
            "노선명_x",
            "표준버스정류장ID",
            "버스정류장ARS번호",
            "역명",
            "6_9시_승차",
            "6_9시_하차",
            "노드ID",
            "X좌표",
            "Y좌표",
            "순번",
        ]
    ].reset_index(drop=True)


# 폴리움을 위해 노선별로 데이터 프레임 만들기
def node_group(station_name):
    if isinstance(station_name, str):
        selected_station = station_info(station_name)

        # 해당 정류장을 지나다니는 노선번호 리스트
        node_num_list = selected_station["노선번호"].unique().tolist()

        # 해당 정류장을 지나다니는 노선 버스 데이터프레임
        df = pd.DataFrame()
        for i in node_num_list:
            df = pd.concat([df, select_node_num_info(i)], axis=0)

        # 동일한 노선끼리 그룹화하기
        group_node = df.groupby("노선번호")
        node_name = []
        for id in node_num_list:
            globals()["node_{}".format(id)] = group_node.get_group(id)
            node_name.append("node_{}".format(id))

        return node_name, df

    elif isinstance(station_name, pd.DataFrame):
        node_num_list = station_name["노선번호"].unique().tolist()

        group_node = station_name.groupby("노선번호")
        node_name = []
        for id in node_num_list:
            globals()["node_{}".format(id)] = group_node.get_group(id)
            node_name.append("node_{}".format(id))

        return node_name


def bus_node(df, node_name):
    """_summary_

    Args:
        df (_type_): ex)b
        node_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    return df[df["노선번호"] == node_name][
        ["X좌표", "Y좌표", "노선명_y", "순번" "6_9시_승차", "6_9시_하차", "NODE_ID"]
    ]


# 다람쥐 버스 df 만들기
def squirrel_bus():
    route_info_8331 = select_node_num_info("8331")

    route_info_8221 = select_node_num_info("8221")

    route_info_8441 = select_node_num_info("8441")

    route_info_8552 = select_node_num_info("8552")

    route_info_8551 = select_node_num_info("8551")

    route_info_8771 = select_node_num_info("8771")

    route_info_8761_1 = select_node_num_info("6713").loc[21:23, :]

    route_info_8761_2 = select_node_num_info("6713").loc[38:40, :]
    # 다람쥐 버스 정류장 df
    mou_bus = pd.concat(
        [
            route_info_8331,
            route_info_8221,
            route_info_8441,
            route_info_8552,
            route_info_8551,
            route_info_8771,
            route_info_8761_1,
            route_info_8761_2,
        ],
        axis=0,
    )

    return mou_bus


# 다람쥐 버스 정류장 노선 및 승하차 인원 표시
def folium_bus(
    히트맵_df: dict = None,
    히트맵_컬럼: str = None,
    bus_df: dict = None,
    유사도_군집_df: dict = None,
    tile_type: str = "gray",
):
    """_summary_

    Args:
        bus_df (dict, optional): _description_. Defaults to None.
        tile_type (str, optional): Defaults to "gray". ex) "midnight"

    Returns:
        _type_: _description_
    """
    import branca.colormap as cm
    import folium
    from folium import CircleMarker
    from folium.plugins import HeatMap
    from collections import defaultdict

    vworld_key = "BD606474-76A5-3146-848E-21906146E125"

    seoul_center = [37.5665, 126.9780]
    seoul_map = folium.Map(location=seoul_center, zoom_start=12)

    # 배경지도 타일 설정하기
    layer = tile_type
    # "midnight"
    # "gray"
    tileType = "png"
    tiles = f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/{layer}/{{z}}/{{y}}/{{x}}.{tileType}"
    attr = "Vworld"

    folium.TileLayer(tiles=tiles, attr=attr, overlay=True, control=True).add_to(
        seoul_map
    )

    label_colors = {
        0: "#FFC0CB",  # pink
        1: "#0000FF",  # blue
        2: "#008000",  # green
        3: "#FFA500",  # orange
        4: "#800080",  # purple
        5: "#FF0000",  # red
        6: "#008080",  # Teal
        7: "#000080",  # navy
        8: "#00FF00",  # lime
        9: "#A9A9A9",  # darkgray
        10: "#A52A2A",  # Brown
        11: "#FFFF00",  # Yellow
    }

    color_map = cm.LinearColormap(
        colors=[
            "pink",
            "blue",
            "green",
            "orange",
            "purple",
            "red",
            "Teal",
            "navy",
            "Lime",
            "DarkGray",
            "Brown",
            "Yellow",
        ],
        vmin=0.6,
        vmax=0.9,
        caption="Class",
    ).to_step(n=11)

    # 노선 그리기
    if bus_df is None:
        pass
    else:
        node = node_group(bus_df)
        for i in node:
            folium.PolyLine(
                locations=eval(i)[["Y좌표", "X좌표"]].values.tolist(),
                opacity=0.6,
                color="yellow",
            ).add_to(seoul_map)

        # 승하차 인원 표시
        for index, rows in bus_df.iterrows():
            X, Y, on, off = (
                rows["X좌표"],
                rows["Y좌표"],
                rows["RIDE_SUM_6_10"],
                rows["ALIGHT_SUM_6_10"],
            )
            # fill_color = label_colors.get(label, '#FF0000')  # 지정되지 않은 라벨은 red로 설정
            CircleMarker(
                location=[Y, X],
                radius=on / 25000,
                tooltip=on,
                popup={"승차": on, "하차": off},
                color="greenyellow",
                fill_opacity=0.2,
            ).add_to(seoul_map)
            CircleMarker(
                location=[Y, X],
                radius=off / 25000,
                tooltip=off,
                color="orangered",
                fill_opacity=0.2,
            ).add_to(seoul_map)

    # 유사도 정류장 1315개 표시
    if 유사도_군집_df is None:
        pass
    else:
        for index, rows in 유사도_군집_df.iterrows():
            X, Y, label = (rows["X좌표"], rows["Y좌표"], rows["gmm_cluster"])
            fill_color = label_colors.get(label, "#FF0000")  # 지정되지 않은 라벨은 red로 설정
            CircleMarker(
                location=[Y, X],
                radius=0.1,
                # radius= off /70000,
                # tooltip= off ,
                fill=True,
                color=fill_color,
                fill_opacity=0.4,
            ).add_to(seoul_map)

    # Add heatmap
    steps = 15
    gradient_map = defaultdict(dict)
    for i in range(steps):
        gradient_map[1 / steps * i] = color_map.rgb_hex_str(1 / steps * i)

    if 히트맵_df is None:
        pass

    else:
        HeatMap(
            data=zip(히트맵_df["Y좌표"], 히트맵_df["X좌표"], 히트맵_df[히트맵_컬럼]),
            radius=13,
            gradient=gradient_map,
        ).add_to(seoul_map)

    seoul_map.save("다람쥐버스_노선_승하차_히트맵.html")
    return seoul_map
