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
