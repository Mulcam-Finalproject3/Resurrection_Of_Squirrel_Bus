#유사도

    # 유클라디안 유사도

def euclidean_similar_data(df_A, df_B, num_similar):
    from scipy.spatial.distance import cdist
    import numpy as np
    distances = cdist(df_A.values, df_B.values, metric='euclidean')
    sorted_indices = np.argsort(distances)
    similar_indices = sorted_indices[:, :num_similar]
    similar_data = df_B.iloc[similar_indices.flatten()]
    return similar_data

    # 코사인 유사도


def cosine_similarity_matrix(X, Y):
    from scipy.spatial.distance import cdist
    import numpy as np
    # 코사인 유사도 행렬 계산하기
    cosine_sim = 1 - cdist(X, Y, metric='cosine')
    return cosine_sim

def cosine_similar_data(df_A, df_B, num_similar):
    import numpy as np
    # df_A와 df_B의 데이터를 배열로 변환
    X = df_A.values
    Y = df_B.values

    # 코사인 유사도 행렬 계산
    similarity_matrix = cosine_similarity_matrix(X, Y)

    # 유사도 행렬에서 유사한 인덱스 추출하기
    sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
    similar_indices = sorted_indices[:, :num_similar]

    # 유사한 데이터 추출
    similar_data = df_B.iloc[similar_indices.flatten()]

    return similar_data

# PCA 2차원 축소

def visualize_similar_data(title,similar_data):  # title은 string으로 기입하기 
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(similar_data.values)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.show()

# Folium으로 지도에 표시하는 함수 
import folium

def visualize_similar_data_on_map(html_name,similar_data,daram_df):
    # 서울 중심 좌표
    seoul_center = [37.5665, 126.9780]
    
    # Folium 지도 객체 생성
    map_seoul = folium.Map(location=seoul_center, zoom_start=12)

    # 유사한 데이터 위치 표시
    for _, row in similar_data.iterrows():
        latitude = row['Y좌표']  
        longitude = row['X좌표']  
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opaticy=0.6,
            popup=folium.Popup(row['정류소명'], max_width=300),
        ).add_to(map_seoul)
    
    # 기존 다람쥐 버스 노선 표시 
    for _, row in daram_df.iterrows():
        latitude = row['Y좌표']  
        longitude = row['X좌표']  
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opaticy=0.6,
            popup=folium.Popup(row['정류소명'], max_width=300),
        ).add_to(map_seoul)
        
 

    # Folium 지도 출력
    map_seoul.save(html_name)  # 결과를 HTML 파일로 저장

def visualize_similar_not_PCA(title,similar_data):  # title은 string으로 기입하기 
    import numpy as np
    import matplotlib as plt
    similar_data_np = np.array(similar_data)
    plt.scatter(similar_data_np[:, 0], similar_data_np[:, 1])
    plt.xlabel('RIDE')
    plt.ylabel('ALIGHT')
    plt.title(title)
    plt.show()


def PCA_bus_infra(bus_infra_pop_copy_df):
    from sklearn.preprocessing import StandardScaler
    pca_column_df = bus_infra_pop_copy_df.loc[:,'academy_cnt':'population_15to64']
    pca_column_df.head()
    pca_bus_df =StandardScaler().fit_transform(pca_column_df)
    pca = PCA(n_components=2)
    pca.fit(pca_bus_df)
    pca_transformed_df = pca.transform(pca_bus_df)
    return pca_transformed_df

def scatter(X,Y,hue,title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.scatterplot(x=X, y=Y, hue=hue)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.show()


