
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import numpy as np
from eda_module import *
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import folium
import matplotlib.cm as cm
import math 


plt.rcParams['font.family'] = 'Malgun Gothic' # 한글깨짐 방지
warnings.filterwarnings('ignore')

df_infra = pd.read_csv('./data/final_tb_infra_population.csv')
df_bus_info = pd.read_csv('./data/bus_route_info.csv')

# 불러온 csv의 type 전처리
df_bus_info['ROUTE_ID'] = df_bus_info['ROUTE_ID'].astype('str')
df_bus_info['순번'] = df_bus_info['순번'].astype('str')
df_bus_info['NODE_ID'] = df_bus_info['NODE_ID'].astype('str')
df_bus_info['ARS_ID'] = df_bus_info['ARS_ID'].astype('str')
df_bus_info['X좌표'] = df_bus_info['X좌표'].astype('str')
df_bus_info['Y좌표'] = df_bus_info['Y좌표'].astype('str')
    
df_infra['NODE_ID'] = df_infra['NODE_ID'].astype('str')
df_infra['X좌표'] = df_infra['X좌표'].astype('str')
df_infra['Y좌표'] = df_infra['Y좌표'].astype('str')
df_infra['법정동코드'] = df_infra['법정동코드'].astype('str')


def get_daram_95station_df():
    
    daram_bus_list = ['8771', '8761', '8552', '8441', '8551', '8221', '8331']
    df_daram = df_bus_info[df_bus_info['노선명'].isin(daram_bus_list)]  

    df_merged = pd.merge(df_daram, df_infra,left_on ='NODE_ID', right_on = 'NODE_ID', how = 'left'  )

    return df_merged

def get_daram_14_station_df():
    df_daram = get_daram_95station_df()

    # 다람쥐 버스 데이터 분할 
    # 시작점과 끝점 데이터만 
    start_station = ['111000128','113000113','120000156','120000109','105000127','122000305','123000209']
    end_station =   ['111000291','118000048','119000024','120000018','105000072','122000302','123000043']

    st_end_station= start_station + end_station

    df_daram_final = df_daram[df_daram['NODE_ID'].isin(st_end_station)]

    return df_daram_final



# 서울 전체 버스정류장 중에서 다람쥐버스 정류장이 아닌 것들만 추출
def get_not_daram_station():
    daram_list = get_daram_95station_df()['NODE_ID'].tolist()
    df_not_daram = df_infra[~df_infra['NODE_ID'].isin(daram_list)]

    return df_not_daram
    

# 코사인 유사도를 바탕으로 유사한 데이터 뽑기
def get_cosine_similarity(df_A, df_B, data_num):
    '''df_A: dataframe which is a criteria for calculating cosine similarity'''

    # 코사인 유사도 
    def cosine_similarity_matrix(X, Y): # X, Y는 array
        cosine_sim = 1-cdist(X,Y, metric='cosine') # 코사인 유사도 행렬
        return cosine_sim

    X = df_A.values
    Y = df_B.values

    cos_matrix = cosine_similarity_matrix(X,Y)
    sorted_cos_matrix = np.argsort(cos_matrix, axis=1)[:,::-1]

    # array 형태
    top_similar_cos = sorted_cos_matrix[:,:data_num]

    # array형태로 나온 유사한 데이터들을 flatten화 하여 1차원으로 변경

    df_bus_not_daram = get_not_daram_station()
    similar_data = df_bus_not_daram.iloc[top_similar_cos.flatten()]

    return similar_data

# scaler
def scaler(df,scaler):
    """_summary_

    Args:
        df (dataframe): insert pandas dataframe.
        scaler (_type_): choose scaler among standard, minmax and robust.

    Returns:
        _type_: dataframe
    """
    if scaler == 'standard':
        scaled_data = StandardScaler().fit_transform(df)
        df_scaled = pd.DataFrame(scaled_data, columns = df.columns)
    elif scaler == 'minmax':
        scaled_data = MinMaxScaler().fit_transform(df)
        df_scaled = pd.DataFrame(scaled_data, columns = df.columns)
    elif scaler == 'robust':
        scaled_data = RobustScaler().fit_transform(df)
        df_scaled = pd.DataFrame(scaled_data, columns = df.columns)
    else:
        print('올바른 scaler를 입력해주세요.')
    return df_scaled



# PCA 설명력 
def pca_explained_variance_ration(df,min_n,max_n) :
    for num in range(min_n, max_n):

        pca = PCA(n_components=num)
        pca_transformed = pca.fit_transform(df)

        print(num,'차원 분산 설명력 : ',sum(pca.explained_variance_ratio_))


# PCA
def func_pca(df, n_comp):
    pca = PCA(n_components = n_comp)
    pca_transformed = pca.fit_transform(df)
    
    pca_lst = []
    word = 'pca_'
    for i in range(1, n_comp+1):
        word += str(i)
        pca_lst.append(word)

    df_pca = pd.DataFrame(columns=pca_lst, data=pca_transformed)
    print('분산 설명력 : ',sum(pca.explained_variance_ratio_))

    return df_pca


# 최적 cluster 갯수 찾기
# elbow_method 함수
def elbow_method(range_min, range_max, df):
    sse = []
    for i in range(range_min,range_max):
        km = KMeans(n_clusters=i, init='k-means++',random_state=0)
        km.fit(df)
        sse.append(km.inertia_)

    plt.plot(range(range_min,range_max), sse, marker='o')
    plt.xlabel('nums of cluster')
    plt.ylabel('SSE')
    plt.show()


### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성
def visualize_silhouette(cluster_lists, X_features): 

    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")


def calinski_harabasz(min_cluster, max_cluster, df):    
    for num in range(min_cluster, max_cluster):
        km = KMeans(n_clusters = num, max_iter=500, random_state=0)
        km.fit_predict(df)
        print('calinski-score',num,'개 군집:', calinski_harabasz_score(df, km.labels_))


def clustering_kmeans(df, cluster_num, init, max_iter, random_state):
    kmeans = KMeans(n_clusters=cluster_num, init=init, max_iter=max_iter, random_state=random_state)
    y_pred = kmeans.fit_predict(df)
    df['kmeans_label'] = y_pred

    return df



def get_clustering_folium(df, X_col, Y_col, label_column=None):
    import branca.colormap as cm
    import folium

    seoul_center = [37.5665, 126.9780]
    seoul_map = folium.Map(location=seoul_center, zoom_start=12)

    label_colors = {
        0: '#FFC0CB',   # pink
        1: '#0000FF',   # blue
        2: '#008000',   # green
        3: '#FFA500',   # orange
        4: '#800080',   # purple
        5: '#FF0000',   # red
        6: '#008080',   # skyblue
        7: '#000080',    # navy
        8: '#00FF00',    # lime
        9: '#A9A9A9',    # darkgray
        10: '#A52A2A',   # Brown
        11: '#FFFF00',   # Yellow
        12: '#D3D3D3',   #Light Gray: 
    }


     
    if label_column != None:
        folium_data = df[[X_col, Y_col, label_column]]
        for index, rows in folium_data.iterrows():
            
            X, Y, label = rows[X_col], rows[Y_col], rows[label_column]
            fill_color = label_colors.get(label, '#FF0000')  # 지정되지 않은 라벨은 red로 설정
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