import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
import folium
import matplotlib.cm as cm
import math 
import glob
import os
import sys
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글깨짐 방지
warnings.filterwarnings('ignore')

current_path = os.path.abspath(__file__)
while os.path.split(current_path)[1] != 'src': 
    current_path = os.path.dirname(current_path)
csv_path = os.path.join(current_path, 'Data','csv')

sys.path.append(current_path)

from Data.preprocessing import *

df_infra = get_final_infra_df()
df_bus_info = pd.read_csv(os.path.join(csv_path, 'bus_route_info.csv'))
tb_infra_population = pd.read_csv(os.path.join(csv_path, 'tb_infra_population.csv'))
# df_bus_info = pd.read_csv(rf'{csv_path}/bus_route_info.csv')
# tb_infra_population = pd.read_csv(rf'{csv_path}/tb_infra_population.csv')



# 다람쥐 버스 리스트, 기+종점 리스트
daram_bus_list = ['8771', '8761', '8552', '8441', '8551', '8221', '8331']
start_station = ['111000128','113000113','120000156','120000109','105000127','122000305','123000209']
end_station =   ['111000291','118000048','119000024','120000018','105000072','122000302','123000043']
st_end_station= start_station + end_station

# 데이터 형식 변환 함수
def preprocessing_cluster(df):
    num_col = ['academy_cnt', 'kindergarten_cnt', 'mart_cnt', 'restaurant_cnt',
              'school_cnt', 'university_cnt', 'subway_cnt', 'tour_cnt', 'cafe_cnt',
              'hospital_cnt', 'culture_cnt', 'univ_hospital_cnt', 'public_office_cnt',
              'employee_cnt', 'alone_ratio', 'emp_corp_ratio', 'population_15to64']

    str_col_A = [i for i in df.columns if i not in num_col]
    df[str_col_A] = df[str_col_A].astype('str')

    if 'RIDE_SUM_6_10' in df.columns and 'ALIGHT_SUM_6_10' in df.columns:
        df = df.drop(['RIDE_SUM_6_10', 'ALIGHT_SUM_6_10'], axis=1)
    
    return df

df_bus_info = preprocessing_cluster(df_bus_info)
df_infra = preprocessing_cluster(df_infra)


def get_daram_95station_df():
    df_daram = df_bus_info[df_bus_info['노선명'].isin(daram_bus_list)]  
    df_merged = pd.merge(df_daram, df_infra,left_on ='NODE_ID', right_on = 'NODE_ID', how = 'left'  )
    return df_merged

def get_daram_14_station_df():
    df_daram = get_daram_95station_df()

    # 기+종점 데이터만 
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
def pca_explained_variance_ration(df) :
    pca = PCA(n_components=df.shape[1])
    pca_transformed = pca.fit_transform(df)
    pca_colum = ['pca_'+ str(i+1)for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(data = pca_transformed,
                        columns = pca_colum)
        
    df_pca_result = pd.DataFrame({'설명가능한 분산 비율(고윳값)': pca.explained_variance_,
                                '기여율': pca.explained_variance_ratio_},
                                index = pca_colum)
    df_pca_result['누적 기여율'] = df_pca_result['기여율'].cumsum()
    
    return df_pca_result

# PCA
def func_pca(df, n_comp):
    pca = PCA(n_components = n_comp)
    pca_transformed = pca.fit_transform(df)
    
    pca_lst = []
    
    for i in range(1, n_comp+1):
        word = 'pca_'
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


# KMeans
def clustering_kmeans(df, cluster_num, init, max_iter, random_state):
    kmeans = KMeans(n_clusters=cluster_num, init=init, max_iter=max_iter, random_state=random_state)
    y_pred = kmeans.fit_predict(df)
    new_df = df.copy()
    new_df['kmeans_label'] = y_pred

    return new_df


# GMM
def bic_aic(df,min_components, max_components):
    gms_per_k = [GaussianMixture(n_components=k, n_init=20, random_state=42).fit(df)
                for k in range(min_components, max_components)]
    # atrribute_error가 뜨면 범위를 1을 제외하고 입력할 것

    bics = [model.bic(df) for model in gms_per_k]
    aics = [model.aic(df) for model in gms_per_k]

    plt.figure(figsize = (8,3))
    plt.plot(range(min_components,max_components),bics, "bo-", label="BIC")
    plt.plot(range(min_components,max_components),aics, "go-", label="AIC")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("information Criterion", fontsize=14)
    plt.axis([2, 20.5, np.min(aics)-50, np.max(aics)+50])

    plt.legend()

    # 그림 저장
    # save_fit("aic_bic_vs_k_plot")
    plt.show()



def clustering_gmm(df, cluster_num, random_state):
    gmm = GaussianMixture(n_components= cluster_num, random_state=random_state).fit(df)
    y_pred = gmm.predict(df)

    new_df = df.copy()
    new_df['gmm_label'] = y_pred

    return new_df


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
        10: '#A52A2A',
        11: '#FFFF00',
        12: '#D3D3D3', #Light Gray: 
    # Dark Gray: #A9A9A9
    # Brown: #A52A2A
    # Yellow: #FFFF00
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



# def main_kmeans():
#     # 특정 경로 내의 모든 CSV 파일 가져오기
    
#     df_daram_14 = get_daram_14_station_df()
#     df_bus_not_daram = get_not_daram_station()

#     # numerical data만 추출
#     df_daram_14_num = df_daram_14.select_dtypes(include=['int','float'])
#     df_bus_not_daram_num = df_bus_not_daram.select_dtypes(include=['int','float'])

#     df_similar = get_cosine_similarity(df_daram_14_num, df_bus_not_daram_num, 100)
#     get_clustering_folium(df_similar, 'X좌표','Y좌표')
    

#     # kmeans
#     df_similar_num = df_similar.select_dtypes(include=['int', 'float'])
#     df_similar_num = df_similar_num.reset_index(drop=True)

#     df_scaled = scaler(df_similar_num, 'standard')

#     # visualize
#     elbow_method(2, 15, df_scaled)
#     visualize_silhouette([3,4,5,6,7,8,9,10,11,12,13,14,15],df_scaled)
#     calinski_harabasz(2,15, df_scaled)

#     cluster_num = 8
#     init_method = 'k-means++'
#     max_iter = 300
#     random_state = 0

#     df_kmeans = clustering_kmeans(df_scaled, cluster_num, init_method, max_iter, random_state)
#     kmeans_labels = df_kmeans['kmeans_label']

#     df_final = df_similar.copy()
#     df_final['kmeans_label'] = kmeans_labels.values

#     df_final

# if __name__ == '__main__':
#     main()
