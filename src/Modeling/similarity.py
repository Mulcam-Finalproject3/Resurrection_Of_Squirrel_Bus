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
