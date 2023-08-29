
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from git import Repo
warnings.filterwarnings('ignore')

current_path = os.path.abspath(__file__)
while os.path.split(current_path)[1] != 'src': 
    current_path = os.path.dirname(current_path)
csv_path = os.path.join(current_path, 'Data\csv')

sys.path.append(current_path)

from Data.preprocessing import *
bus_route_info = pd.read_csv(rf'{csv_path}\bus_route_info.csv')
tb_infra_population = pd.read_csv(rf'{csv_path}\tb_infra_population.csv')
final_tb_infra_population = get_final_infra_df()



# barplot 위한 전처리
def df_preprocess(df):
    df['ride_sum'] = df['06시-07시 승차인원']+df['07시-08시 승차인원']+df['08시-09시 승차인원']+df['09시-10시 승차인원']
    df['alight_sum'] = df['06시-07시 하차인원'] + df['07시-08시 하차인원'] + df['08시-09시 하차인원'] + df['09시-10시 하차인원']
    df = df[df['시']=='서울']
    df_filtered= df.groupby(['호선명','지하철역','시','구','동'])[['ride_sum','alight_sum']].mean().reset_index()
    return df_filtered


def get_subway_barplot(df, x_col=None, y_col=None, orderby_col=None):
    df = df.sort_values(by=orderby_col, ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(y=y_col, x=x_col, data=df[:30])
    plt.title(orderby_col, fontsize=12)
    plt.xlabel(x_col, fontsize=8)       # Set x-axis label font size
    plt.ylabel(y_col, fontsize=8)       # Set y-axis label font size
    plt.xticks(fontsize=8)              # Set x-axis tick label font size
    plt.yticks(fontsize=8)  
    plt.show()
    return 

# 스케일러
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


# 다람쥐 버스 95개의 인프라 df
def get_daram_95station_df():
    df_infra = final_tb_infra_population
    df_bus_info = bus_route_info

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

    daram_bus_list = ['8771', '8761', '8552', '8441', '8551', '8221', '8331']
    df_daram = df_bus_info[df_bus_info['노선명'].isin(daram_bus_list)]  

    df_merged = pd.merge(df_daram, df_infra,left_on ='NODE_ID', right_on = 'NODE_ID', how = 'left'  )

    return df_merged

# 다람쥐 버스의 기&종점(14개)만의 인프라 df
def get_daram_14_station_df():
    df_daram = get_daram_95station_df()

    # 다람쥐 버스 데이터 분할 
    # 시작점과 끝점 데이터만 
    start_station = ['111000128','113000113','120000156','120000109','105000127','122000305','123000209']
    end_station =   ['111000291','118000048','119000024','120000018','105000072','122000302','123000043']

    st_end_station= start_station + end_station

    df_daram_final = df_daram[df_daram['NODE_ID'].isin(st_end_station)]

    return df_daram_final


# 다람쥐 버스 기&종점(14개) 인프라 데이터 +  standard scaling
def get_scaled_daram_df():
    
    df_daram_final = get_daram_14_station_df()

   # barplot을 위한 최종 dataframe
    plot_col = [ 'academy_cnt', 'kindergarten_cnt', 'mart_cnt', 'restaurant_cnt',
                    'school_cnt', 'university_cnt','subway_cnt', 'tour_cnt', 'cafe_cnt',
                    'hospital_cnt', 'culture_cnt', 'univ_hospital_cnt', 'public_office_cnt',
                    'employee_cnt', 'alone_ratio', 'emp_corp_ratio', 'population_15to64']
    df_daram_barplot = df_daram_final[plot_col]

    df_daram_scaled = scaler(df_daram_barplot, 'standard')

    노선 = df_daram_final['노선명'].tolist()
    순번 = df_daram_final['순번'].tolist()

    df_daram_scaled['노선명'] = 노선
    df_daram_scaled['순번'] = 순번

    df_daram_scaled.columns = ['academy', 'kindergarten', 'mart', 'restaurant',
       'school', 'university', 'subway','tour', 'cafe', 'hospital',
       'culture', 'univ_hospital', 'public_office', 'employee',
       'alone_ratio', 'emp_corp_ratio', 'population_15to64', '노선명','순번']
    
    df_daram_scaled = df_daram_scaled.set_index('노선명')
    df_daram_scaled = df_daram_scaled.sort_values(by='순번')

    return df_daram_scaled



def get_barplot_start_end(df):
    plt.rcParams['font.family'] = 'Malgun Gothic' # 한글깨짐 방지
    daram_bus_list = ['8771', '8761', '8552', '8441', '8551', '8221', '8331']

    for bus in daram_bus_list:
        df_start = df[df.index == bus].iloc[0, :-1]
        df_end = df[df.index == bus].iloc[1, :-1]

        # Create a figure with two subplots side by side (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

        # Plot the first bar chart on the first subplot (left)
        sns.barplot(x=df_start.index, y=df_start.values, ax=ax1, label='infra_count')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)  # Corrected line
        ax1.axhline(y=0, color='gray', linestyle='--')
        ax1.legend()
        ax1.set_title(f"{bus} 버스 시작 정류장의 infra",fontsize=20)

        # Plot the second bar chart on the second subplot (right)
        sns.barplot(x=df_end.index, y=df_end.values, ax=ax2, label='infra_count')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)  # Corrected line
        ax2.legend()
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_title(f"{bus} 버스 종점 정류장의 infra",fontsize=20)

        # Adjust layout for better spacing between subplots
        plt.tight_layout()

        # Show the combined plot with both bar charts
        plt.show();

# 상관관계 heatmap
def get_heatmap_data():
    df_corr = tb_infra_population
    df_check = df_corr[['academy_cnt','bank_cnt', 'kindergarten_cnt', 'mart_cnt', 'restaurant_cnt',
                                'school_cnt', 'university_cnt', 'subway_cnt', 'tour_cnt', 'cafe_cnt',
                                'hospital_cnt', 'culture_cnt', 'univ_hospital_cnt', 'public_office_cnt',
                                'tot_family', 'tot_ppltn', 'corp_cnt', 'employee_cnt',
                                'population_code_15to64', 'household_cnt_family', 'household_cnt_alone']]
    
    df_final = final_tb_infra_population
    
    df_check2 = df_final[['academy_cnt',
       'kindergarten_cnt', 'mart_cnt', 'restaurant_cnt', 'school_cnt',
       'university_cnt', 'subway_cnt', 'tour_cnt', 'cafe_cnt', 'hospital_cnt',
       'culture_cnt', 'univ_hospital_cnt', 'public_office_cnt', 'employee_cnt',
       'alone_ratio', 'emp_corp_ratio', 'population_15to64']]
    
    return df_check, df_check2

def get_heatmap(data):
    plt.rcParams['font.family'] = 'Malgun Gothic' # 한글깨짐 방지
    sns.set(rc={'figure.figsize':(10,10)})
    sns.heatmap(data.corr(),annot = True,linewidths=.5, annot_kws = {'size':7})


    plt.tight_layout()
    plt.show();



def get_barplot_daram_vs_all():
    plt.rcParams['font.family'] = 'Malgun Gothic' # 한글깨짐 방지
    df_daram = get_daram_95station_df()

    final_tb_infra_population['NODE_ID'] = final_tb_infra_population['NODE_ID'].astype('str')
    final_tb_infra_population['X좌표'] = final_tb_infra_population['X좌표'].astype('str')
    final_tb_infra_population['Y좌표'] = final_tb_infra_population['Y좌표'].astype('str')
    final_tb_infra_population['법정동코드'] = final_tb_infra_population['법정동코드'].astype('str')


    df_ride_top30 = df_daram.sort_values(by='RIDE_SUM_6_10', ascending=False)[:30]
    df_alight_top30 = df_daram.sort_values(by='ALIGHT_SUM_6_10',ascending=False)[:30]

    a = df_ride_top30.describe().loc['mean']
    b = df_alight_top30.describe().loc['mean']
    c = final_tb_infra_population.describe().loc['mean']

    RIDE = pd.DataFrame({'top30_mean':a, 'total_mean':c})
    ALIGHT = pd.DataFrame({'top30_mean':b, 'total_mean':c})

    lst_df = [RIDE, ALIGHT]
    lst_name = ['승차','하차']

    for df in zip(lst_df,lst_name):
   
        df_fin_infra = df[0][:13]
        df_employee_cnt = df[0].iloc[[13,16]]
        df_ratio = df[0][14:16]

        fig, axs = plt.subplots(1, 3, figsize=(20,5))



        df_fin_infra.plot.bar(rot=90, ax=axs[0]) 
        axs[0].set_xticklabels(df_fin_infra.index, fontsize=12)
        axs[0].legend(['다람쥐 버스','전체 버스'],fontsize=10)
        axs[0].set_title('다람쥐 버스 vs 전체 버스_'+df[1])

        df_employee_cnt.plot.bar(rot=0, ax = axs[1])  
        axs[1].set_xticklabels(df_employee_cnt.index, fontsize=12)
        axs[1].legend(['다람쥐 버스','전체 버스'],fontsize=10)
        axs[1].set_title('다람쥐 버스 vs 전체 버스_'+df[1])

        df_ratio.plot.bar(rot=0, ax = axs[2])  
        axs[2].set_xticklabels(df_ratio.index, fontsize=12)
        axs[2].legend(['다람쥐 버스','전체 버스'],fontsize=10)
        axs[2].set_title('다람쥐 버스 vs 전체 버스_'+df[1])

        plt.tight_layout()
        plt.show();