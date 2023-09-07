import pandas as pd
import os
import sys

current_path = os.path.abspath(__file__)
while os.path.split(current_path)[1] != "src":
    current_path = os.path.dirname(current_path)
csv_path = os.path.join(current_path, "Data", "csv")


# 법정동 별 인구 테이블
def get_population_data_by_bubjeong():
    # 행정-법정 테이블
    code_path = os.path.join(csv_path, "행정_법정코드_2021.xlsx")
    code = pd.read_excel(code_path, sheet_name="법정동_행정동코드")
    code = code.astype("str")

    ## 5자리 이하 코드는 '구','시'를 의미하는 코드 -> 필요없음
    code_mod = code[code["행정구역코드"].str.len() > 5]

    ## 신 행정코드 = 8자리. 따라서 기존 코드에 0을 붙임
    code_mod["adm_code"] = [
        code + "0" if len(code) < 8 else code for code in code_mod["행정구역코드"]
    ]

    ## 행정동 별 법정동 갯수는 얼마나 되는지?
    code_cnt = (
        code_mod.groupby("adm_code")
        .count()[["법정동코드"]]
        .rename(columns={"법정동코드": "법정동갯수"})
    )

    # 인구 테이블(통계청)
    df_pop = pd.read_csv(os.path.join(csv_path, "population_house_corp.csv"))
    # df_pop = pd.read_csv(rf'{csv_path}/population_house_corp.csv')
    str_col = ["adm_cd", "adm_nm"]
    df_pop[str_col] = df_pop[str_col].astype("str")
    df_pop = df_pop.drop_duplicates(subset=["adm_cd"])
    df_pop.rename(columns={"population_code_15to64": "population_15to64"}, inplace=True)

    df_info = code_mod[["행정동(행정기관명)", "adm_code", "법정동", "법정동코드"]]

    # 행정 + 인구 테이블 merge
    df_merged = pd.merge(
        df_pop, code_cnt, left_on="adm_cd", right_on=code_cnt.index, how="outer"
    )

    # 행정동 정보(인구수, 사업체 수 등)을 각 행정동의 법정동 갯수로 나눔
    cal_col = [
        "tot_family",
        "tot_ppltn",
        "corp_cnt",
        "employee_cnt",
        "population_15to64",
        "household_cnt_family",
        "household_cnt_alone",
    ]
    df_merged[cal_col] = df_merged.apply(
        lambda row: row[cal_col] / row["법정동갯수"], axis=1
    )
    df_merged = df_merged.drop("법정동갯수", axis=1)

    df_final = pd.merge(
        df_info, df_merged, left_on="adm_code", right_on="adm_cd", how="outer"
    )
    df_final = df_final.groupby(["법정동코드", "법정동"]).sum().reset_index()

    return df_final


def preprocessing_infra():
    # 인프라 테이블 완성하기
    """Extract.py에서 kakao_infra함수를 통해 뽑은 df를 넣으면 됨"""
    df = pd.read_csv(os.path.join(csv_path, "df_kakao_infra.csv"))
    # df = pd.read_csv(rf'{csv_path}/df_kakao_infra.csv')
    str_col = [
        "NODE_ID",
        "ARS_ID",
        "정류소명",
        "X좌표",
        "Y좌표",
        "법정동코드",
        "법정동_구",
        "법정동",
        "행정동코드",
        "행정동",
    ]

    df = df.apply(lambda row: row.astype("str") if row.name in str_col else row, axis=0)

    ## 대학교 infra 붙이기
    univ_path = os.path.join(csv_path, "university.csv")
    df_univ = pd.read_csv(univ_path)
    df_univ["dong_name"] = df_univ["dong_name"].apply(lambda x: x.strip())
    df = pd.merge(df, df_univ, left_on="법정동", right_on="dong_name", how="left")
    df = df.fillna(0)
    df = df.drop("dong_name", axis=1)

    ## 대학병원 infra 붙이기
    univ_hos_path = os.path.join(csv_path, "university_hospital.csv")
    df_univ_hos = pd.read_csv(univ_hos_path)
    df = pd.merge(df, df_univ_hos, left_on="법정동", right_on="dong_name", how="left")
    df = df.fillna(0)
    df = df.drop("dong_name", axis=1)

    # total_bus_time csv용량이 너무 커서 집계 후 df인 total_ride_alight.csv로 대체
    total_ride_alight = pd.read_csv(os.path.join(csv_path, "total_ride_alight.csv"))
    # total_ride_alight = pd.read_csv(rf'{csv_path}/total_ride_alight.csv')
    total_ride_alight["NODE_ID"] = total_ride_alight["NODE_ID"].astype("str")
    # 인프라 + 승/하차 merge
    df_final = pd.merge(
        df, total_ride_alight, left_on="NODE_ID", right_on="NODE_ID", how="left"
    )
    ride_mean = df_final["RIDE_SUM_6_10"].mean()
    alight_mean = df_final["ALIGHT_SUM_6_10"].mean()

    df_final["RIDE_SUM_6_10"] = df_final["RIDE_SUM_6_10"].fillna(ride_mean)
    df_final["ALIGHT_SUM_6_10"] = df_final["ALIGHT_SUM_6_10"].fillna(alight_mean)

    return df_final


def get_final_infra_df():
    df_population = get_population_data_by_bubjeong()
    df_infra = preprocessing_infra()
    df_final = pd.merge(df_infra, df_population, on=["법정동코드", "법정동"], how="left")
    df_final = df_final.dropna()

    col_int = [
        "academy_cnt",
        "kindergarten_cnt",
        "mart_cnt",
        "restaurant_cnt",
        "school_cnt",
        "subway_cnt",
        "tour_cnt",
        "cafe_cnt",
        "hospital_cnt",
        "culture_cnt",
        "public_office_cnt",
        "university_cnt",
        "univ_hospital_cnt",
        "RIDE_SUM_6_10",
        "ALIGHT_SUM_6_10",
        "employee_cnt",
        "population_15to64",
    ]
    df_final[col_int] = df_final[col_int].astype("int")

    # 새로운 feature 조합
    ## alone_ratio
    df_final["alone_ratio"] = (
        df_final["household_cnt_alone"] / df_final["tot_ppltn"]
    )  # 1인가구 수
    # emp_corp_ratio
    df_final["emp_corp_ratio"] = df_final["employee_cnt"] / df_final["corp_cnt"]

    # 필요없는 column 을 drop
    drop_col = [
        "ARS_ID",
        "tot_family",
        "행정동코드",
        "행정동",
        "household_cnt_alone",
        "tot_ppltn",
        "corp_cnt",
        "household_cnt_family",
    ]

    # df_final.rename(columns = {'법정동_x':'법정동'},inplace=True)
    df_final = df_final.drop(drop_col, axis=1)
    df_final = df_final[
        [
            "NODE_ID",
            "정류소명",
            "X좌표",
            "Y좌표",
            "법정동코드",
            "법정동_구",
            "법정동",
            "academy_cnt",
            "kindergarten_cnt",
            "mart_cnt",
            "restaurant_cnt",
            "school_cnt",
            "university_cnt",
            "subway_cnt",
            "tour_cnt",
            "cafe_cnt",
            "hospital_cnt",
            "culture_cnt",
            "univ_hospital_cnt",
            "public_office_cnt",
            "employee_cnt",
            "alone_ratio",
            "emp_corp_ratio",
            "population_15to64",
            "RIDE_SUM_6_10",
            "ALIGHT_SUM_6_10",
        ]
    ]

    return df_final


# mysql에서 데이터 호출
def call_csv(table):
    import pandas as pd
    from sqlalchemy import create_engine

    db_connection_str = "mysql+pymysql://root:1234@127.0.0.1/new_schema"
    db_connection = create_engine(db_connection_str).raw_connection()

    pop_table = pd.read_sql("SELECT * FROM {}".format(table), con=db_connection)

    # 데이터프레임
    return pop_table


def get_bus_station_list(bus_route_info, bus_num):
    bus_list = bus_route_info[bus_route_info["노선명"] == bus_num]["NODE_ID"].tolist()
    bus_list_str = [str(value) for value in bus_list]
    return bus_list_str


def get_daram_bus_station(bus_numbers_list):
    no_daram_bus_df = pd.DataFrame()
    bus_df = pd.DataFrame()

    for bus_numbers in bus_numbers_list:
        node_list = bus_stations[bus_numbers]
        for node in node_list:
            condition = bus_infra_pop_df["NODE_ID"] == node
            rows_1 = bus_infra_pop_df[condition]
            rows_1["노선번호"] = bus_numbers
            bus_df = pd.concat([bus_df, rows_1], axis=0)

            condition = bus_infra_pop_df["NODE_ID"] != node
            rows_2 = bus_infra_pop_df[condition]
            rows_2["노선번호"] = bus_numbers
            no_daram_bus_df = pd.concat([no_daram_bus_df, rows_2], axis=0)
    no_daram_bus_df = no_daram_bus_df.drop_duplicates(subset=["NODE_ID"])

    return bus_df, no_daram_bus_df
