# MySQL DB 실행
def action_mysql():
    import os
    import csv
    import mysql.connector

    mydb = mysql.connector.connect(
        host="15.152.232.36", user="multi", password="Campus123!"
    )
    mycursor = mydb.cursor()
    mycursor.execute("CREATE DATABASE IF NOT EXISTS MT_Project_DB")
    mycursor.execute("USE MT_Project_DB")


# mysql에서 데이터 호출
def call_dataframe(table):
    import pandas as pd
    from sqlalchemy import create_engine

    db_connection_str = "mysql+pymysql://multi:Campus123!@15.152.232.36/MT_Project_DB"
    db_connection = create_engine(db_connection_str)

    pop_table = pd.read_sql("SELECT * FROM {}".format(table), con=db_connection)
    print("컬럼명 : ", list(pop_table.columns))

    # 데이터프레임
    return pop_table


# 엑셀 데이터 mysql에 적재 , 데이터 담긴 부분만 수정 필요
def load_mysql_excel(databasename, tablename, filename):
    import pandas as pd
    import pymysql
    import glob
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import os

    # MySQL 설정
    username = "multi"
    password = "Campus123!"
    hostname = "15.152.232.36"
    database_name = databasename
    desired_table_name = tablename
    current_working_directory = os.getcwd()

    cnx = pymysql.connect(
        user=username, password=password, host=hostname, charset="utf8"
    )
    cursor = cnx.cursor()

    # 파일 불러오기
    files = glob.glob(current_working_directory + "/data/" + filename)
    df_list = [pd.read_excel(file) for file in files]
    df = pd.concat(df_list, ignore_index=True)
    engine = create_engine(
        "mysql+pymysql://{user}:{pw}@{host}/{db}".format(
            user=username, pw=password, db=database_name, host=hostname
        )
    )
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        df.to_sql(
            desired_table_name,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=1000,
        )
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
    cursor.close()
    cnx.close()


# csv파일 mysql에 적재 , 데이터 담긴 부분만 수정 필요
def load_mysql_csv(databasename, tablename, filename):
    import pandas as pd
    import pymysql
    import glob
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import os

    # MySQL 설정
    username = "multi"
    password = "Campus123!"
    hostname = "15.152.232.36"
    database_name = databasename
    desired_table_name = tablename
    current_working_directory = os.getcwd()

    cnx = pymysql.connect(
        user=username, password=password, host=hostname, charset="utf8"
    )
    cursor = cnx.cursor()

    # 파일 불러오기
    files = glob.glob(current_working_directory + "/data/" + filename)
    df_list = [pd.read_csv(file) for file in files]
    df = pd.concat(df_list, ignore_index=True)
    engine = create_engine(
        "mysql+pymysql://{user}:{pw}@{host}/{db}".format(
            user=username, pw=password, db=database_name, host=hostname
        )
    )
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        df.to_sql(
            desired_table_name,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=1000,
        )
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
    cursor.close()
    cnx.close()
