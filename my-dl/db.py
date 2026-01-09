import pymysql

def get_conn():
    return pymysql.connect(
        host='43.201.38.213',
        user='septures',
        password='1234',
        db='my_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    