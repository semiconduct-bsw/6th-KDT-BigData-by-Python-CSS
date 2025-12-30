import pymysql

def get_conn():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        db='my_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
