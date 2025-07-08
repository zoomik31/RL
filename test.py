import psycopg2
connection = psycopg2.connect(user="postgres",
                                  password="admin",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="map")

cursor = connection.cursor()
# Распечатать сведения о PostgreSQL
cursor.execute("SELECT map, x, y from map_1")
record = cursor.fetchall()
print("Результат", record)