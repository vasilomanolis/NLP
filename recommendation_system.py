import psycopg2

# testing dumb credentials
path = "lennon.cn9xzlhoi7ey.us-east-2.rds.amazonaws.com"
database = "lennon"
username = "vasilis"
password = ""


conn = psycopg2.connect(host=path,database=database, user=username, password=password)


print ("Opened database successfully")

