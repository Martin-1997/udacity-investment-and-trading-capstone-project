from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Type: Product
def add(product):
    db.session.add(product)
    db.session.commit()

def reset():
    db.drop_all()
    db.create_all()

import sqlite3

conn = sqlite3.connect('database.db')# If the file/datebase does not exist, it gets created
# conn.execute("CREATE TABLE users (name TEXT, addr TEXT)") # This code should only run once per database
name = "Some"
addr = "Dude"
cursor = conn.cursor() # Where to insert in the database
# cursor.execute("INSERT INTO users (name, addr) VALUES (?,?)", (name, addr))
# conn.commit()

cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
print(rows)
# rows[0] # first entry
# rows[0][0] # first entry, first value

conn.close()
