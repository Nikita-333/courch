from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
class User(db.Model):
    tablename = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    nickname = db.Column(db.String(80), unique=True, nullable=False)
