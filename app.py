import bcrypt
from flask import Flask, render_template, redirect, url_for,session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql.functions import current_user
from werkzeug.security import check_password_hash, generate_password_hash

from forms import RegistrationForm, LoginForm
from model import User, db
import os

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
db.init_app(app)

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html', user=current_user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        nickname = form.nickname.data
        email = form.email.data
        password = form.password.data
        confirm_password = form.confirm_password.data

        if password == confirm_password:
            hashed_password = generate_password_hash(password)
            new_user = User(nickname=nickname, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('index'))
        else:
            return "Passwords do not match. Please try again."

    return render_template('register.html', form=form,user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            return redirect(url_for('index'))
        else:
            return "Неверные email или пароль. Попробуйте снова."

    return render_template('login.html', form=form,user=current_user)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # with app.app_context():
    #     db.create_all()
    app.run(debug=True)