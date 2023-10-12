import bcrypt
from flask import Flask, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql.functions import current_user
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import LoginManager, login_user, logout_user
from forms import RegistrationForm, LoginForm
from model import User, db
import os

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html', user=current_user)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        nickname = form.nickname.data

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Этот email уже зарегистрирован.', 'danger')
            return redirect(url_for('register'))

        new_user = User(email=email, password=generate_password_hash(password), nickname=nickname)
        db.session.add(new_user)
        db.session.commit()
        flash('Вы успешно зарегистрировались!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Вы успешно вошли в аккаунт!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Неверные учетные данные. Попробуйте снова.', 'danger')
    return render_template('login.html', form=form)


@app.route('/logout')
def logout():
    logout_user()  # Выход пользователя из системы
    flash('Вы успешно вышли из аккаунта!', 'success')
    return redirect(url_for('index'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)