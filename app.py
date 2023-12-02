import tempfile
import cv2
import numpy as np
from flask import Flask, render_template, redirect, url_for, session, flash, send_from_directory, request, abort
from flask_uploads import configure_uploads
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required
from werkzeug.utils import secure_filename
from flask_login import current_user
from forms import RegistrationForm, LoginForm, photos, FileUploadForm
from model import User, db, Image
from tensorflow.keras.models import load_model
import os

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['UPLOADED_PHOTOS_DEST'] = '/home/rydyar/cours/cur/uploads'
configure_uploads(app, photos)

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


@app.route('/user_images')
@login_required
def user_images():
    user_images = Image.query.filter_by(user_id=current_user.id).all()
    return render_template('image.html', user_images=user_images)


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
    logout_user()
    flash('Вы успешно вышли из аккаунта!', 'success')
    return redirect(url_for('index'))


@app.route('/download_image/<filename>')
@login_required
def download_image(filename):
    image = Image.query.filter_by(filename=filename, user_id=current_user.id).first()
    if image is None:
        abort(404)

    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename, as_attachment=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


@app.route('/upload_image', methods=['GET', 'POST'])
@login_required
def upload_image():
    form = FileUploadForm()
    if form.validate_on_submit():
        if 'file' in request.files:
            image = request.files['file']
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))

            new_image = Image(user_id=current_user.id, filename=filename)
            db.session.add(new_image)
            db.session.commit()

    return render_template('upload_image.html', form=form)


@app.route('/improve_image/<filename>', methods=['GET', 'POST'])
@login_required
def improve_image(filename):
    model = load_model('final_model.h5')

    original_image = Image.query.filter_by(filename=filename, user_id=current_user.id).first()

    img = cv2.imread(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], original_image.filename), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    SIZE = 256
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0

    predicted = model.predict(img.reshape(1, SIZE, SIZE, 3))
    predicted = np.clip(predicted, 0.0, 1.0).reshape(SIZE, SIZE, 3)
    predicted = (predicted * 255).astype('uint8')

    temp_filename = tempfile.mktemp(suffix='.png')
    cv2.imwrite(temp_filename, predicted)

    return send_from_directory(os.path.dirname(temp_filename), os.path.basename(temp_filename), as_attachment=True)


if __name__ == '__main__':
    # with app.app_context():
    #     db.create_all()
    app.run(debug=True)
