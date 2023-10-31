from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from flask_uploads import UploadSet, IMAGES
from model import Image
from flask_wtf.file import FileField, FileRequired, FileAllowed

photos = UploadSet('photos',IMAGES)
class RegistrationForm(FlaskForm):
    nickname = StringField('Nickname', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class FileUploadForm(FlaskForm):

    file = FileField(
        validators=[
            FileRequired('File field should not be empty'),
            FileAllowed(photos,['jpg','png'])
        ]
    )
    submit = SubmitField('Upload')