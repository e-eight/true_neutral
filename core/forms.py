from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField


class BookSearchForm(FlaskForm):
    title = StringField(u"Book title")
    summary = TextAreaField(u"Book Summary")
    search = SubmitField(u"Get recommendations")
