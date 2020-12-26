from flask import Flask, render_template, request

from core.forms import BookSearchForm
from core.utils import *


@app.route("/", methods=["GET", "POST"])
def index():
    search = BookSearchForm(request.form)
    if request.method == "POST":
        return search_results(search)

    return render_template("index.html", form=search)


@app.route("/recommendations")
def search_results(search):
    pass
