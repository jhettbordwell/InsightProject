# Tracing everything by following this website:
# https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/

# Forms information:
# https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-iii-web-forms

from flask import Flask, render_template, flash, redirect, url_for
from wtforms.fields.core import Label
from config.config import Config

# Importing my own module
from pythonForms.form import LetsGo, DiagnosisForm, ADHDForm, AnxietyForm, BipolarForm, DepressionForm, SchizophreniaForm

import matplotlib.pyplot as plt
import numpy as np
import os
# Importing machine learning section
# from import findCompatibility

app = Flask(__name__)
app.config.from_object(Config)

rainbow_folder = os.path.join('static', 'test')
app.config['UPLOAD_FOLDER'] = rainbow_folder

@app.route("/", methods=['GET','POST'])
def home():
    form = LetsGo()
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'action.png')

    if form.validate_on_submit():
        return redirect(url_for('diagnosisform'))
    return render_template("home.html", form=form, user_image=full_filename)

@app.route("/diagnosisform", methods=['GET','POST'])
def diagnosisform():
    form = DiagnosisForm()
    if form.validate_on_submit():
        content = form.diagnosis.data
        return furtherquestions(content)
    return render_template("diagnosisform.html", form=form)

def create_image():
    x = np.arange(10000).reshape((100,-1))
    plt.imshow(x)
    plt.savefig('static/test/test.png')

@app.route("/furtherquestions/", methods=['GET', 'POST'])
def furtherquestions(content, title=""):
    if content.strip() == 'ADHD':
        form = ADHDForm()
    elif content.strip() == 'Anxiety':
        form = AnxietyForm()
    elif content.strip() == 'Bipolar-Disorder':
        form = BipolarForm()
    elif content.strip() == 'Depression':
        form = DepressionForm()
    elif content.strip() == 'Schizophrenia':
        form = SchizophreniaForm()

    if form.validate_on_submit():
        if ((form.concern1.data == form.concern2.data) or
            (form.concern1.data == form.concern3.data) or
            (form.concern2.data == form.concern3.data)):
            error_msg = "Side effect fields must not be duplicates"
            return furtherquestions(content, title=error_msg)
        
        answers = {}
        answers['Diagnosis'] = content
        answers['SE1'] = form.concern1.data
        answers['SE1_rate'] = form.concern1_rating.data
        answers['SE2'] = form.concern2.data
        answers['SE2_rate'] = form.concern2_rating.data
        answers['SE3'] = form.concern3.data
        answers['SE3_rate'] = form.concern3_rating.data
        answers['alcohol'] = (form.lifestyle.data == 1 or form.lifestyle.data == 3)
        answers['marijuana'] = (form.lifestyle.data == 2 or form.lifestyle.data == 3)
        answers['addiction'] = bool(form.addiction.data)
        answers['eff_rating'] = form.eff_rating.data

        flash("I have all the answers!")
        return answers

    return render_template("furtherquestions.html", title=title, form=form)

@app.route("/about")
def about():
    return render_template("about.html")
                           

if __name__=="__main__":
    app.run(debug=True)


