# Tracing everything by following this website:
# https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/

# Forms information:
# https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-iii-web-forms

from flask import Flask, render_template, flash, redirect, url_for
from wtforms.fields.core import Label
from config.config import Config

# Importing my own module
from pythonForms.form import LetsGo, DiagnosisForm, ADHDForm, AnxietyForm, BipolarForm, DepressionForm, SchizophreniaForm, SetupForm

app = Flask(__name__)
app.config.from_object(Config)

@app.route("/", methods=['GET','POST'])
def home():
    form = LetsGo()
    if form.validate_on_submit():
        return redirect(url_for('about'))
    return render_template("home.html", form=form)

@app.route("/form", methods=['GET','POST'])
def searchForm():
    form = DiagnosisForm()
    if form.validate_on_submit():
        content = form.diagnosis.data
        return redirect("furtherquestions.html", content=content)
    return render_template("form.html", form=form)

@app.route("/furtherquestions/<string:content>", methods=['GET', 'POST'])
def conditionForm(content):
    if content == 'ADHD':
        SetupForm('ADHD')
        form = ADHDForm()
    elif content == 'Anxiety':
        SetupForm('Anxiety')
        form = AnxietyForm()
    elif content == 'Bipolar-Disorder':
        SetupForm('Bipolar-Disorder')
        form = BipolarForm()
    elif content == 'Depression':
        SetupForm('Depression')
        form = DepressionForm()
    elif content == 'Schizophrenia':
        SetupForm('Schizophrenia')
        form = SchizophreniaForm()
        
    if form.validate_on_submit():
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
    return render_template("furtherquestions.html", form=form)

@app.route("/about")
def about():
    return render_template("about.html")
                           

if __name__=="__main__":
    app.run(debug=True)


