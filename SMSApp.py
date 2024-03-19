from flask import Flask, request, jsonify
from flask import render_template, url_for
import json
from preprocess import preprocessing,predict_sms
# Login page
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Implement login logic here
        # If login is successful, set session and redirect to index
        session['logged_in'] = True
        return redirect(url_for('index'))
    return render_template('login.html')

# Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Implement registration logic here
        # If registration is successful, redirect to login
        return redirect(url_for('login'))
    return render_template('register.html')

# Index page
@app.route('/index', methods=['GET', 'POST'])
def index():
    # if 'logged_in' not in session:
    #     return redirect(url_for('login'))

    if request.method == 'POST':
        sms_text = request.form['Body']
        prediction = predict_sms(sms_text)
        return render_template('index.html', prediction=prediction)

    return render_template('index.html')
