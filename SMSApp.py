from flask import Flask, request, jsonify
from flask import render_template, url_for
import json
from preprocess import preprocessing,predict_sms
import os
import psycopg2

DATABASE_URL = os.environ['DATABASE_URL']

conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cur = conn.cursor()
def execute_query(query):
    try:
        cur.execute(query)
        conn.commit()
        print("Query executed successfully.")
    except (Exception, psycopg2.Error) as error:
        print("Error executing query:", error)
    finally:
        if conn:
            cur.close()
            conn.close()
create_tables_query = """
-- Create the User table
CREATE TABLE "User" (
  UserID SERIAL PRIMARY KEY,
  Username VARCHAR(255),
  Password VARCHAR(255),
  Role VARCHAR(255),
  Email VARCHAR(255)
);

-- Create the SMSMessage table
CREATE TABLE SMSMessage (
  MessageID SERIAL PRIMARY KEY,
  MessageText TEXT
);

-- Create the ClassificationResult table
CREATE TABLE ClassificationResult (
  UserID INT,
  MessageID INT,
  IsSpam BOOLEAN,
  PRIMARY KEY (UserID, MessageID),
  FOREIGN KEY (UserID) REFERENCES "User"(UserID),
  FOREIGN KEY (MessageID) REFERENCES SMSMessage(MessageID)
);

-- Create the Feedback table
CREATE TABLE Feedback (
  UserID INT,
  MessageID INT,
  FeedbackText VARCHAR(255),
  PRIMARY KEY (UserID, MessageID),
  FOREIGN KEY (UserID) REFERENCES "User"(UserID),
  FOREIGN KEY (MessageID) REFERENCES SMSMessage(MessageID)
);
"""

execute_query(create_tables_query)
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
