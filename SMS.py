from flask import Flask, request, jsonify
from flask import render_template, url_for
import json
from preprocess import preprocessing,predict_sms
# Define the API endpoint for receiving SMS messages
app = Flask(__name__)
@app.route('/',methods=['GET'])
def home():
     return render_template('home.html')

@app.route('/', methods=['POST'])
def sms_process():
    input_text = request.form.get('Body')
 
    if input_text is None:
        return jsonify({'prediction': 'No input text provided'})
    response=predict_sms(input_text)
    
    return  jsonify(response)





if __name__ == '__main__':
    app.run(debug=True)
