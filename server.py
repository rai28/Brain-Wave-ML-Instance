from flask import *
import json
import time
from ML_Model.main import runmodel
from ML_Model.summarizer import summarizer
import os
app = Flask(__name__)


# home route for the server
@app.route('/', methods=['GET'])
def homeRoute():
    # return a status code of 200
    return 'Server is running', 200


# when a post request is made to the server, then get userData from the request and pass it to the runmodel function and return the result
@app.route('/detect-emotion', methods=['POST'])
def get_user_data():
    # get json data from request
    userData = request.get_json()
    # get text property from json data
    text = userData['diary']
    #calling the function summarizer
    summarizedText = summarizer(text)
    
    # pass text to runmodel function
    result = runmodel(text)
    # return result and summarized text
    return json.dumps({'emotion': result, 'summarizedText': summarizedText}), 200


if __name__ == '__main__':

    from waitress import serve
    serve(app, host="0.0.0.0", port=9000)