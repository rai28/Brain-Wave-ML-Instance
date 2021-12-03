from flask import *
import json
import time
from ML_Model.main import runmodel
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
    text = userData['blog']
    # pass text to runmodel function
    result = runmodel(text)
    # return result
    return json.dumps(result)


if __name__ == '__main__':
    
    app.run("localhost", port=9000 , debug=True)
