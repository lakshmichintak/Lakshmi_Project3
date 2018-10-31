import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import flask
# define the app
app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default


# API route
@app.route('/api', methods=['POST'])
def api():
    """API function
    All model-specific logic to be defined in the get_model_api()
    function
    """
    clf = 'finalized_model.sav'
   #Load the saved model
    print("Loading the model...")
    loaded_model = None
    with open(clf,'rb') as f:
     loaded_model = pickle.load(f)

    print("The model has been loaded...doing predictions now...")


    input_data = request.json
    x = np.matrix(input_data["example"])
    df = pd.DataFrame(x,columns=['Credit_History','LoanAmount'])
    print("DataFrame")
    print(df)
    predictions = loaded_model.predict(df)
    print("predictions",predictions)
    y_pred = list(predictions)
    # Put the result in a nice dict so we can send it as json
    #results = {"predicted": y_pred[0]}
    results = {"score": str(y_pred[0])}

    # Return a response with a json in it
    # flask has a quick function for that that takes a dict
    return flask.jsonify(results)


@app.route('/')
def index():
    return "Index API"

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='0.0.0.0', debug=True)

