import flask
from flask import request, jsonify
import pymongo
import json
import bson.json_util as json_util
from operator import length_hint 
app = flask.Flask(__name__)
app.config["DEBUG"] = True


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["kaggle"]
wine = db["wine"]
cursor = wine.find().limit(1000)
db_data=json_util.dumps(list(cursor))

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''


@app.route('/api/v1/resources/wine/all', methods=['GET'])
def api_all():
    return jsonify(db_data)


@app.route('/api/v1/resources/wine', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    for data in db_data:
        if data['id'] == id:
            results.append(data)

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)

app.run()