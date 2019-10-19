from flask import Flask, request, render_template
from vocab import Vocabulary
from collections import Counter, defaultdict
import json
import numpy as np

from lib import QueryEngine

print("Initializing query engine...")
CLUBS = json.load(open('clubs.json'))
ENGINE = QueryEngine()

for club in CLUBS:
    ENGINE.add_document(
        [(club['name'], 3.0),
         (club['mission'], 1.0)])
ENGINE.recompute_stats()


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    result = ""
    for id, scores in ENGINE.query(text):
        result += CLUBS[id]['name'] + '<br/>'
    return result

@app.route('/api/list', methods=['GET', 'POST'])
def api_list():
    tpe = request.values.get('type', None)
    result = []
    if tpe == 'club' or tpe is None:
        result.extend(CLUBS)
    return json.dumps(result)

@app.route('/api/search', methods=['GET', 'POST'])
def api_search():
    tpe = request.values.get('type', None)
    query = request.values.get('query', None)

    result = []
    for id, scores in ENGINE.query(query):
        result.append({
            'id': id,
            'scores': scores})
    return json.dumps(result)

if __name__ == "__main__":
    print("Starting up!")
    app.run(host='0.0.0.0', port=8080)
