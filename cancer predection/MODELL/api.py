import os 
import json
import pandas as pd
import numpy
from flask import Flask, render_template, request, jsonify
from pandas.io.json import json_normalize
from sklearn.externals import joblib
from werkzeug.wrappers import Request, Response
from flask_cors import CORS





app = Flask(__name__)
port = int(os.getenv('PORT', 5500))
CORS(app)
@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/api/model', methods=['POST'])
def model():
    if request.method == 'POST':
        try:
            post_data = request.get_json()
            json_data = json.dumps(post_data)
            s = pd.read_json(json_data)          
            p = joblib.load("./classifier_model.pkl")
            r = p.predict(s)
            return str(r)
        
        except Exception as e:
            return (e)
			
if __name__ == '__main__':

	app.run(host='0.0.0.0', port=port, debug=True)
