# APP imports
from app import app,api,model,graph
from app.predictor import process_image

# Packages
from flask_restplus import reqparse,Resource
from flask import render_template
import numpy as np
import json

parser = reqparse.RequestParser()
parser.add_argument('image',type=str)

@api.route("/predict")
class predict(Resource):
    '''accepts an image array and returns survey predictions for morphology shape'''
    def post(self):
        args = parser.parse_args()
        image = np.array(json.loads(args['image']))
        image = process_image(image)
        with graph.as_default():
            prediction = model.predict(image)
        return prediction.tolist()[0]

@app.route("/")
def index():
    return render_template("index.html")

