from flask import Flask,Blueprint
from app.predictor import load_galaxy_model
from flask_restplus import Api
from tensorflow import get_default_graph
app = Flask(__name__)

app = Flask(__name__)
app.config.from_object('config.Config')

blueprint = Blueprint('api',__name__,url_prefix='/api')
api_name = 'Galaxy Morphology Predictor'
api = Api(blueprint,default=api_name,doc='/documentation')
app.register_blueprint(blueprint)

model = load_galaxy_model()
graph = get_default_graph()

from app import views


