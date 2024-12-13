from flask import Flask

# Initialize the Flask app here
app = Flask(__name__, template_folder="../templates")

# Import routes to make sure they are registered with the app
from app import routes
