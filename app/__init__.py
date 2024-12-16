from flask import Flask

def create_app():
    app = Flask(__name__, template_folder="templates")
    
    # Import and register the blueprint
    from .routes import routes_bp
    app.register_blueprint(routes_bp)
    
    return app

app = create_app()