from flask import Flask
from .routes import routes_bp
from whitenoise import WhiteNoise

def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )

    # Registrar blueprint do app principal
    app.register_blueprint(routes_bp)

    # Adicionar WhiteNoise para arquivos estáticos
    app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/")
    
    return app

app = create_app()