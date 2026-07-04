"""uWSGI (WSGI) 用エントリ。FastAPI (ASGI) を a2wsgi でラップする。"""
from a2wsgi import ASGIMiddleware

from src.api.app import app

application = ASGIMiddleware(app)
