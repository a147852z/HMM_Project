"""啟動：python -m uvicorn kcore.main:app --reload --port 8900"""
from .api import create_app

app = create_app()
