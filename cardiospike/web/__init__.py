import os

from cardiospike import PACKAGE_DIR

API_HOST = os.getenv("API_HOST", "localhost")

STATIC_DIR = PACKAGE_DIR / "static"
TEMPLATES_DIR = PACKAGE_DIR / "templates"
