import os

from cardiospike import PACKAGE_DIR

API_HOST = os.getenv("API_HOST", "localhost")
STATIC_DIR = str(PACKAGE_DIR / "web" / "static")
