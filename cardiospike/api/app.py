from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cardiospike.api.routes import router

app = FastAPI(title="CardioSpike API", docs_url="/")

origins = ["http://localhost", "localhost", "0.0.0.0"]

app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

app.include_router(router=router)
