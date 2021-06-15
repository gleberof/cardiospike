from fastapi import FastAPI

from cardiospike.api.routes import router

app = FastAPI(title="CardioSpike API", docs_url="/")
app.include_router(router=router)
