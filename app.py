from fastapi import FastAPI
import uvicorn
from api.model import TripRequest
import common
from api.predict import predict

app = FastAPI()

@app.post("/predict/")
async def create_item(trip: TripRequest):
    response_ = predict(trip)
    return {"status": "ok", "data": response_}
    # data = trip.model_dump()
    # return data
    # return {"item_id": item_id, "item": item}


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0",
                port=8000, reload=True)