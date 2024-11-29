from fastapi import FastAPI, HTTPException
import uvicorn

from pydantic import BaseModel, PrivateAttr, Field, PositiveFloat, computed_field

app = FastAPI()


class Item(BaseModel):
    name: str
    price: PositiveFloat
    tax: PositiveFloat | None = Field(None, gt=0.0, lt=1.0)

    @computed_field
    def price_with_tax(self) -> float:
        coef_tax = (1 + self.tax) if self.tax else 1
        return self.price * coef_tax


items = {
    1: Item(name="item 1", price=1.0, tax=0.2),
    2: Item(name="item 2", price=3.0),
    3: Item(name="item 3", price=5.0, tax=0.05)
}


@app.get("/")
def root():
    return {"message": "Hello!"}


@app.get("/items/list")
def read_items():
    return {"items": items}


@app.get("/items/{item_id}")
def read_item(item_id: int):

    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")

    return {"item_id": item_id, "item": items[item_id]}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):

    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")

    items[item_id] = item
    return {"item_id": item_id, "item": item}


@app.post("/items/")
async def create_item(item: Item):

    item_id = max(items.keys()) + 1 if len(items) !=0 else 1
    items.update({item_id: item})
    return {"item_id": item_id, "item": item}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",
                port=8000, reload=True)
