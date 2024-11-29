
from pydantic import BaseModel, computed_field

class DateTimeBase(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int

    def to_datetime_str(self):
        return f"{self.year}-{self.month}-{self.day} {self.hour}:{self.minute}:{self.second}"

class TripRequest(BaseModel):
    vendor_id: int
    pickup_datetime_: DateTimeBase
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str

    @computed_field
    def pickup_datetime(self) -> str:
        return self.pickup_datetime_.to_datetime_str()