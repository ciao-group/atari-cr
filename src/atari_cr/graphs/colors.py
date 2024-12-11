from typing import TypedDict

class Colors(TypedDict):
    blue: str
    orange: str

COLORBLIND: Colors = {
    "blue": "#74ADD1",
    "orange": "#FDAE61",
}