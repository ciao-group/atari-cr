""" Plot that shows the correlation between the pause frequencies of humans and the agent. """
from polars import DataFrame

data = DataFrame({
    "Category": ["Agent", "Human"],
    "Asterix": [5.2, 2.8],
    "Seaquest": [3.5, 0.9],
    "H.E.R.O.": [0, 0.3],
})
print(data)