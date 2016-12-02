This is a project for CWRU EECS 440 Fall 2016.

Example usage:
```python
from dataset import *
from model import *
from main import run

dataset = Dataset.mock()
model = RandomClassifier(dataset)

run(model, dataset, iterations=10000)
```
