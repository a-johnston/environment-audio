This is a project for CWRU EECS 440 Fall 2016.

Example interactive usage:
```python
from dataset import *
from model import *
from main import run

dataset = Dataset.mock()
model = RandomClassifier(dataset)

run(model, dataset, iterations=10000)
```

Example CLI usage:
```
./trainer.py RandomClassifier mock
```

Models must subclass the ```model.Model``` class, which provides methods for training, online classification, and accuracy testing. When initiated, models are provided the dataset in order to directly provide the required input and output shape. They can additionally take ```*args and **kwargs``` on a per-model basis.
