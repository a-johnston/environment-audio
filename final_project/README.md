This is a project for CWRU EECS 440 Fall 2016.

The code is written for Python 3.5. It may be compatible with other versions.

# Example interactive usage:
```python
from dataset import *
from model import *
from main import run

dataset = Dataset.mock()
model = RandomClassifier(dataset)

run(model, dataset, iterations=10000)
```

# Example CLI usage:
```
./trainer.py RandomClassifier mock
```

Models must subclass the ```model.Model``` class, which provides methods for training, online classification, and accuracy testing. When initiated, models are provided the dataset in order to directly provide the required input and output shape. They can additionally take ```*args and **kwargs``` on a per-model basis.

# Example scripts:

## Neural Network
```
./trainer.py SimpleFFNet data [1000,1000]

```
The values in the array are the number of nodes in each layer expressed as a JSON array. The network is fully connected with no dropout. The data parameter specifies the folder in which the data is stored.

## Naive Bayes
```
python nbayes.py
python skNaiveBayes.py
```
```nbayes``` runs a naive Bayes model that discretizes the data before training. ```skNaiveBayes``` evaluates the naive Bayes performance using a Gaussian naive Bayes model. The data folder and other model parameters are parameter variables in the respective files.

# Other Key Files
```dataset.py``` is a utility file that provides classes and methods for parsing data into a common format that other algorithms use. It also handles the processing of .wav input files

# File Organization
The algorithms expect the data to be organized into folders by label in a ```data/``` folder. For example, for an inside/outside binary classifier, the .wav files are sorted into two folders (inside and outside). The directory structure would be:
```
root
 |- data
     |- inside
     |- outside
```