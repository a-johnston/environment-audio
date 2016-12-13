from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics

from dataset import Dataset

import numpy as np

acc_accum = 0.0
acc_count = 10

for i in range(0, acc_count):
    for classifier in [GaussianNB()]:
        dataset = Dataset.load_wavs(data_folder='data', split=0.9, sample_length=1.0)
        training = dataset.training()
        testing = dataset.testing()

        classifier.fit(training[0], np.argmax(training[1], axis=1))

        score = metrics.accuracy_score(np.argmax(testing[1], axis=1), classifier.predict(testing[0]))
        acc_accum += score
print('Accuracy (%s): %f' % (type(classifier), acc_accum / float(acc_count),))
