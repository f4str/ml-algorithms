import os
import sys
from sklearn import svm

path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'src'))
sys.path.append(path)

import data_loader

# load data
training_data = data_loader.load_training_data()
test_data = data_loader.load_test_data()
print('data loaded')

# train network
clf = svm.SVC()
clf.fit(training_data[0], training_data[1])
print('training complete')

# test
predictions = [int(x) for x in clf.predict(test_data[0])]
correct = sum(int(x == y) for x, y in zip(predictions, test_data[1]))
print(f'{correct} / {len(test_data[1])} correct')
