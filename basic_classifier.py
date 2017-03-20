import csv
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

f = open("/Users/crista/Documents/Michigan/Classes/Security/MalGAN/MalGAN/data/API_truncation50_random_split_trainval_1gram_feature.csv")
training_data = csv.reader(f)

g = open("/Users/crista/Documents/Michigan/Classes/Security/MalGAN/MalGAN/data/API_truncation50_random_split_test_1gram_feature.csv")
testing_data = csv.reader(g)

train_features = []
train_labels = []

for row in training_data:
	print row
	row = [ int(item) for item in row ]
	train_features.append(row[ : len(row)-1])
	train_labels.append(row[len(row)-1 : ])

positive_labels = [ 1 for item in train_labels if item == 1]
negative_labels = [1 for item in train_labels if item == 0]
print "Total training instances:", len(train_features)
print "Positive labels:", len(positive_labels)
print "Negative labels:", len(negative_labels)

clf = svm.SVC()
clf.fit(train_features, train_labels)

test_features = []
test_labels = []

for row in testing_data:
	print row
	row = [ int(item) for item in row ]
	test_features.append(row[ : len(row)-1])
	test_labels.append(row[len(row)-1 : ])

print "Total testing instances:", len(test_features)
predictions = clf.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print "Accuracy:", accuracy