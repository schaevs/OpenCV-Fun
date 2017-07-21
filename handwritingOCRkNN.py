#following tutorial from opencv-python-tutroals.readthedocs.io

import numpy as np
import cv2
from matplotlib import pyplot as plt

samplesDir = '/home/alexanderschaevitz/opencv/samples/data/'
img = cv2.imread(samplesDir + 'digits.png')
print img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#split image into 5000 20x20 cells
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

x = np.array(cells)

train = x[:,:50].reshape(-1,400).astype(np.float32) # 2500 x 400
test = x[:,50:100].reshape(-1,400).astype(np.float32) # 2500 x 400

# create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# initiative kNN, train the data, then test with test data for k = 1
knn = cv2.KNearest()
knn.train(train,train_labels)
ret, result, neighbours, dist = knn.find_nearest(test,k=5)

#check accuracy of classification
#compare result with test labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy
