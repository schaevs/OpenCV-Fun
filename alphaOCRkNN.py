import numpy as np
import cv2
import matplotlib.pyplot as plt

samplesDir = '/home/alexanderschaevitz/opencv/samples/data/'
data = np.loadtxt( samplesDir + 'letter-recognition.data', dtype = 'float32', 
                   delimiter = ',', converters = {0: lambda ch: ord(ch)-ord('A')}
                 )

train, test = np.vsplit(data,2)

responses, trainData = np.hsplit(train,[1])
labels,     testData = np.hsplit(test,[1])

knn = cv2.KNearest()
knn.train(trainData, responses)
ret, result, neighbours, dist = knn.find_nearest(testData, k=3)

correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000
print accuracy
