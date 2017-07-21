import numpy as np
import cv2
import matplotlib.pyplot as plt

samplesDir = '/home/alexanderschaevitz/opencv/samples/cpp/'
data = np.loadtxt( samplesDir + 'letter-recognition.data', dtype = 'float32', 
                   delimiter = ',', converters = {0: lambda ch: ord(ch)-ord('A')}
                 )
