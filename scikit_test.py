from sklearn import preprocessing
import numpy as np
import tflowtools as TFT

def gen_yeast_cases():
    f = open('datasets/yeast.txt','r')
    features, labels = ([],[])
    for line in f:
        split_line = list(map(float,line.split(",")))
        features.append(split_line[:len(split_line)-1])
        labels.append(split_line[-1]-1)
    features = np.array(features)
    features_scale = preprocessing.scale(features)
    output = []
    for i in range(len(features_scale)):
        output.append([features_scale[i].tolist(),TFT.int_to_one_hot(int(labels[i]),10)])
    return output

y = gen_yeast_cases()
