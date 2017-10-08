import random as r
import Gann_test_MSE as Ann
from datetime import datetime
# Problem definition
nbits = 15
size = 500
insize = nbits
outsize = nbits+1


# Hyperparameters
layers_max = 2
layers_min = 0
nodes_max = nbits*4
nodes_min = 8
learn_max = 0.4
learn_min = 0.05
epochs = 8000



f = open("random_search_results.txt",'a')
f.write('test run on: '+str(datetime.now())+"\n")
# generate random values, do test run and save results
while True:
    # initialization
    layers = r.randint(layers_min, layers_max)
    lrate = r.random() * (learn_max - learn_min) + learn_min
    layer_size = r.randint(nodes_min, nodes_max)
    nodes = [layer_size] * layers
    nodes.insert(0,insize)
    nodes.append(outsize)

    result = Ann.countex(epochs=epochs,lrate=lrate,dims=nodes)

    f.write('result:  '+str(result)+'  lrate:  '+str(lrate)+'  epochs:  '+str(epochs)+'  dims:  '+str(nodes)+"\n")
f.write("\n \n \n")
print('done')