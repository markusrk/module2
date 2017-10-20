import matplotlib.pyplot as PLT
import Gann_test
import tflowtools as TFT
import numpy as np
import matplotlib as PLT

#f_and_l = TFT.gen_all_parity_cases(3)
#f = []
#l = []
#for x in f_and_l:
#    f.append(x[0])
#    l.append(x[1])
#print( len(TFT.gen_all_parity_cases(3)))

#TFT.dendrogram(f,l)

#t = tf.nn.relu(5)
#print(t.op)

#gann = Gann_test.countex(epochs=6000,dims=[15,10,16],mbs=30,lrate=0.002,activation_func=['tanh','tanh'],sm=True)
#gann = Gann_test.wine_classifier(epochs=10000,dims=[11,40,20,6],mbs=1000,lrate=0.8,activation_func=[ 'tanh', 'tanh', 'tanh', 'tanh'],sm=True)
#gann = Gann_test.glass_classifier(epochs=3001,dims=[9,27,27,27,27,9,6],mbs=20,lrate=0.2,activation_func=['relu','relu','relu','relu','relu','relu'],sm=True)
#gann = Gann_test.yeast_classifier(epochs=2000,dims=[8,24,10],mbs=30,lrate=0.004,activation_func=['relu','relu','relu'],sm=True)
#PLT.show()

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("datasets/MNIST/", one_hot=True)
#print("")

#m =[[1,2],[1,2]]
#m = np.array(m)
#p = TFT.hinton_plot(m)



# def test_case_man():
#     nbits = 4
#     vfrac = 0.1
#     tfrac=0.1
#     case_generator = (lambda: TFT.gen_all_one_hot_cases(2 ** nbits))
#     cman = Gann_test.Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
#     print(cman.get_validation_cases())

#
# import json
#
# def test_function(v1,v2):
#     print(str(v1+v2))
#
# d = {'dims':[9,40,20,6],
#      'activation_func':'tanh',
#      'softmax':True,
#      'cost_func':'CE',
#      'lr': 0.5,
#      'initial_weight_range':'scaled',
#      'data_source':'gen_glass_cases',
#      'case_count': None,
#      'vfrac':0.1,
#      'tfrac':0.1,
#      'mbs':170,
#      'map_bs':20,
#      'epochs':100,
#      'map_layers':None,
#      'dendogram_layers':None,
#      'acc_lim': 0.95,
#      }
# f = open("configs/fast.txt",'w')
# json.dump(d,f,indent=4)


# import tf_raw3 as raw
# import tensorflow as tf
#
# y =[[0.,1.]]
# x =[[5.,5.]]
#
# sess= tf.InteractiveSession()
# a = raw.CE(y,x)
# print(sess.run(a))

import numpy as np

n = np.array([1,1,1,1])
for x, v in np.ndenumerate(n):
     print(str(x)+str(v))