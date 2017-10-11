import matplotlib.pyplot as PLT
import Gann_test
import Gann_test_MSE
import tflowtools as TFT

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
#gann = Gann_test.wine_classifier(epochs=4000,dims=[11,18,6],mbs=30,lrate=0.004,activation_func=['relu','relu'],sm=True)
gann = Gann_test.glass_classifier(epochs=3001,dims=[9,27,27,27,27,9,6],mbs=20,lrate=0.2,activation_func=['relu','relu','relu','relu','relu','relu'],sm=True)
#gann = Gann_test.yeast_classifier(epochs=2000,dims=[8,24,10],mbs=30,lrate=0.004,activation_func=['relu','relu','relu'],sm=True)
PLT.show()

#print(TFT.gen_wine_cases()[0])

#import sys
#print(getattr(tf.nn,"crelu"))

def test_case_man():
    nbits = 4
    vfrac = 0.1
    tfrac=0.1
    case_generator = (lambda: TFT.gen_all_one_hot_cases(2 ** nbits))
    cman = Gann_test.Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    print(cman.get_validation_cases())