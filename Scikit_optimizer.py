import Gann_test as Ann
from skopt import gp_minimize, dump, load
from skopt.plots import plot_convergence
import datetime
import matplotlib.pyplot as plt

dataset = 'yeast'

# dict for different runs
export_names = {'bit':'bit','yeast':'yeast','glass':'glass','wine':'wine'}
scikit_names = {'bit':'bit','yeast': 'yeast','glass':'glass','wine':'wine'}
classifier_names = {'bit':'countex','yeast':'yeast_classifier','glass':'glass_classifier','wine':'wine_classifier'}
insizes = {'bit':15,'glass':9,'yeast':8,'wine':11}
outsizes = {'bit':16,'glass':6,'yeast':10,'wine':6}
nbitss = {'bit': 15}
sizes = {'bit': 500}

# problem specific variabels
export_name = export_names[dataset]
scikit_name = scikit_names[dataset]
classifier_name = classifier_names[dataset]
nbits = 8
size = 500
insize = insizes[dataset]
outsize = outsizes[dataset]

# search function
def f(params):
    epochs, lrate, dims, layer_size, act_func, mbs = params
    act_func=[act_func]*(dims+2)
    nodes = [layer_size] * dims
    nodes.insert(0, insize)
    nodes.append(outsize)
    return -getattr(Ann, classifier_name)(epochs=epochs, lrate=lrate, dims=nodes,activation_func=act_func,mbs=mbs)

# Search space
space=[(4000,8000)# epochs
    ,(0.0001,0.005) # lrate
    ,(0,2)    # dimensions
    ,(0,45)  # layer size
    ,('sigmoid','tanh','relu',)  # activation function
    ,(10,32)  # MBS
    ]


try:
    res = load('scikit_results/'+scikit_name+'.pk1')
except:
    res = gp_minimize(f, space, n_calls=10)

#plot_convergence(res)
#plt.show()

file = open('scikit_results/'+export_name+'.txt', 'a')
file.write('New test started running at: ' + str(datetime.datetime.now()) +'on: '+dataset+ '\n')

while True:
    for x in range(len(res.func_vals)-10,len(res.func_vals)):
        file.write('result;'+str(res.func_vals[x])+';params;'+str(res.x_iters[x])+'\n')
    file.close()
    file = open('scikit_results/'+export_name+'.txt', 'a')
    res = gp_minimize(f, space,n_calls=10,x0=res.x_iters,y0=res.func_vals)
    dump(res,'scikit_results/'+scikit_name+'.pk1')
    print('just finished saving at: ' + str(datetime.datetime.now()))
    try:
        f = open('scikit_results/stop.txt')
        break
    except:
        continue

print('optimizer finished')
#print(res.func_vals)
