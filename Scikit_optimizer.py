import Gann_test_MSE as Ann
from skopt import gp_minimize, dump, load
from skopt.plots import plot_convergence
import datetime
import matplotlib.pyplot as plt

# problem specific variabels
nbits = 15
size = 500
insize = nbits
outsize = nbits+1

# search function
def f(params):
    epochs, lrate, dims, layer_size = params
    nodes = [layer_size] * dims
    nodes.insert(0, insize)
    nodes.append(outsize)
    return -Ann.countex(epochs=epochs, lrate=lrate, dims=nodes)

# Search space
space=[(5000,15000)# epochs
    ,(0.05,0.5) # lrate
    ,(0,4)    # dimensions
    ,(0,45)  # layer size
    ,('relu','sigmoid','tanh')]


try:
    res = load('scikit_results/test2.pk1')
except:
    res = gp_minimize(f, space, n_calls=10)

plot_convergence(res)
plt.show()

file = open('scikit_results/simple_export.txt', 'a')
file.write('New test started running at: ' + str(datetime.datetime.now()) + '\n')

while True:
    for x in range(len(res.func_vals)-10,len(res.func_vals)):
        file.write('result;'+str(res.func_vals[x])+';params;'+str(res.x_iters[x])+'\n')
    file.close()
    file = open('scikit_results/simple_export.txt', 'a')
    res = gp_minimize(f, space,n_calls=10,x0=res.x_iters,y0=res.func_vals)
    dump(res, 'scikit_results/test2.pk1')
    print('just finished saving at: ' + str(datetime.datetime.now()))
    try:
        f = open('scikit_results/stop.txt')
        break
    except:
        continue

print('optimizer finished')
#print(res.func_vals)
