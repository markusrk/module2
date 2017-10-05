import Gann_test
import time
import tflowtools as TFT
import tutor3

#f_and_l = TFT.gen_all_parity_cases(3)
#f = []
#l = []
#for x in f_and_l:
#    f.append(x[0])
#    l.append(x[1])
#print( len(TFT.gen_all_parity_cases(3)))

#TFT.dendrogram(f,l)

gann = Gann_test.mapping_test()


def test_case_man():
    nbits = 4
    vfrac = 0.1
    tfrac=0.1
    case_generator = (lambda: TFT.gen_all_one_hot_cases(2 ** nbits))
    cman = Gann_test.Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    print(cman.get_validation_cases())