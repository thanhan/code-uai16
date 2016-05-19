import util

import active
import numpy as np
import util2
import scipy.stats
import random

mat = None
rel = None
turk_data = None
turk_data_uncer = None
turk_data_id = None

bal_mat = None
bal_rel = None
bal_turk_data = None
bal_turk_data_uncer = None
bal_turk_data_id = None

dic_workers = None
import pickle

def main(dataset = 'proton-beam', read_file = False, rand_shuffle = None):
    global mat, rel, turk_data, turk_data_uncer, turk_data_id, dic_workers
    
    if read_file and dataset == 'RCT':
        f = open('start_RCT.pkl')
        (turk_data_id, rel) = pickle.load(f)
        f.close()
        return

    if dataset.startswith('sim'): # simulated data
        (rel, turk_data_id, dic_workers) = simulate(dataset)
        return

    util.main(dataset)
    mat = util.mat
    rel = util.rel


    if dataset.startswith('RCT'):
        turk_data_id = sorted(util.turk_dic.items())
        turk_data_id = map(lambda x: zip(*x), list(zip(*turk_data_id)[1]))
        turk_data_id = map(lambda a: ( list(a[0]), list(a[1]) ), turk_data_id )
    else:
        util2.main(dataset, util.turk_dic)
        turk_data = util2.turk_data
        turk_data_uncer = util2.turk_data_uncer
        turk_data_id = util2.turk_data_id
        
        
    if rand_shuffle != None:
        random.shuffle(turk_data_id, lambda : rand_shuffle)
        random.shuffle(rel, lambda : rand_shuffle)


def get_balance_d():
    n = len(rel)
    a = np.arange(n)
    np.random.shuffle(a)

    n0 = 0; n1 = 0; indices = []
    for i in a:
        x = rel[i]
        if n0 < n1 and x == 1: continue
        if n1 < n0 and x == 0: continue
        indices.append(i)
        if x == 0: n0 += 1
        if x == 1: n1 += 1

    global bal_mat, bal_rel, bal_turk_data, bal_turk_data_uncer, bal_turk_data_id
    bal_mat = mat[indices]
    bal_rel = [rel[i] for i in indices]
    #bal_turk_data = [turk_data[i] for i in indices]
    #bal_turk_data_uncer = [turk_data_uncer[i] for i in indices]
    bal_turk_data_id = [turk_data_id[i] for i in indices]

if __name__ == "__main__":
    main()


### Simulation:
# Mean/ var of U,V
m = [2, -3]
var = 1

w = 50
n = 20000
theta = 0.05
wk_per_item = 3

def S(x):
    return 1.0 / ( 1.0 + np.exp(-x))

def get_worker_labels(true, sen_fpr):
    (sen, fpr) = sen_fpr
    if true == 1:
        if random.random() < sen:
            return 1
        else:
            return 0
    else:
        if random.random() < fpr:
            return 1
        else:
            return 0

def select_worker(k):
    """
    the second half k times more likely
    """
    #x = np.nonzero( np.random.multinomial(1, np.asarray([1,2,3,4,5])/15.0, size=1) )[1][0]
    #y = np.random.randint(10)
    #return x * 10 + y

    #x = np.nonzero( np.random.multinomial(1, np.asarray([1,k])/(1.0+k*1.0), size=1) )[1][0]
    if np.random.random() < 1.0 / ( 1.0 + k * 1.0 ):
        return np.random.randint(w / 2)
    else:
        return (w/2) + np.random.randint(w / 2)

def simulate(data_name):
    """
    Generate simulated data
    data_name contains arguments for simmulation
    """
    argv = data_name.split('_')

    #k = float(argv[2])
    k = 5
    if argv[1] == 'ss':
        cov = float(argv[2])
        #cov = 0
        C = [[var,cov],
             [cov,var]]

        workers = [] #(sen, fpr)
        dic_workers = {}
        for j in range(w):
            x = scipy.stats.multivariate_normal.rvs(m, C)
            sen = S(x[0]); fpr = S(x[1])
            workers.append( (sen, fpr) )
            dic_workers[str(j)] = (sen, 1 - fpr)

        rel = [] # true label
        turk_data_id = []

        for i in range(n):
            true = scipy.stats.bernoulli.rvs(theta)
            rel.append ( true )
            turk_data_id.append ( ([], []) )
            #list_workers = range(w); random.shuffle(list_workers)
            selected_workers = [select_worker(k) for count in range(wk_per_item)]
            for j in selected_workers:
                #print j, len(workers), i, len(turk_data_id)
                l = get_worker_labels(true, workers[j])
                turk_data_id[i][0].append(l)
                turk_data_id[i][1].append(str(j))

    elif argv[1] == 'tc':
        alpha = float(argv[2])
        beta = 1

        workers = [] #(sen, fpr)
        dic_workers = {}
        for j in range(w):

            sen = random.betavariate(alpha, beta)
            fpr = 1 - random.betavariate(alpha, beta)
            workers.append( (sen, fpr) )
            dic_workers[str(j)] = (sen, 1 - fpr)

        rel = [] # true label
        turk_data_id = []

        for i in range(n):
            true = scipy.stats.bernoulli.rvs(theta)
            rel.append ( true )
            turk_data_id.append ( ([], []) )
            #list_workers = range(w); random.shuffle(list_workers)
            selected_workers = [select_worker(k) for count in range(wk_per_item)]
            for j in selected_workers:
                #print j, len(workers), i, len(turk_data_id)
                l = get_worker_labels(true, workers[j])
                turk_data_id[i][0].append(l)
                turk_data_id[i][1].append(str(j))

    return (rel, turk_data_id, dic_workers)
