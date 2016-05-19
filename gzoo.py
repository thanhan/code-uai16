import csv
import crowd_model
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re


def read(loc = "data.csv", lim = 10000, tasks = [1,2]):
    
    csvfile = open(loc, 'rb')
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    cnt = 0
    
    dic = [] # list of dics, one for each task
    for i in range(len(tasks)): dic.append({})
    
    #tasks = map(lambda x: x -1, tasks)
    
    for row in reader:
        if row[1] == "\N":continue
        task = int(row[0])
        if task not in tasks: continue
        task = tasks.index(task)
        item = row[2]
        lw   = ( int(row[1]), row[3] ) # label, worker
        
        
        #if task == 1:
        #    print task, lw
        if item not in dic[task]: dic[task][item] = ([],[])
        dic[task][item][0].append(lw[0])
        dic[task][item][1].append(lw[1])
        
        cnt += 1
        if cnt >= lim: break
    
    csvfile.close()
    
    return dic
    


def mv_label(l):
    """
    return mv label in the list l
    """
    u = np.unique(l).tolist()
    f = np.zeros_like(u)
    for x in l:
        i = u.index(x)
        f[i] += 1 

    imax = np.argmax(f)
    return u[imax]

def create_cond_agg(dic):
    """
    keep task 2 labels only when ans to task 1 is correct (MV)

    """
    for item in dic[1].keys():
        if item not in dic[0]: continue
        truel = mv_label(dic[0][item][0])
        # find set of workers that provide the 'wrong' label for this item
        wrong_w = []
        for (l, w) in zip(*dic[0][item]):
            if l != truel:
                wrong_w.append(w)
        # delete labels that this wrong set gives in same item task 2
        (new_ls, new_ws) = ([],[])
        for (l, w) in zip(*dic[1][item]):
            if w not in wrong_w:
                new_ls.append(l)
                new_ws.append(w)
        dic[1][item] = (new_ls, new_ws)

   


    
def take_workers(dic, l = 10):
    """
    return list of workers with more than l labels
    """
    res = []
    for (w, val) in dic.items():
        if val[2] > l:
            res.append(w)
            
    return res
    
    
def filter_lc(lc, workers, min_labels = 3):
    """
    items in lc:
    keep those from listed workers (keep all if worker = 'all')
    keep items with at least 3 labels
    """
    
    for (index, lw_set) in enumerate(lc.crowd_labels):
            new_wl = ([],[])
            for (i, w) in enumerate(lw_set[1]):
                l = lw_set[0][i]
                if w in workers or workers == 'all':
                    new_wl[0].append(l)
                    new_wl[1].append(w)
                    
            lc.crowd_labels[index] = new_wl
    
    new_crowd_labels = []
    for lw_set in lc.crowd_labels:
        if len(lw_set[0]) >=  min_labels:
            new_crowd_labels.append(lw_set)
    
    lc.crowd_labels = new_crowd_labels
    lc.n = len(lc.crowd_labels)
    
# (1,6) -> (1,14)    
def main(lim = 100000, tasks = [1,2], pos = [1,4], min_l = 10, cond_agg = False):
    dic = read(lim = lim, tasks = tasks)
    
    if cond_agg:
      create_cond_agg(dic) 
    #create gold
    
    lc0 = crowd_model.labels_collection(dic[0].values(), len(dic[0])*[None])
    lc0.to_binary(pos[0])
    
    mv0 = crowd_model.mv_model(lc0)
    lc0_gold = crowd_model.labels_collection(dic[0].values(), mv0.mv_lab > 0.5)
    gold0 = lc0_gold.get_true_ss()
    
    
    lc1 = crowd_model.labels_collection(dic[1].values(), len(dic[1])*[None])
    lc1.to_binary(pos[1])
    
    mv1 = crowd_model.mv_model(lc1)
    lc1_gold = crowd_model.labels_collection(dic[1].values(), mv1.mv_lab > 0.5)
    gold1 = lc1_gold.get_true_ss()
    
    
    # use only those with more than 10L
    list0 = take_workers(gold0, min_l)
    list1 = take_workers(gold1, min_l)
    
    #return (list0, list1)
    
    lc0 = crowd_model.labels_collection(dic[0].values(), len(dic[0]) * [None])
    lc1 = crowd_model.labels_collection(dic[1].values(), len(dic[1]) * [None])
    
    filter_lc(lc0, list0)
    filter_lc(lc1, list1)
    
    return (lc0, lc1)
    

def get_lc(lim = 100000, tasks = [1,2], pos = [1,4], min_worker_l = 10, min_item_l = 3):
    """
    Return a pair of LC.
    One is used to predict the other
    """
    dic = read(lim = lim, tasks = tasks)
    lc0 = crowd_model.labels_collection(dic[0].values(), len(dic[0])*[None])
    lc0.to_binary(pos[0])
    
    mv0 = crowd_model.mv_model(lc0)
    lc0_gold = crowd_model.labels_collection(dic[0].values(), mv0.mv_lab > 0.5)
    gold0 = lc0_gold.get_true_ss()
    
    
    lc1 = crowd_model.labels_collection(dic[1].values(), len(dic[1])*[None])
    lc1.to_binary(pos[1])
    
    mv1 = crowd_model.mv_model(lc1)
    lc1_gold = crowd_model.labels_collection(dic[1].values(), mv1.mv_lab > 0.5)
    gold1 = lc1_gold.get_true_ss()
    
    
    #find common workers with at least min_l labels
    common_w = []
    for w in mv0.dic_ss.keys():
        if w in mv1.dic_ss.keys():
            if gold0[w][2] > min_worker_l and gold1[w][2] > min_worker_l:
                common_w.append(w)
    
    
    filter_lc(lc0, common_w, min_labels = min_item_l)
    filter_lc(lc1, common_w, min_labels = min_item_l)
        
    return (lc0, lc1)




def logit(p):
    if p == 0: p = 0.0000000000000001
    if p == 1: p = 0.9999999999999999
    return np.log( p / (1-p) )
    
def measure_correlation(gold0, gold1, l = 10, pos = 0, list_w = []):
    
    a = []
    b = []
   
    for w in gold0.keys():
      if w in gold1 and (w in list_w or list_w == []):
        if gold0[w][0] != None and gold1[w][0] != None:
          if gold0[w][2] > l and gold1[w][2] > l:
              if pos == 0:
                  a.append(logit(gold0[w][pos]))
                  b.append(logit(gold1[w][pos]))
              else:
                  a.append(logit(1 - gold0[w][pos]))
                  b.append(logit(1 - gold1[w][pos]))
 

    print len(a), np.mean(a), np.mean(b)
    return np.cov(a,b) 
    
    
def create_gold(task, pos, lim = 1000000000, cond_agg = False):
    if type(task) == int: task = [task]
    dic = read(lim = lim, tasks = task)
    print "read dic: DONE"
    if not cond_agg:
        lc0 = crowd_model.labels_collection(dic[0].values(), len(dic[0])*[None])
        filter_lc(lc0, 'all', min_labels = 10)
        lc0.to_binary(pos)
    
        mv0 = crowd_model.mv_model(lc0)
        print "MV: DONE"
        lc0_gold = crowd_model.labels_collection(lc0.crowd_labels, mv0.mv_lab > 0.5)
        gold0 = lc0_gold.get_true_ss()
        return gold0
    else:
        # cond_agg: task = [t0, t1] do use cond_agg; return gold for second task 
        create_cond_agg(dic)
         
        lc1 = crowd_model.labels_collection(dic[1].values(), len(dic[1])*[None])
        filter_lc(lc1, 'all', min_labels = 10)
        lc1.to_binary(pos[1])
    
        mv1 = crowd_model.mv_model(lc1)
        print "MV: DONE"
        lc1_gold = crowd_model.labels_collection(lc1.crowd_labels, mv1.mv_lab > 0.5)
        gold1 = lc1_gold.get_true_ss()
        return gold1


    
    
def load_gold(cond_agg = False):

    import pickle
    f = open('gzoo_gold2.pkl')
    g2 = pickle.load(f)
    f = open('gzoo_gold1.pkl')
    g1 = pickle.load(f)
    f = open('gzoo_gold3.pkl')
    g3 = pickle.load(f)
    #f = open('gzoo_lc2.pkl')
    #lc2 = pickle.load(f)
    #f = open('gzoo_lc1.pkl')
    #lc1 = pickle.load(f)
    #f = open('gzoo_lc3.pkl')
    #lc3 = pickle.load(f)
    f = open('gzoo_gold4.pkl')
    g4 = pickle.load(f)

    f = open('gzoo_gold2_ca.pkl')
    g2_ca = pickle.load(f)
    
    if not cond_agg:
        return (g1, g2, g3, g4)
    else:
        return (g1, g2_ca, g3, g4)


def plot_gold(g1, g2, lc, p = 0):
    """
    plot sen/spe of g1 against g2
    only consider workers in lc
    """

    mv = crowd_model.mv_model(lc)
    s1 = []; s2 = []

    for w in g1.keys():
        if w in g2 and g1[w][p] != None and g2[w][p] != None and w in mv.dic_ss:
            s1.append(g1[w][p])
            s2.append(g2[w][p])

    plt.xticks((0, 0.5, 1), ("0", "0.5", "1"))
    plt.tick_params(labelsize = 25)
    plt.yticks((0, 0.5, 1), ("0", "0.5", "1"))

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.scatter(s1, s2, marker = '.', s=50, c = 'black')

    plt.xlabel('task 1 sen.', fontsize = 25)
    plt.ylabel('task 2 sen.', fontsize = 25)



def read_log(filenames = []):
    """
    """
    dic_sen = {}
    dic_spe = {}

    for filename in filenames:
        f = open(filename)
        for line in f:
            if line.startswith('Items in new task'): 
                n = int(line.split(':')[1])
            elif line.startswith('single'):
                algo = 'single'
            elif line.startswith('accum'):
                algo = 'accum'
            elif line.startswith('multi'):
                algo = 'multi'
            #elif line.startswith('Average Error'):
            #    esen = float(line.split()[2])
            #    espe = float(line.split()[3])
            elif line.startswith('RMSE'):
                esen = float(line.split()[1])
                espe = float(line.split()[2])

                if (n, algo) not in dic_sen: dic_sen[(n, algo)] = []
                dic_sen[(n, algo)].append(esen)
                if (n, algo) not in dic_spe: dic_spe[(n, algo)] = []
                dic_spe[(n, algo)].append(espe)
 
 
            else:
                continue

        
    return (dic_sen, dic_spe)




def read_log_new(filenames = []):
    """
    new format no "RMSE"
    """
    dic_sen = {}
    dic_spe = {}

    for filename in filenames:
        f = open(filename)
        for line in f:
            algo = None
            if line.startswith('Items in new task'): 
                n = int(line.split(':')[1])
            elif line.startswith('single'):
                algo = 'single'
            elif line.startswith('accum'):
                algo = 'accum'
            elif line.startswith('multi'):
                algo = 'multi'
            else:
                algo = None
            #elif line.startswith('Average Error'):
            #    esen = float(line.split()[2])
            #    espe = float(line.split()[3])
            if algo != None:
                esen = float(re.findall(r"[\d.]+", line)[0])
                espe = float(re.findall(r"[\d.]+", line)[1])
 

                if (n, algo) not in dic_sen: dic_sen[(n, algo)] = []
                dic_sen[(n, algo)].append(esen)
                if (n, algo) not in dic_spe: dic_spe[(n, algo)] = []
                dic_spe[(n, algo)].append(espe)
 
 
            else:
                continue

        
    return (dic_sen, dic_spe)







def read_log_all(t = 'task12'):
    task12 = ['gzoo_1M_12_01.txt','gzoo_1M_12_02.txt','gzoo_1M_12_03.txt','gzoo_1M_12_04.txt' ,'gzoo_1M_12_05.txt']
    #task12 = ['gzoo_1M_12_01.txt', 'gzoo_1M_12_02.txt', 'gzoo_1M_12_03.txt', 'gzoo_1M_12_05.txt']

    task12_ca = ['gzoo_1M_12_01_ca.txt','gzoo_1M_12_02_ca.txt','gzoo_1M_12_03_ca.txt' ,'gzoo_1M_12_04_ca.txt', 'gzoo_1M_12_05_ca.txt']

    task34 = ['gzoo_1M_34_01.txt', 'gzoo_1M_34_02.txt','gzoo_1M_34_03.txt', 'gzoo_1M_34_04.txt', 'gzoo_1M_34_05.txt']

    if t == 'task12':
    	return read_log(task12)
    elif t == 'task12_ca':
        return read_log(task12_ca)
    elif t == 'task34':
        return read_log(task34)



def get_latex_res(dic_sen, dic_spe, vals = None):
    algo = ['single', 'accum', 'multi']
    if vals == None:
        vals = [64, 323, 1295, 6476]

    res = []
    for val in vals:
        s = ""
        for a in algo:
            x = np.mean(dic_sen[(val, a)])
            s = s + ("%.3f" % x) + " & "

        for a in algo:
            x = np.mean(dic_spe[(val, a)])
            s = s + ("%.3f" % x) + " & "

        res.append(s)

    return res


def plot_sen_spe(dic_sen, dic_spe, vals = None):
    """
    """
    label  = {'single': 'Single', 'accum': 'Accum', 'multi': 'Multi'}
    marker = {'single': '.', 'accum': 'x', 'multi': 's'}
    algo = ['single', 'accum', 'multi']
     
    if vals == None:
        vals = [64, 323, 1295, 6476]

    plt.xlim(0,3)
    plt.ylim(0, 0.3)

    for a in algo:
        y = []
        for v in vals:
            x = np.mean(dic_sen[(v, a)])
            y.append(x)
        print a, y
        plt.plot([0, 1, 2, 3], y, label = label[a], marker = marker[a], markersize = 15, linewidth = 5)


    plt.xlabel('Percentage of target task labels', fontsize = 25)
    plt.ylabel('RMSE', fontsize = 30)
    plt.legend(loc = 'upper right', fontsize = 30)
    
    plt.tick_params(labelsize = 25)
    plt.xticks([0,1,2,3], [1, 5, 20, 100])
    plt.yticks((0, 0.15, 0.3), ("0", "0.15", "0.3"))
    #plt.set_xticklabels(['1','','5','','20','','100'])


def plot_multi_err():
    """
    """
    f = open('gzoo1000000_1_2_0.2_pickle.pkl')
    res = pickle.load(f)
    sing = res[(0.5, 'single')]
    multi = res[(0.5, 'multi')]
    (g1, g2, g3, g4) = load_gold()

    a = []; b = []
    for w in multi:
        a.append(abs(g2[w][0]- sing[w][0])); b.append(abs(g2[w][0] - multi[w][0]))
    

    plt.xlim(0,1); plt.ylim(0,1)
    plt.scatter(a, b, marker = '.')
    plt.plot([0, 1], [0, 1], ls="-", c=".5")

    plt.xlabel('single')
    plt.ylabel('multi')




def get_roc(workers, gold_set, a):
    """
    return false pos rate and true pos rate
    """
    ep = 0.00000000001
    tp = ep; fp = ep;
    tn = ep; fn = ep
    for w in workers:
        if w in a:
            # positive
            if w in gold_set: tp += 1
            else: fp+= 1
        else:
            #negative
            if w not in gold_set: tn += 1 
            else: fn += 1

    return (fp*1.0/ (fp+tn), tp*1.0/(tp+fn))



def get_pre_rec(workers, gold_set, a):
    """
    return precision and recall
    """
    ep = 0.00000000001
    tp = ep; fp = ep;
    tn = ep; fn = ep
    for w in workers:
        if w in a:
            # positive
            if w in gold_set: tp += 1
            else: fp+= 1
        else:
            #negative
            if w not in gold_set: tn += 1 
            else: fn += 1

    return (tp*1.0/ (tp+fp), tp*1.0/(tp+fn))



def reduce_dic_ss(accum_dic, single_dic):
    """
    return accum_dic with only items that also appear in single_dic
    """
    new_dic = {}
    for w in single_dic:
        if w in accum_dic:
            new_dic[w] = accum_dic[w]

    return new_dic


def create_roc(filename = 'gzoo1000000_1_2_0.2_pickle.pkl', percent = 1):
    """
    plot the roc scatter plot
    read from pkl file (get the dic_ss)
    """
    f = open(filename)
    res = pickle.load(f)
    sing  = res[(percent, 'single')]
    multi = res[(percent, 'multi')]
    accum = res[(percent, 'accum')]
    accum = reduce_dic_ss(accum, sing) #only take W who appear in single

    print len(sing), len(multi), len(accum)

    (g1, g2, g3, g4) = load_gold()

    # init thresh and list of workers
    workers = []
    thresh = [0.999]
    eps = 0.0001
    for w in multi:
        if w in sing:
            workers.append(w)
            thresh.append(g2[w][0] - eps)
 
    sing_x = []; sing_y = []
    multi_x = []; multi_y = []
    accum_x = []; accum_y = []

    #thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for t in thresh:
        # get set of true worker with sen < t
        gold_set = []
        for w in workers:
            if g2[w][0] < t:
                gold_set.append(w)

        if len(gold_set) < 10: continue

        sing_set = []
        for w in workers:
            if sing[w][0] < t:
                sing_set.append(w)

        multi_set = []
        for w in workers:
            if multi[w][0] < t:
                multi_set.append(w)

        accum_set = []
        for w in workers:
            if accum[w][0] < t:
                accum_set.append(w)


        (x, y) = get_roc(workers, gold_set, sing_set)
        sing_x.append(x)
        sing_y.append(y)

        (x, y) = get_roc(workers, gold_set, multi_set)
        multi_x.append(x)
        multi_y.append(y)

        (x, y) = get_roc(workers, gold_set, accum_set)
        accum_x.append(x)
        accum_y.append(y)

        #print t, get_roc(workers, gold_set, sing_set), get_roc(workers, gold_set, multi_set), get_roc(workers, gold_set, accum_set)
 
 
 

    plt.xlim(0.9,1); plt.ylim(0.9,1)

    plt.scatter(sing_x,   sing_y, color = 'blue',  marker = '.', label = 'single')
    plt.scatter(accum_x, accum_y, color = 'green', marker = '+', label = 'accum')
    plt.scatter(multi_x, multi_y, color = 'red',   marker = 'x', label = 'multi')


    plt.legend(loc = 'lower right')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

