import start
import active
import crowd_model
import sklearn
import numpy as np
import util
import sys
import random
import pickle
import copy
import matplotlib.pyplot as plt
import gzoo
import scipy


def get_true_fnp(data, rel):
    n = len(rel)

    fn = 0; fp = 0

    for i in range(n):
        l = data[i][0][0]
        true = rel[i]
        if true == 0 and l == 1:
            fp += 1
        elif true == 1 and l == 0:
            fn += 1

    return (fn, fp)

def lookup_ss(dic, wid):
    if wid in dic: return dic[wid]
    return (1.0, 1.0)

def get_false_prob(ss, l, pc):
    """
    pc = prob of 1
    """
    p0 = 1.0 - pc
    p1 = 1.0 * pc

    sen, spe = ss

    if l == 0:
        p0*= spe; p1*= 1.0 - sen
    else:
        p0*= 1 - spe; p1*= sen

    if l == 0:
        return p1 * 1.0 / (p0 + p1)
    else:
        return p0 * 1.0 / (p0 + p1)



def is_wrong(l, truel):
    if l == truel: return 0
    return 1


def count_items(turk_data):
    if turk_data == None: return (0,0,0)
    (data, rel) = turk_data
    num_labels = 0
    count_fn = 0
    count_fp = 0
    for (index, lw_set) in enumerate(data):
        for (l, worker) in zip(*lw_set):
            num_labels += 1
            if l == 0 and rel[index] == 1:
                count_fn += 1
            elif l == 1 and rel[index] == 0:
                count_fp += 1
                
            #ss = lookup_ss(dic, worker)
            #if l == 0:
                #print l, ss, get_false_prob(ss, l, pc), is_wrong(l, rel[index])
                #e_fn += abs( get_false_prob(ss, l, pc) - is_wrong(l, rel[index]) )
                #e_fn += pow( get_false_prob(ss, l, pc) - is_wrong(l, rel[index]) , 2)
                #e_fn += get_false_prob(ss, l, pc) - is_wrong(l, rel[index]) 
            #else:
                #print l, ss, get_false_prob(ss, l, pc), is_wrong(l, rel[index])
                #e_fp += abs( get_false_prob(ss, l, pc) - is_wrong(l, rel[index]) )
                #e_fp += pow( get_false_prob(ss, l, pc) - is_wrong(l, rel[index]) , 2)
                #e_fp += get_false_prob(ss, l, pc) - is_wrong(l, rel[index]) 

    return (num_labels, count_fn, count_fp)

def weighted_eval_cm(dic, gold, turk_data = None):

    (num_labels, count_fn, count_fp) = count_items(turk_data)
    
    n = 0
    n_all = 0 # include None
    for (wid, g) in gold.items():
        n_all += g[2]
        if not g[0] == None:
            n += g[2]

    
    
    
    sen = 0; spe = 0; num = 0    
    predict_fn = 0
    predict_fp = 0
    
    for (wid, g) in gold.items():
        if not g[0] == None:
            if wid in dic:
                ss = dic[wid]
                num += 1
            else:
                ss = (0.5, 0.5)
            pr_w = g[2] * 1.0 / n
            sen += pr_w * abs(ss[0] - g[0])
            spe += pr_w * abs(ss[1] - g[1])

    if turk_data != None:
      (data, rel) = turk_data
      pc = sum(rel) * 1.0 / len(rel) # class prior
            
      for (wid, g) in gold.items():
        if wid in dic:
            ss = dic[wid]
        else:
            ss = (0.5, 0.5)
        pr_w = g[2] * 1.0 / n_all
        predict_fn += pc *     pr_w * ( 1 - ss[0] ) 
        predict_fp += (1-pc) * pr_w * ( 1 - ss[1] ) 
    
      predict_fn *= num_labels
      predict_fp *= num_labels
    
    return (sen, spe, predict_fn, predict_fp, count_fn, count_fp)
    #e_fn = 0; e_fp = 0

    


    #(true_fn, true_fp) = get_true_fnp(data, rel)
    #print predict_fn, predict_fp, true_fn, true_fp
    #e_fn = abs(true_fn - predict_fn)
    #e_fp = abs(true_fp - predict_fp)

    #return (sen, spe, num, e_fn*1.0/num_labels, e_fp*1.0/num_labels)
    #return (sen, spe, num, abs(e_fn), abs(e_fp) )
    #return (sen, spe, num, e_fn, e_fp )
    #return (sen, spe, num, e_fn, e_fp)

def eval_cm(dic, gold, ww = False, turk_data_id = None, detail = False, ss_default = (0.5, 0.5), list_w = None):
    """
    evaluate confusion matrix wrt gold
    """
    if ww: return weighted_eval_cm(dic, gold, turk_data_id)
    
    if list_w == None: list_w = gold.keys()
    
    sen = 0; spe = 0; num = 0
    sen1 = 0; spe1 = 0#
    sen2 = 0; spe2 = 0  # total sen/spe error
    x = []; y = []
    for (wid, g) in gold.items():
        if g[0] != None:
            if wid in dic and wid in list_w:
                ss = dic[wid]
                num += 1
                sen1 += abs(ss[0] - g[0])
                spe1 += abs(ss[1] - g[1])
                sen2 += ss[0] - g[0]
                spe2 += ss[1] - g[1]
                
                sen += abs(ss[0] - g[0])
                spe += abs(ss[1] - g[1])
                
                x.append( abs(ss[0] - g[0]) )
                y.append( abs(ss[1] - g[1]) )

                if detail and wid in dic:
                    print wid, ss[0], g[0], ss[0] - g[0]
                
            else:
                #sen += 1
                #spe += 1
                pass
            
            
            
    #print sen1, spe1, sen2
    #print "Average Error", sen*1.0/num, spe*1.0/num
    x = np.asarray(x); y = np.asarray(y)
    #print "RMSE", pow(np.sum(x*x)/num, 0.5) , pow(np.sum(y*y)/num, 0.5)
    #return (sen, spe, num)
    return (pow(np.sum(x*x)/num, 0.5) , pow(np.sum(y*y)/num, 0.5), num)



def main_online(n, dataset = 'RCT', rand_shuffle = None, bs = 500, tc_w = 0.1, vs_w = 0.1):
    start.main(dataset)

    lc = crowd_model.labels_collection(start.turk_data_id, start.rel)
    #lc.preprocess()
    gold_dic = lc.get_true_ss()

    if rand_shuffle == None:
        rand_shuffle = random.random()
    random.shuffle(start.turk_data_id, lambda : rand_shuffle)
    #random.shuffle(start.turk_data   , lambda : rand_shuffle)
    random.shuffle(start.rel         , lambda : rand_shuffle)

    lc1 = crowd_model.labels_collection([], [])
    lc2 = crowd_model.labels_collection([], [])
    lc3 = crowd_model.labels_collection([], [])
    tc = crowd_model.tc_model(lc1)
    vs = crowd_model.vss_model(lc2)
    mv = crowd_model.mv_model(lc3)

    res = []
    for i in range(0, n, bs):
        tc.online_em(start.turk_data_id[i:i+bs], num_it = 5, w = tc_w)
        tc_ss = eval_cm(tc.dic_worker_ss, gold_dic)

        vs.online_em(start.turk_data_id[i:i+bs], no_train = True, w = vs_w)
        vs.get_dic_ss()
        vs_ss = eval_cm(vs.dic_ss, gold_dic)

        mv.online(start.turk_data_id[i:i+bs])
        mv_ss = eval_cm(mv.dic_ss, gold_dic)

        print i + bs, tc_ss, vs_ss, mv_ss
        res.append([i+bs, tc_ss, vs_ss, mv_ss])

    return (res, tc, mv, vs)
    #compare_conf_mat(tc.dic_worker_ss, tc.dic_worker_ss, gold = gold_dic)

    #vss = crowd_model.vss_model(lc)
    #vss.em()
    #vss.get_dic_ss();
    #compare_conf_mat(vss.dic_ss, vss.dic_ss, gold = gold_dic)

def main_offline(n = None, dataset = 'proton-beam', rand_shuffle = None, alpha = 1, num_it = 3):
    start.main(dataset)

    lc = crowd_model.labels_collection(start.turk_data_id, start.rel)
    #lc.preprocess()

    if dataset.startswith('sim'):
        gold_dic = start.dic_workers
    else:
        gold_dic = lc.get_true_ss()

    if rand_shuffle == None:
        rand_shuffle = random.random()
    random.shuffle(start.turk_data_id, lambda : rand_shuffle)
    #random.shuffle(start.turk_data   , lambda : rand_shuffle)
    random.shuffle(start.rel         , lambda : rand_shuffle)

    if n == None: n = len(start.rel)
    #lc1 = crowd_model.labels_collection(start.turk_data_id[:n], np.hstack((start.rel[:ngold], (nitem -ngold)*[None])))
    lc1 = crowd_model.labels_collection(start.turk_data_id[:n], n*[None])

    #print n, lc1.crowd_labels[0]
    for alpha in [1]:
        tc = crowd_model.tc_model(lc1)
        tc.em(num_it = num_it)
        tc_ss = eval_cm(tc.dic_worker_ss, gold_dic)

        #print "tc ", alpha, tc_ss
        print "tc ", tc_ss
        sys.stdout.flush()

    lc2 = crowd_model.labels_collection(start.turk_data_id[:n], n*[None])
    mv = crowd_model.mv_model(lc2)
    mv_ss = eval_cm(mv.dic_ss, gold_dic)


    print "mv", mv_ss
    sys.stdout.flush()

    lc3 = crowd_model.labels_collection(start.turk_data_id[:n], n*[None])

    for full_cov in [False, True]:
        vs = crowd_model.vss_model(lc3, full_cov = full_cov)
        vs.em(num_it = num_it)
        vs.get_dic_ss();
        vs_ss = eval_cm(vs.dic_ss, gold_dic)

        print "vs Full_Cov = ", full_cov,  vs_ss
        sys.stdout.flush()

    return (gold_dic, tc, mv, vs)


def main_sim(dataset, n, num_it = 20):
    #dataset = 'sim-ss-' + str(c)
    c = float(dataset.split('_')[2])
    for i in range(10):
		print c
		main_offline(n, dataset, alpha = c, num_it = num_it)


def main_real(dataset, n, num_it = 20):
    for i in range(10):
        print n
        main_loss(n, dataset, num_it = num_it, rand_shuffle = 0.1*i )


#save_turk = None
#save_rel = None
#start = None

def restore_start():
    global start
    start.turk_data_id = copy.deepcopy(save_turk)
    start.rel = copy.deepcopy(save_rel)

def main_loss(n = None, dataset = 'RCT', rand_shuffle = None, num_it = 3, split = None, prior = 1):
    """
    Save worker sen/spe
    Estimate loss (FP + FN)
    Error = Weighted by worker prevalance
    prior = prior for the crowd model
    """

    start.main(dataset, True)
    #restore_start()
    
    lc = crowd_model.labels_collection(start.turk_data_id, start.rel)
    gold_dic = lc.get_true_ss()

    if dataset == 'RCT':
        split = 151224 # take all the data
    else:
        split = len(start.rel) / 2

    random.shuffle(start.turk_data_id, lambda : rand_shuffle)
    random.shuffle(start.rel, lambda : rand_shuffle)
    test_data = (start.turk_data_id[split:], start.rel[split:])


    crowd_model.global_psen = (prior,1); crowd_model.global_pspe = (prior,1); crowd_model.global_pfpr = (1,prior)

    lc1 = crowd_model.labels_collection(start.turk_data_id[:n], n*[None])

    tc = crowd_model.tc_model(lc1)
    tc.em(num_it)
    tc_ss = eval_cm(tc.dic_worker_ss, gold_dic)
    print "tc", tc_ss; sys.stdout.flush()

    #hc = crowd_model.hc_model(lc1)
    #hc.build_model_def()
    #hc.infer_dic_ss()
    #tc_ss = eval_cm(tc.dic_worker_ss, gold_dic, True, test_data)
    #hc_ss = eval_cm(hc.dic_ss, gold_dic)
    #print "hc ", hc_ss; sys.stdout.flush()

    
    lc2 = crowd_model.labels_collection(start.turk_data_id[:n], n*[None]);
    mv = crowd_model.mv_model(lc2)
    #mv_ss = eval_cm(mv.dic_ss, gold_dic, True, test_data)
    mv_ss = eval_cm(mv.dic_ss, gold_dic)

    print "mv", mv_ss; sys.stdout.flush()

    lc3 = crowd_model.labels_collection(start.turk_data_id[:n], n*[None]);
    #for full_cov in [False, True]:
    vs_diag = crowd_model.vss_model(lc3, full_cov = False)
    vs_diag.em(num_it = num_it)
    vs_diag.get_dic_ss();
    #vs_diag_ss = eval_cm(vs_diag.dic_ss, gold_dic, True, test_data)
    vs_diag_ss = eval_cm(vs_diag.dic_ss, gold_dic)
    print "vs Full_Cov = False",  vs_diag_ss; sys.stdout.flush()

    vs_full = crowd_model.vss_model(lc3, full_cov = True)
    vs_full.em(num_it = num_it)
    vs_full.get_dic_ss();
    #vs_full_ss = eval_cm(vs_full.dic_ss, gold_dic, True, test_data)
    vs_full_ss = eval_cm(vs_full.dic_ss, gold_dic)

    print "vs Full_Cov = True",  vs_full_ss; sys.stdout.flush()

    # save sen-spe:
    filename = 'save_ss_' + dataset + ' ' + str(n) + '_' + str(rand_shuffle)
    f = open(filename, 'w')
    pickle.dump((tc.dic_worker_ss, mv.dic_ss, vs_diag.dic_ss, vs_full.dic_ss), f)


def reproduce(n = None, dataset = 'RCT', rand_shuffle = None, num_it = 3, split = None):
    """
    read save_ss files
    reproduce evaluation
    """
    
    filename = 'save_ss_' + dataset + ' ' + str(n) + '_' + str(rand_shuffle)
    f = open(filename, 'r')
    (tc_dic, mv_dic, vs_diag_dic, vs_full_dic) = pickle.load(f)
    
    start.main(dataset)
    lc = crowd_model.labels_collection(start.turk_data_id, start.rel)
    gold_dic = lc.get_true_ss()
    random.shuffle(start.turk_data_id, lambda : rand_shuffle)
    random.shuffle(start.rel, lambda : rand_shuffle)
    test_data = (start.turk_data_id[split:], start.rel[split:])

    print n    
    print "tc ", eval_cm(tc_dic, gold_dic, True, test_data)
    print "mv ", eval_cm(mv_dic, gold_dic, True, test_data)
    print "vs Full_Cov = False ", eval_cm(vs_diag_dic, gold_dic, True, test_data)
    print "vs Full_Cov = True " , eval_cm(vs_full_dic, gold_dic, True, test_data)
    f.close()
    
    
def reprodude_output(n = 1000, dataset = 'RCT', split = 80000):
    #global start, save_turk_id, save_rel
    
    #start.main(dataset)
    
    #save_turk = copy.deepcopy(start.turk_data_id)
    #save_rel = copy.deepcopy(start.rel)
    
    for i in range(0, 10, 1):
        reproduce(n, dataset, 0.1*i, 3, split)
    



def setup(dataset = 'proton-beam', n = 1000, ngold = 0, rand_shuffle = None):
    start.main(dataset)
    
    if rand_shuffle != None:
        random.shuffle(start.turk_data_id, lambda : rand_shuffle)
        random.shuffle(start.rel, lambda : rand_shuffle)
    
    lc_gold = crowd_model.labels_collection(start.turk_data_id, start.rel)
    gold_dic = lc_gold.get_true_ss()
    
    lc1 = crowd_model.labels_collection(start.turk_data_id[:n], start.rel[:ngold] + (n-ngold)*[None])
    tc = crowd_model.tc_model(lc1)
    
    lc2 = crowd_model.labels_collection(start.turk_data_id[:n], start.rel[:ngold] + (n-ngold)*[None])
    mv = crowd_model.mv_model(lc2)

    lc3 = crowd_model.labels_collection(start.turk_data_id[:n], start.rel[:ngold] + (n-ngold)*[None])
    vs_full = crowd_model.vss_model(lc3, full_cov = True)
    
    lc4 = crowd_model.labels_collection(start.turk_data_id[:n], start.rel[:ngold] + (n-ngold)*[None])
    vs_diag = crowd_model.vss_model(lc3, full_cov = False)
    
    return (gold_dic, mv, tc, vs_full, vs_diag)



def main(nitem, ngold, nsam, dataset = 'proton-beam'):
    start.main(dataset)
    sep_val = nitem;
    money = (10000000000, 1, 100)
    #adata = active.active_data(start.mat[:sep_val,:], start.turk_data[:sep_val], start.rel[:sep_val], start.turk_data_id[:sep_val], start.turk_data_uncer[:sep_val], money=money))

    for i in range(nitem):
        adata.query_crowd_all(0)

    for i in range(ngold):
        adata.query_expert_fix(i)


    (res, res_mv, dic_ds) = util.aggregate(has_gold = ngold > 0)

    print sklearn.metrics.confusion_matrix(start.rel[ngold:nitem], np.asarray(res[ngold:]) > 0.5)

    print sklearn.metrics.roc_auc_score(start.rel[ngold:nitem], res[ngold:])

    print sklearn.metrics.confusion_matrix(start.rel[ngold:nitem], np.asarray(res_mv[ngold:]) > 0.1)

    print sklearn.metrics.roc_auc_score(start.rel[ngold:nitem], res_mv[ngold:])


    #return (res, res_mv)

    #ss = crowd_model.ss_model(adata.lc, 1, 2, 0.1, nsam)
    #ss.em()
    #ss.infer_true_l()

    #print ss.mu
    #print ss.C

    #print sklearn.metrics.confusion_matrix(start.rel[ngold:nitem], np.asarray(ss.prob[ngold:]) > 0.5)

    #print sklearn.metrics.roc_auc_score(start.rel[ngold:nitem], ss.prob[ngold:])

    #ss.workers_ss()
    #compare_conf_mat(dic_ds, ss.dic_ss)


    return adata
    
    


def compare_conf_mat(dic_ds, dic_ss, gold = None, filename = 'temp/proton_gold.txt'):
    ss_sen = 0
    ds_sen = 0
    ss_spe = 0
    ds_spe = 0
    if gold == None:
        f = open(filename)
        gold = {}
        for line in f:
            l = line.split()
            wid = l[0]
            if l[1] == "None": continue
            sen = float(l[1])
            spe = float(l[2])
            gold[wid] = (sen, spe)
        f.close()
    num = 0
    for (wid, gold) in gold.items():
        if not gold[0] == None and wid in dic_ss and wid in dic_ds:
            num += 1
            ss = dic_ss[wid]
            ds = dic_ds[wid];
            ss_sen += abs(ss[0] - gold[0])
            ds_sen += abs(ds[0] - gold[0])
            ss_spe += abs(ss[1] - gold[1])
            ds_spe += abs(ds[1] - gold[1])
            print "gold = ", (gold[0], gold[1]), "first = ", (ds[0], ds[1]), "second = ", (ss[0], ss[1])

    print (ds_sen, ds_spe), (ss_sen, ss_spe), num
    
    
def main_multitask_sr(rand_shuffle = None, num_prev = 3):
    """
    Multitask
    Systematic review
    """
    
    start.main('proton-beam', rand_shuffle = rand_shuffle)
    proton_n = len(start.turk_data_id)
    proton_turk_data = copy.deepcopy(start.turk_data_id)
    proton_rel = copy.deepcopy(start.turk_data_id)
    proton_lc = crowd_model.labels_collection(proton_turk_data, proton_n * [None])
    proton_vs = crowd_model.vss_model(proton_lc)
    
    start.main('appendicitis', rand_shuffle = rand_shuffle)
    appen_n = len(start.turk_data_id)
    appen_turk_data = copy.deepcopy(start.turk_data_id)
    appen_rel = copy.deepcopy(start.turk_data_id)
    appen_lc = crowd_model.labels_collection(appen_turk_data, appen_n * [None])
    appen_vs = crowd_model.vss_model(appen_lc)
    
    
    start.main('dst', rand_shuffle = rand_shuffle)
    dst_n = len(start.turk_data_id)
    dst_turk_data = copy.deepcopy(start.turk_data_id)
    dst_rel = copy.deepcopy(start.turk_data_id)
    dst_lc = crowd_model.labels_collection(appen_turk_data, appen_n * [None])
    dst_vs = crowd_model.vss_model(appen_lc)
    
    start.main('omega3', rand_shuffle = rand_shuffle)
    lc_gold = crowd_model.labels_collection(start.turk_data_id, start.rel)
    gold_dic = lc_gold.get_true_ss()
    
    if num_prev == 3:
        prev_data = proton_turk_data + appen_turk_data + dst_turk_data
        prev_vs   = [proton_vs, appen_vs, dst_vs]
    elif num_prev == 2:
        prev_data = proton_turk_data + appen_turk_data
        prev_vs   = [proton_vs, appen_vs]
    else:
        prev_data = proton_turk_data
        prev_vs   = [proton_vs]
    
    n = len(start.turk_data_id)
    for m in [100, 200, 500, 1000]:
        new_lc = crowd_model.labels_collection(start.turk_data_id[:m], m*[None])
        single_task = crowd_model.vss_model(new_lc)
        
        accum_lc = crowd_model.labels_collection(prev_data + \
                                                 start.turk_data_id[:m], (proton_n + m)*[None])
        accum = crowd_model.vss_model(accum_lc)
        
        new_vs = crowd_model.vss_model(new_lc)
        multi =  crowd_model.multitask(prev_vs + [new_vs], inter_cor = 0.1)
        
        single_task.em(4)
        accum.em(4)
        multi.em(3)
        
        print m
        #print "single", weighted_eval_cm(single_task.dic_ss, gold_dic)
        #print "accum ", weighted_eval_cm(      accum.dic_ss, gold_dic)
        #print "multi ", weighted_eval_cm(multi.datasets[2].dic_ss, gold_dic)
        
        print "single", eval_cm(single_task.dic_ss, gold_dic)
        print "accum ", eval_cm(      accum.dic_ss, gold_dic)
        print "multi ", eval_cm(multi.datasets[-1].dic_ss, gold_dic)
        #print multi.C
    
    
    
def main_multitask(list_lc, list_gold, rand_shuffle = 0.1, list_per = [1, 5, 20, 100]):

    random.shuffle(list_lc[0].crowd_labels, lambda : rand_shuffle)
    random.shuffle(list_lc[1].crowd_labels, lambda : rand_shuffle)
    
    
    # reduce gold to only those in data:
    for i in range(len(list_lc)):
        lc = list_lc[i]
        gold = list_gold[i]
        mv = crowd_model.mv_model(lc)
        new_gold = {}
        for (w, val) in gold.items():
            if w in mv.dic_ss:
                new_gold[w] = val
        list_gold[i] = new_gold
    
    prev_data = list_lc[0].crowd_labels
    
    prev_vs   = [crowd_model.vss_model(list_lc[0])]
    
    
    res = {}# (algo, size) -> dic_ss

    for p in list_per:
        m = int(p / 100.0 * len(list_lc[-1].crowd_labels))
        new_lc = crowd_model.labels_collection ( list_lc[-1].crowd_labels[:m], m * [None])
        single_task = crowd_model.vss_model(new_lc)
        
        accum_lc = crowd_model.labels_collection(prev_data + \
                                                 new_lc.crowd_labels[:m], (len(prev_data) + m)*[None])
        accum = crowd_model.vss_model(accum_lc)
        
        new_vs = crowd_model.vss_model(new_lc)
        multi =  crowd_model.multitask(prev_vs + [new_vs], inter_cor = 0.2)
        
        #return (single_task, multi, list_gold)
        print "Items in new task:", m
        print "Total Labels:", new_lc.total_labels()
        
        
        
        single_task.em(4)
        list_w = single_task.dic_ss.keys()# list of workers apprears in m first items
        print "single", eval_cm(single_task.dic_ss, list_gold[-1], list_w = list_w)
        
        accum.em(4)
        print "accum ", eval_cm(      accum.dic_ss, list_gold[-1], list_w = list_w)
        
        multi.em(3)
        print "multi ", eval_cm(multi.datasets[-1].dic_ss, list_gold[-1], list_w = list_w)
        print multi.mu
        print multi.C        
        sys.stdout.flush()

        res[(p, 'single')] = single_task.dic_ss
        res[(p, 'accum')]  = accum.dic_ss
        res[(p, 'multi')]  = multi.datasets[-1].dic_ss
        
    
    return res

def extract_parens(s):
    #print s
    s = s[s.find('('):]
    x = float ( s.split()[0].strip('(),)') )
    y = float ( s.split()[1].strip('(),)') )
    #print (x,y)
    return (x,y)
    
def read_multi(filename, pos = 0):
    f = open(filename)
    
    algo = ['single', 'accum', 'multi']
    
    dic = {}
    
    for num in range(3):
        dic = {}
        print f.readline().strip()
        for rs in np.arange(0, 1, 0.1):
          for m0 in [100, 200, 500, 1000]:
            m = int( f.readline().strip() )
            
            for a in algo:
                res = extract_parens( f.readline().strip() )
                
                if (m,a) not in dic: dic[(m,a)] = []
                dic[(m, a)].append(res[pos])
            #print line

    f.close()
    
    return dic
    
def plot_multi(dic):

    algo = ['single', 'accum', 'multi']
    x = [100, 200, 500, 1000]
    
    label  = {'single': 'Singletask', 'accum': 'Accumulate', 'multi': 'Multitask'}
    marker = {'single': 'o', 'accum': 'x', 'multi': '.'}
    #msize = {'mv': 15, 'tc': 5, 'vs_false': 15, 'vs_true': 5}
    
    fig, ax = plt.subplots()
    
    for a in algo:
        y = []
        for p in x:
            res = dic[(p, a)]
            #m = np.asarray(res) - np.asarray(true)
            #y.append(np.mean(m))
            #z.append(np.std(res))
            y.append(np.mean(res))

            #print algo, p, m, np.mean(m), np.std(m)
        print algo, y
        num = len(y)
        #ax.errorbar(range(num), y, yerr=z, label = label[algo], marker = marker[algo])
        ax.plot(range(num), y, label = label[a], marker = marker[a], markersize = 5)
    
    
    ax.set_xticklabels(["100","","200","", "500","","1000"])
    ax.legend(loc = 'upper right')
    
    plt.xlabel("Size")
    plt.ylabel("Error")


def main_gzoo_multi():
    #multitask for gzoo
    (lc0, lc1) = gzoo.main(1000000, [1,2], [1,4], 100)
    import pickle
    f = open('gzoo_gold01.pkl')
    (gold0, gold1) = pickle.load(f)
    f.close()
    
    list_rs = [] 
    for i in sys.argv:
        if str.isdigit(i[0]):
            list_rs.append(float(i))
            
    print list_rs
    
    for rs in list_rs:
        print "rs = ", rs
        main_multitask([lc0, lc1], [gold0, gold1], rs)


# task position (of positive label)
tpos = [0, 1, 4, 6, 8]

def main_gzoo2(lim, t1 = 1, t2 = 2, rs = 0.1, cond_agg = False):
    g = [0,1,2,3,4]
    (g[1], g[2], g[3], g[4]) = gzoo.load_gold(cond_agg)
    #(lc1, lc2) = gzoo.get_lc(lim, [1,2], [1,4], 100)
    #(lc1, lc2) = gzoo.main(lim, [1,2], [1,4], 100)
    (lc1, lc2) = gzoo.main(lim, [t1, t2], [tpos[t1], tpos[t2]], 100, cond_agg = cond_agg)
    res = main_multitask([lc1, lc2], [g[t1], g[t2]], rs) 
    return res
   

def S(x):
    return crowd_model.S(x)



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


def get_labels(workers, n, theta, w, wk_per_item = 5):
    rel = [] # true label
    turk_data_id = []

    for i in range(n):
            true = scipy.stats.bernoulli.rvs(theta)
            rel.append ( true )
            turk_data_id.append ( ([], []) )
            selected_workers = range(w); random.shuffle(selected_workers)
            selected_workers = selected_workers[:wk_per_item]
            #selected_workers = [select_worker(k) for count in range(wk_per_item)]
            for j in selected_workers:
                #print j, len(workers), i, len(turk_data_id)
                l = get_worker_labels(true, workers[j])
                turk_data_id[i][0].append(l)
                turk_data_id[i][1].append(str(j))
    
    return (rel, turk_data_id)

def simulate_multitask(x):
    """
    Generate simulated data
    data_name contains arguments for simmulation
    """
    n = 5000
    w = 200
    theta = 0.5
    if True:
        #cov = 0
        #m = [2, -2, 2, -2]
        #C = [[2, 0, x, 0],
        #     [0, 2, 0, x],
        #     [x, 0, 2, 0],
        #     [0, x, 0, 2]]
        m =  [1.49,  -1.45, 2.18, -2.59]
        C =  [[1.80,    0,   x,   0], 
              [   0, 1.30,   0,   x],
              [   x,    0,1.06,   0],
              [   0,    x,   0,   1.89]]

        workers1 = [] #(sen, fpr)
        dic_workers1 = {}
        workers2 = []
        dic_workers2 = {}
        for j in range(w):
            x = scipy.stats.multivariate_normal.rvs(m, C)
            sen_1 = S(x[0]); fpr_1 = S(x[1])
            sen_2 = S(x[2]); fpr_2 = S(x[3])

            workers1.append( (sen_1, fpr_1) )
            dic_workers1[str(j)] = (sen_1, 1 - fpr_1)
            workers2.append( (sen_2, fpr_2) )
            dic_workers2[str(j)] = (sen_2, 1 - fpr_2)

       
        rel1, turk1 = get_labels(workers1, n, theta, w)
        rel2, turk2 = get_labels(workers2, n, theta, w)

        return (dic_workers1, rel1, turk1, dic_workers2, rel2, turk2)


def main_sim_multi(cor = 0.75, rs = 0.5):
    """
    multitask simulated data
    """
    dic1, rel1, turk1, dic2, rel2, turk2 = simulate_multitask(cor)
    lc1 = crowd_model.labels_collection(turk1, rel1)
    lc2 = crowd_model.labels_collection(turk2, rel2)

    for rs in [0.1, 0.2, 0.3, 0.4, 0.5]:
        res = main_multitask([lc1, lc2], [dic1, dic2], rs) 
        import pickle
        f = open('simult_' + str(cor) + '.pkl', 'w')
        pickle.dump(res, f)
        f.close()
 
   


################################################
################################################
# multitask on simulated data
if __name__ == "SIM__main__":
    cor = float (sys.argv[1])
    print cor
    main_sim_multi(cor)
   
################################################
# experiments on Galaxy Zoo
if __name__ == "Gzoo__main__":
    lim = int  (sys.argv[1])
    t1  = int  (sys.argv[2])
    t2  = int  (sys.argv[3])
    rs  = float(sys.argv[4])
    ca  = sys.argv[5] == 'ca'

    print ca
    res = main_gzoo2(lim, t1, t2, rs, ca)

    import pickle
    f = open('gzoo' + sys.argv[1] + "_" + sys.argv[2] + '_' + sys.argv[3] + '_' + sys.argv[4] + '_' + sys.argv[5] + '.pkl', 'w')
    pickle.dump(res, f)
    f.close()


# experiments on RCT
if __name__ == "__main__":

    #dataset = 'proton-beam'
    dataset = 'RCT'
    
    #dataset = sys.argv[1]
    
    #for n in [100, 200, 500, 1000]:
    #for n in [2000, 4000]:
    #    print n
    
    prior = int(sys.argv[1])
    # multitask for systematic review
    for n in [100, 500, 2000, 10000, 40000, 150000]:
        print n, prior
        for rs in np.arange(0.1, 0.6, 0.1):
            main_loss(n, dataset, rand_shuffle = rs, prior = prior)
            
     
       
    #n = int(sys.argv[2])
    #num_it = int(sys.argv[3])
    #c = float(sys.argv[2])
    #if dataset.startswith("sim"):
    #    main_sim(dataset, n, num_it)
    #else:
    #    main_real(dataset, n, num_it)
    #for i in range(5):
    #n = None
    #for dataset in ['proton-beam', 'dst', 'omega3', 'appendicitis']:
        #print dataset
        #print n
        #main_offline(n, dataset)

    
