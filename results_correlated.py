import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import matplotlib

def read(filename):
    dic = {}
    f = open(filename)
    a = ['n','tc','mv','vs_false', 'vs_true']
    i = 0
    for line in f:
        if line.startswith('iteration'): continue
        
        if (a[i] == 'n') and ( line.strip().isdigit() ):
            n = float(line)
            print n
        else:
            if a[i] == 'n': i+= 1
            #if len(line) <= 7:continue
            #print a[i], line[:10]
            l = line[line.find('('):]
            sen = float(l.split()[0].strip('(),)'))
            spe = float(l.split()[1].strip('(),)'))

            #num = float(l.split()[2].strip('(),)'))
            #efn = float(l.split()[3].strip('(),)'))
            #efp = float(l.split()[4].strip('(),)'))
            predict_fn = float(l.split()[2].strip('(),)'))
            predict_fp = float(l.split()[3].strip('(),)'))
            
            true_fn = float(l.split()[4].strip('(),)'))
            true_fp = float(l.split()[5].strip('(),)'))
            
            if (n, a[i]) not in dic:
                dic[(n, a[i])] = []
            dic[(n, a[i])].append((sen, spe, predict_fn, predict_fp, true_fn, true_fp))

        i+= 1
        if i == 5: i = 0

    f.close()
    return dic

list_files = ["output_correlated.txt", "output_correlated2.txt", "output_correlated3.txt",
              "output_correlated5_1.txt", "output_correlated5_2.txt", "output_correlated5_3.txt",
              "output_correlated_5k.txt", "output_correlated_10k.txt", "output_correlated_20k.txt",
              "output_correlated_40k.txt", "output_correlated_80k.txt", "output_correlated_150k.txt"
              ]

list_sim = ["output_sim_01.txt", "output_sim02.txt", "output_sim03.txt",  "output_sim_0.txt"]

list_wubl = ["output_wubl1.txt", "output_wubl5.txt", "output_wubl10.txt",  "output_wubl20.txt"]

list_crk5 = ["output_crk505.txt", "output_crk50.txt", "output_crk5-1.txt", "output_crk5.txt"]

list_RCT = ["output_RCT_5k.txt", "output_RCT_10k.txt", "output_RCT_20k.txt", "output_RCT_40k.txt"]

list_condor = ["condor/out_RCT_1k_all.txt", "condor/out_RCT_5k_all.txt", "condor/out_RCT_10k_all.txt", "condor/out_RCT_20k_all.txt",
    "condor/out_RCT_40k_all.txt", "condor/out_RCT_80k_all.txt", "condor/out_RCT_2k_all.txt"]


list_small = ["output_proton-beam_500.txt", "output_proton-beam_2000.txt", "output_appendicitis_150.txt", "output_appendicitis_600.txt",
    "output_dst_800.txt", "output_dst_3200.txt", "output_omega3_2400.txt", "output_omega3_600.txt"
    ]


list_rep = ["output_RCT_1k_rep.txt", "output_RCT_5k_rep.txt", "output_RCT_10k_rep.txt", "output_RCT_20k_rep.txt", "output_RCT_40k_rep.txt", "output_RCT_80k_rep.txt", "output_RCT_2k_rep.txt"]

list_byron = ["output_RCT_1k_byron.txt", "output_RCT_5k_byron.txt", "output_RCT_10k_byron.txt", 
    "output_RCT_20k_byron.txt", "output_RCT_40k_byron.txt", "output_RCT_80k_byron.txt"]
    #"output_RCT_2k_byron.txt"]
    
list_proton = ["output_proton-beam_500.txt", "output_proton-beam_1k.txt", "output_proton-beam_2k.txt"]

def read_all(l = list_byron):
    dic = {}
    for f in l:
        dic = read(f, dic)
    return dic


def process(dic, p, s = 0, normalize = 1.0):
    #x = [5000, 10000, 20000, 40000, 80000, 150000]
    #x = [1000, 5000, 10000]
    a = ['vs_true', 'vs_false', 'tc', 'mv']


    data = {}
    for algo in a:
        y = zip(*dic[(p, algo)])[s]
        m = np.mean(y)
        sd = np.std(y)
        print p, algo, "%.4f" % (m/normalize) #, "%.2f" % sd
        data[algo] = np.asarray(y) * 1.0 / normalize
        #print data[algo]

    #print data['mv']
    print 'vsfalse', scipy.stats.ttest_1samp(data['tc'] - data['vs_false'], 0)
    print 'tc', scipy.stats.ttest_1samp(data['tc'] - data['vs_true'], 0)
    print 'mv', scipy.stats.ttest_1samp(data['mv'] - data['vs_true'], 0)

ax = None
fig = None

def plot(dic, id = 0, plot_type = 'fn', plot_true = True, ylim = 0.35):
    #x = [1000, 5000, 10000, 20000, 40000, 80000];#, 80000, 150000]
    #x = [500, 1000, 2000]
    x = [100, 500, 2000, 10000, 40000, 150000];#, 80000, 150000]
    a = ['mv', 'tc', 'vs_false', 'vs_true']
    label  = {'mv': 'Majority Vote', 'tc': 'Two Coin', 'vs_false': 'DiagCov', 'vs_true': 'FullCov'}
    marker = {'mv': 'x', 'tc': '^', 'vs_false': '+', 'vs_true': 's'}
    msize = {'mv': 15, 'tc': 5, 'vs_false': 15, 'vs_true': 5}

    #true_fn = zip(*dic[(1000, a[0])])[4]
    #true_fp = zip(*dic[(1000, a[0])])[5]
    #if plot_type == 'fn':
    #    true = true_fn
    #se:
    #    true = true_fp
    
    global ax, fig
    fig, ax = plt.subplots()
    plt.ylim([0, ylim])

    num = 0
    
    for algo in a:
        y = []
        z = []
        for p in x:
            res = zip(*dic[(p, algo)])[id]
            #m = np.asarray(res) - np.asarray(true)
            #y.append(np.mean(m))
            z.append(np.std(res))
            y.append(np.mean(res))

            #print algo, p, m, np.mean(m), np.std(m)
        print algo, y
        #print algo, z
        num = len(y)
        #ax.errorbar(range(num), y, yerr=z, label = label[algo], marker = marker[algo])
        ax.plot(range(num), y, label = label[algo], marker = marker[algo], markersize = msize[algo])
        #set_xticklabels(nonRepetitive_x)
        #ls = '-', marker = '.'kk
    #ax.set_xticklabels(["1","","2", "", "5","","10","","20","","40", "", "80"])
    
    if plot_true:
        if plot_type == 'fn':
            ax.plot(range(num), num * [np.mean(true_fn)], label = 'True')
        else:
            ax.plot(range(num), num * [np.mean(true_fp)], label = 'True')
        
    #ax.set_xticklabels(["1", "5","10","20","40","80"])
    #ax.set_xticklabels(["500","", "1K","","2K"])
    ax.set_xticklabels(["100", "500", "2000" ,  "10000",  "40000",  "150000"])
    ax.legend(loc = 'upper right')
    #ax.legend(loc = 'lower left')
    plt.xlabel("Number of items")
    plt.ylabel("RMSE")
    #plt.ylabel("Predicted Number")

def plot_gold(gold):
    #plt.xlim([0.2,1])
    #plt.ylim([0.7,1])
    x = []
    y = []
    for (wid,(sen, spe, n)) in gold.items():
      if wid.startswith('S'):
        x.append(sen)
        y.append(spe)
    plt.scatter(x,y, c = 'r', marker = 'o', label = 'Novice')

    x = []; y = []
    for (wid,(sen, spe, n)) in gold.items():
      if wid.startswith('E'):
        x.append(sen)
        y.append(spe)
    plt.scatter(x,y, c = 'b', marker = 'x', label = 'Expert')

    plt.legend(loc = 'lower left')
    plt.xlabel("Sensitivity")
    plt.ylabel("Specificity")



def read_rct(filename):
    dic = {}
    f = open(filename)
    a = ['n','tc','mv','vs_false', 'vs_true']
    i = 0
    cycle = 0 # 1 cycle = tc, mv, vs_f, vs_t
    for line in f:
        if (a[i] == 'n'):
            n = int(line.split()[0])
            print n
        else:
            l = line[line.find('('):]
            sen = float(l.split()[0].strip('(),)'))
            spe = float(l.split()[1].strip('(),)'))
 
            if (n, a[i]) not in dic:
                dic[(n, a[i])] = []
            dic[(n, a[i])].append((sen, spe))

        i += 1
        if i == 5: 
            i = 1
            cycle += 1
            if cycle % 5 == 0:
                i = 0

    f.close()
    return dic


