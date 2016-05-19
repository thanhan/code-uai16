import pymc
import util
import numpy as np
import scipy.stats
import math
import sys
import matplotlib.pyplot as plt


def logit(p):
    if p == 1.0:
        return 3
    if p == 0:
        return -3
    return np.log(p) - np.log(1 - p)

def rct_gold():
    f = open('temp/RCT-gold2.txt', 'r')
    lsen = []
    lspe = []
    for line in f:
        (wid, sen , spe) = line.split()
        sen = logit(float(sen))
        spe = 1 - logit(float(spe))

        lsen.append(sen)
        lspe.append(spe)

    lsen = np.asarray(lsen)
    lspe = np.asarray(lspe)

    f.close()

    return (lsen, lspe)

class labels_collection:
    """
    Crowd labels and True labels
    """
    def __init__(self, crowd_labels, true_labels, write_files = False):
        """
        crowd_labels is a list of (answers, Worker ID):
        E.g.
        ([0, 0, 0, 0, 0],
         ['A27QCN3GYCZTPE',
          'A997OZ3H2B3Q2',
          'AV3CA0IYNRCCJ',
          'A13W8W81TPORYZ',
          'A31A4YKVSOYRVS'])

        write_files = write files for SQUARE/ Get-another-label input
        """
        self.crowd_labels = crowd_labels
        self.true_labels  = true_labels
        self.write_files = write_files
        self.n = len(crowd_labels)

        if self.write_files:
            util.write_aggregate_init()
            self.write_file(crowd_labels, true_labels, start_index = 0)

    def preprocess(self):
        """
        Delete duplicates
        Delete the answer "not sure"
        """
        for cl in self.crowd_labels:
            n = len(cl[1])
            for i in range(n-2, -1, -1):
                if cl[1][i] in cl[1][i+1:] or cl[0][i] < 0:
                    del cl[0][i]
                    del cl[1][i]


    def to_binary(self, positive_lab = 1):
        for (index, lw_set) in enumerate(self.crowd_labels):
            for (i, l) in enumerate(lw_set[0]):
                if l == positive_lab:
                    self.crowd_labels[index][0][i] = 1
                else:
                    self.crowd_labels[index][0][i] = 0

    def add_labels(self, new_crowd_l, new_true_l):
        """
        """
        self.crowd_labels.extend(new_crowd_l)
        self.true_labels.extend(new_true_l)
        self.n += len(new_crowd_l)
        if self.write_files:
            self.write_file(new_crowd_l, new_true_l, start_index = self.n)
            self.n = self.n + len(new_crowd_l)

    def add_true_label(self, index, l):
        """
        Add one true label to an item with crowd labels
        """
        self.true_labels[index] = l
        if self.write_files:
            output_golds = [(index, l)]
            util.write_aggregate_next(golds = output_golds, crowds = [])

    def write_file(self, crowd_labels, true_labels, start_index = 0):
        #create list of output lines
        output_golds = []
        for (i, true_l) in enumerate(true_labels):
            if true_l != None:
                output_golds.append((start_index + i, true_l))
        output_crowds = []
        for (index, lw_set) in enumerate(crowd_labels):
            for (label, worker) in zip(*lw_set):
                output_crowds.append((worker, start_index + index, label))
        util.write_aggregate_next(golds = output_golds, crowds = output_crowds)


    def get_fnfp(self, dic_conf):
        dic = {}
        for (w, c) in dic_conf.items():
            fn = c[1][0]
            fp = c[0][1]
            n  = c[0][0] + c[0][1] + c[1][0] + c[1][1]
            dic[w] = (fn, fp, n)
            
        return dic

    def get_true_ss(self, get_all = False):
        """
        get the true ss for each worker
        from the true labels
        """
        dic_conf = {} # dic of confusion matrix
        for (index, lw_set) in enumerate(self.crowd_labels):
            for (label, worker) in zip(*lw_set):
              if label >= 0:
                #print label, worker
                truel = self.true_labels[index]
                if truel <0: continue

                if worker not in dic_conf:
                    dic_conf[worker] = [[0,0], [0,0]]
                dic_conf[worker][truel][label] += 1
            

        dic_ss = {}
        for (worker, conf_mat) in dic_conf.items():
            #if conf_mat[0][0] >= 5 and conf_mat[0][1] >= 5 and \
            #   conf_mat[1][0] >= 5 and conf_mat[1][1] >= 5:
            if get_all or ( (conf_mat[0][0] +  conf_mat[0][1] >= 5) and  \
               (conf_mat[1][0] + conf_mat[1][1] >= 5) ):
                  try:
                    sen = conf_mat[1][1] * 1.0 / (conf_mat[1][1] + conf_mat[1][0])
                  except ZeroDivisionError:
                    sen = None
                  try:
                    spe = conf_mat[0][0] * 1.0 / (conf_mat[0][0] + conf_mat[0][1])
                  except ZeroDivisionError:
                    spe = None
            else:
                sen = None
                spe = None
            dic_ss[worker] = (sen, spe, sum(conf_mat[0]) + sum(conf_mat[1]) )

        return dic_ss

    def count_wl(self):
        """
        Coutn worker labels
        """
        dic = {}
        for (index, lw_set) in enumerate(self.crowd_labels):
            for (label, worker) in zip(*lw_set):
              if label >= 0:
                  if worker not in dic: dic[worker] = 0
                  dic[worker] = dic[worker] + 1

        return dic

    def total_labels(self):
        """
        Return total number of labels
        """
        s = 0
        for (index, lw_set) in enumerate(self.crowd_labels):
            s += len(lw_set[0])

        return s

def sigmoid(x):
    return 1.0 / ( 1.0 + np.exp(-x))

class ss_model():
    """
    Model of Workers Sensitivity and Specificity and their correlation
    Actually "spe" is the False Positive Rate ( 1 - Specificity)
    """
    def __init__(self, labels_col, pmu = 2, pvar = 1, pcov = 0.1, nsam = 50):
        self.lc = labels_col
        self.model_def = ()

        # params
        self.mu = np.asarray([pmu, -pmu])
        self.mu = np.asarray([2, -3])
        #self.mu = np.zeros(2)
        self.C = np.asarray([[2,0.4],[0.4, 1.5]])

        self.mu2 = np.asarray([2, -5])
        self.C2 = np.asarray([[1, -0.05],[-0.05, 1]])

        self.C[1][1] = 2; self.C[0][0] = 1; self.C[0][1] = self.C[1][0] = 0.5
        self.N = self.lc.n
        self.nsam = nsam
        self.prior_class = np.ones(self.N) * 0.5

        self.build_model_def()


    def get_worker_params(self, worker):
        """
        """
        if worker.startswith('S'):
            return (self.mu, self.C)
        return (self.mu2, self.C2)


    def build_model_def(self):
        """
        build model definition
        from the labels collection
        """
        # params:


        # for each item: a var of true label, hidden if no gold available
        N = self.lc.n
        items = np.empty(N, dtype=object)

        for i in range(N):
            if self.lc.true_labels[i] == None:
                items[i] = pymc.Bernoulli("Item_%i" % i, self.prior_class[i])
            else:
                l = self.lc.true_labels[i]
                items[i] = pymc.Bernoulli("Item_%i" % i, self.prior_class[i], value = (l==1), observed = True)


        # for each worker: 2 hidden var: sen and spe
        worker_dic = {}
        for (index, lw_set) in enumerate( self.lc.crowd_labels):
            for (label, worker) in zip(*lw_set):
                if not worker_dic.has_key(worker):
                    # create distribution for SS of the worker
                    ss = np.empty(2, dtype = object)
                    (mu, C) = self.get_worker_params(worker)
                    ss = pymc.MvNormalCov("ss_%s" % worker, mu, C, value = mu)

                    worker_dic[worker] = ss

        # for each crowd label: a visible var
        wlabels = []
        list_prob = []
        list_p = []
        for (index, lw_set) in enumerate( self.lc.crowd_labels):
            for (label, worker) in zip(*lw_set):
                var_name = "x_" + worker + "_" + str(index)

                ss = worker_dic[worker]
                # prob of 1 given the true label
                prob_1 =     pymc.InvLogit("prob_1_%s_%i" %(worker, index),  ss[0])
                prob_0 =     pymc.InvLogit("prob_0_%s_%i" %(worker, index),  ss[1])

                #p = pymc.Lambda("prob_" + var_name, lambda trul = items[index]:
                #     prob_1 if trul == 1 else prob_0 )


                #def prob_eval(item_index, prob_0, prob_1):
                #    if item_index == 0:
                #        return prob_0
                #    else:
                #        return prob_1

                #p = pymc.Deterministic(eval = prob_eval,
                #  name = "prob_" + var_name,
                #  parents = {"Item_" + str(index): items[index], "prob_1_%s" %worker: prob_1, "prob_0_%s" %worker: prob_0},
                #  dtype=float)

                @pymc.deterministic(plot=False, name = "p_" + var_name)
                def p(item_index = items[index], p1 = prob_1, p0 = prob_0):
                    if item_index == 0: return p0
                    else: return p1

                list_prob.append(prob_1)
                list_prob.append(prob_0)
                list_p.append(p)
                wlabels.append( pymc.Bernoulli(var_name, p, value = label, observed = True) )
                #pymc.InvLogit()

        self.model_def = (items, worker_dic, wlabels, list_prob, list_p, self.mu, self.C, self.prior_class)


    def get_sample(self):
        self.S = pymc.MCMC(self.model_def)
        self.S.sample(self.nsam, progress_bar = False)

    def maximize_params(self):
        a = np.empty((0,2))
        for v in self.S.variables:
            if str(v).startswith("ss"):
                b = self.S.trace(str(v))[:] # samples
                a = np.vstack((a,b))
            elif str(v).startswith("Item") and not v.observed:
                b = self.S.trace(str(v))[:] # samples
                i = int(str(v)[5:])
                #self.prior_class[i] = sum(b) * 1.0 / len(b)

        self.mu = np.mean(a, 0)
        self.C = np.cov(a.T)


    def em(self, n_iters = 2):
        print self.mu
        print self.C

        for it in range(n_iters):
            # E-step:
            self.get_sample()
            #M-step
            self.maximize_params()
            print self.mu
            print self.C
            self.build_model_def()



    def workers_ss(self):
        self.worker_dic = {}
        self.dic_ss = {}
        for v in self.S.variables:
            if str(v).startswith("ss_"):
                worker_id = str(v)[3:]
                b = self.S.trace(str(v))[:] # samples
                mu = np.mean(b, 0)
                C = np.cov(b.T)
                self.worker_dic[worker_id] = (mu, C)
                self.dic_ss[worker_id] = (sigmoid(mu[0]), 1 - sigmoid(mu[1]) )


    def infer_true_l(self):
        self.get_sample()
        self.prob = self.N * [-1]
        for v in self.S.variables:
            if str(v).startswith("Item_") and not v.observed:
                b = self.S.trace(str(v))[:] # samples
                i = int(str(v)[5:])
                self.prob[i] = sum(b) * 1.0 / len(b)

    def test():
        lc = crowd_model.labels_collection(start.turk_data_id[:10], 10*[None])
        ss = crowd_model.ss_model(lc)


#global_psen = (1,1)
#global_pspe = (1,1)
#global_pfpr = (1,1)

global_psen = (4,1)
global_pspe = (4,1)
global_pfpr = (1,4)

#global_psen = (7,3)
#global_pspe = (9,1)
#global_pfpr = (1,9)


class tc_model():
    """
    Two-coin Model
    """

    def __init__(self, lc, prior_sen = (7,3), prior_spe = (9,1), ep = 0.001 , balance_weight = 1.0, p1 = 1.0, p2 = 1.0):
        self.lc = lc
        # dic from worker -> (sen, spe)
        self.dic_worker_ss = {}

        self.prior_sen = prior_sen
        self.prior_spe = prior_spe

        self.prior_sen = global_psen
        self.prior_spe = global_pspe

        self.N = self.lc.n
        self.balance_weight = balance_weight

        # init hidden vars
        self.prob = np.zeros(self.N)
        self.theta = 0.5
        self.p1 = p1; self.p2 = p2
        self.init_prob(ep = ep)

    def init_prob(self, id_range = None, ep = 0.001):
        if id_range == None:
            labels_set = self.lc.crowd_labels
            left = 0; right = self.lc.n
        else:
            left = id_range[0]; right = id_range[1]
            labels_set = self.lc.crowd_labels[left:right]


        for (index, lw_set) in zip (range(left, right), labels_set):
            if self.lc.true_labels[index] == None:
                temp = sum(lw_set[0]) * 1.0 / len(lw_set[0])
                temp = max(min(temp,1-ep), ep)
                self.prob[index] = temp
            else:
                self.prob[index] = self.lc.true_labels[index]

        self.theta = (self.p1 + np.sum(self.prob) ) * 1.0 / ( self.p1 + self.p2 + len(self.prob) )



    def e_step(self, id_range = None):
        """
        Given params ( sen/spe)
        Estimate a distribution over hidden vars (true labels)
        prob[truel|crowd ans] \propto prob[crowd ans| truel] prob[truel]

        id_range: range of indices in the crowd_labels to be considered
        """

        if id_range == None:
            labels_set = self.lc.crowd_labels
            left = 0; right = self.lc.n
        else:
            left = id_range[0]; right = id_range[1]
            labels_set = self.lc.crowd_labels[left:right]

        for (index, lw_set) in zip (range(left, right), labels_set):
          if self.lc.true_labels[index] == None:
            prob_1 = self.theta
            prob_0 = (1-self.theta)
            for (label, worker) in zip(*lw_set):
              if label >= 0:
                (sen, spe) = self.dic_worker_ss[worker]
                if label == 0:
                    prob_1 *= (1 - sen); prob_0 *= spe
                else:
                    prob_1 *= sen; prob_0 *= (1-spe)

            self.prob[index] = prob_1*1.0 / (prob_0 + prob_1)
          else:
            self.prob[index] = self.lc.true_labels[index]

        #print "prob: ", self.prob

    def interpolate(self, old, new, w):
        if old == None: return new
        return old * (1-w) + new * w

    def item_w(self, index):
        #average_p = np.sum(self.prob) * 1.0 / len(self.prob)
        if self.prob[index] > 0.5:
            return self.balance_weight
        else:
            return 1.0

    def m_step(self, id_range = None, w = 1.0):
        """
        Given The distribution over hidden vars
        Maximize params (sen, spe) of each worker
        """

        if id_range == None:
            labels_set = self.lc.crowd_labels
            left = 0; right = self.lc.n
        else:
            left = id_range[0]; right = id_range[1]
            labels_set = self.lc.crowd_labels[left:right]

        # dic: worker -> soft count confusion [true][ans]
        dic_count = {}
        for (index, lw_set) in zip (range(left, right), labels_set):
            for (label, worker) in zip(*lw_set):
              if label >= 0:
                if worker not in dic_count: dic_count[worker] = [[0,0],[0,0]]
                item_w = self.item_w(index)
                dic_count[worker][1][label] += item_w*self.prob[index]
                dic_count[worker][0][label] += item_w*(1-self.prob[index])

        #print "dic_count",  dic_count

        psen = self.prior_sen; pspe = self.prior_spe
        total_change = 0
        for (worker, mat) in dic_count.items():
            sen = (mat[1][1] + psen[0]) * 1.0 / (mat[1][1] + mat[1][0] + psen[0] + psen[1])
            spe = (mat[0][0] + pspe[0]) * 1.0 / (mat[0][0] + mat[0][1] + pspe[0] + pspe[1])
            if worker not in self.dic_worker_ss:
                self.dic_worker_ss[worker] = [1.0, 1.0]
            old_sen = self.dic_worker_ss[worker][0]; old_spe = self.dic_worker_ss[worker][1];
            self.dic_worker_ss[worker][0] = self.interpolate(self.dic_worker_ss[worker][0], sen, w)
            self.dic_worker_ss[worker][1] = self.interpolate(self.dic_worker_ss[worker][1], spe, w)
            total_change += np.abs(old_sen - self.dic_worker_ss[worker][0]) + np.abs(old_spe - self.dic_worker_ss[worker][1])

        self.theta = (self.p1 + np.sum(self.prob) ) * 1.0 / ( self.p1 + self.p2 + len(self.prob) )
        avg_change = total_change * 1.0 / ( 2 * len(self.dic_worker_ss) )

        return avg_change
        #print "dic_ss", self.dic_worker_ss

    def em(self, num_it = 20):
        """
        Learn Params ( confusion mat of each worker) by EM
        """
        self.m_step()
        for it in range(num_it):
            self.e_step()
            avg_change = self.m_step()
            if avg_change < 0.0001: break

    def online_em(self, new_labels, w = 0.1, num_it = 3):
        n = self.lc.n
        l = len(new_labels)
        self.lc.add_labels(new_labels, l * [None])
        self.N = self.lc.n
        self.prob = np.hstack((self.prob, np.zeros(l)))
        self.init_prob(id_range = (n, self.lc.n))
        #self.em()
        #return

        self.m_step(id_range = (n, self.lc.n), w = w)

        for it in range(num_it):
            self.e_step(id_range = (n, self.lc.n))
            self.m_step(id_range = (n, self.lc.n), w = w)

def fix_r(x):
    """
    """
    ep = 0.001
    return min(max(x, ep), 1 - ep)

def Logit(x):
    x = fix_r(x)
    return np.log(x) - np.log(1-x)


def reduce_cov(c):
    #v = (c[0][0] + c[1][1])/2.0
    #c[0][0] = v; c[1][1] = v
    c[0][1] = 0; c[1][0] = 0
    return c

class vss_model:
    """
    Params: mu_1, C_1, mu_2, C_2,
            theta

    Hidden vars: u_1, v_1, u_2, v_2
    U = Sensitivity
    V = False Positive Rate
    Z: true label
    """

    def __init__(self, labels_col, balance_weight = 1.0, full_cov = True):
        """
        """
        self.full_cov = full_cov
        self.lc = labels_col

        # variational distribution q
        self.qz = np.zeros(self.lc.n) # prob of item i being 1
        self.quv = {}
        self.maj_lab = np.zeros(self.lc.n)
        self.maj_ss = {}

        self.theta = None
        self.mu1 = None; self.mu2 = None
        self.C1 = None; self.C2 = None
        self.balance_weight = balance_weight

        self.prior_sen = global_psen
        self.prior_fpr = global_pfpr

        self.init_prob()
        self.get_first_params()

    def get_var(self, a, b):
        """
        Approximate with the variance of a Beta distribution
        """
        m = a * 1.0 / (a + b)

        #if m < 0.1: m = 0.1
        #if m > 0.9: m = 0.9
        #m = 0.5

        #a+= 1; b += 1
        bvar =  a * b * 1.0/ ( (a+b)*(a+b)*(a+b+1) )
        bstd = np.sqrt(bvar)

        # change to Logit scale
        Lstd = ( Logit(m+ bstd) - Logit(m - bstd)) / 2.0
        #print a, b, bstd, m, Lstd, Lstd*Lstd
        var = Lstd * Lstd
        #if var > 1: var = 1
        return var

    def init_prob(self, id_range = None):
        """
        given a new batch of items:
        Set new params and q to empirical
        """
        if id_range == None:
            labels_set = self.lc.crowd_labels
            left = 0; right = self.lc.n
        else:
            left = id_range[0]; right = id_range[1]
            labels_set = self.lc.crowd_labels[left:right]

        # get majority vote label
        for (index, lw_set) in zip (range(left, right), labels_set):
            temp = 0
            nl = 0
            for lab in lw_set[0]:
                if lab >= 0:
                    temp += lab
                    nl += 1
            if nl == 0:
                #print index, lw_set
                self.maj_lab[index] = 0
                self.qz[index] = 0.001
            else:
                self.maj_lab[index] = 1 if (temp*1.0/nl) >= 0.5 else 0
                self.qz[index] = fix_r(temp*1.0/nl)

        if self.theta == None and len(self.maj_lab) > 0:
            self.theta = ( 1 + np.sum(self.qz) )  * 1.0 / ( 2 + self.lc.n )

        # pretend majority vote label is correct
        # find the sen/fpr of each worker
        dic_conf = {} # dic of confusion matrix
        for (index, lw_set) in zip (range(left, right), labels_set):
            for (label, worker) in zip(*lw_set):
              if label >= 0:
                truel = int(self.maj_lab[index])

                if worker not in dic_conf:
                    dic_conf[worker] = [[0,0], [0,0]]
                dic_conf[worker][1][label] += self.qz[index]
                dic_conf[worker][0][label] += (1 - self.qz[index])
                #dic_conf[worker][trul][label] += 1


        # build a dic: worker -> SS using majority voting
        psen = self.prior_sen; pfpr = self.prior_fpr
        for (worker, conf_mat) in dic_conf.items():
            try:
                sen = ( psen[0] + conf_mat[1][1] * 1.0) / (conf_mat[1][1] + conf_mat[1][0] + psen[0] + psen[1])
                varsen = self.get_var(psen[0] + conf_mat[1][1], psen[1] + conf_mat[1][0])
            except ZeroDivisionError:
                print "Zero Div"
                sen = 0.7
                varsen = 0.1
            try:
                fpr = ( pfpr[0] + conf_mat[0][1] * 1.0 ) / (conf_mat[0][0] + conf_mat[0][1] + pfpr[0] + pfpr[1])
                varfpr = self.get_var(pfpr[0] + conf_mat[0][1], pfpr[1] + conf_mat[0][0])
            except ZeroDivisionError:
                print "Zero Div"
                fpr = 0.05
                varfpr = 0.1
            if worker not in self.maj_ss:
                #print worker, sen, fpr
                self.maj_ss[worker] = (sen, fpr, varsen, varfpr)



        # set quv
        for (worker, (sen, fpr, varsen, varfpr)) in self.maj_ss.items():
            if not sen == None:
                if worker.startswith('S'):
                    if worker not in self.quv:
                        #self.quv[worker] = [Logit(sen), self.C1[0][0], Logit(fpr), self.C1[1][1]]
                        self.quv[worker] = [Logit(sen), varsen, Logit(fpr), varfpr]
                else:
                    if worker not in self.quv:
                        #print self.C2
                        #self.quv[worker] = [Logit(sen), self.C2[0][0], Logit(fpr), self.C2[1][1]]
                        self.quv[worker] = [Logit(sen), varsen, Logit(fpr), varfpr]

        self.dic_wl = {}

        # create dic: worker -> [(item, label)] ONLY FOR CURRENT ITEMS
        for (index, lw_set) in zip (range(left, right), labels_set):
            for (label, worker) in zip(*lw_set):
              if label >= 0:
                  if worker not in self.dic_wl: self.dic_wl[worker] = []
                  self.dic_wl[worker].append((index, label))

    def interpolate(self, old, new, w):
        if old == None: return new
        return old * (1-w) + new * w



    def update_qz(self, left = None, right = None):
        
        if left == None:
            left = 0
            right = self.lc.n
            
        for index in range(left, right):
                #if index % 100 == 0:
                #    print index
                #    sys.stdout.flush()

                qz_1 = np.log(self.theta)
                qz_0 = np.log(1 - self.theta)
                for (label, worker) in zip(*self.lc.crowd_labels[index]):
                    if label >=0 and worker in self.quv:
                      qz_1 += expectation_z(self.quv[worker][0], self.quv[worker][1], label)
                      qz_0 += expectation_z(self.quv[worker][2], self.quv[worker][3], label)

                qz_1 = np.exp(qz_1)
                qz_0 = np.exp(qz_0)

                temp = qz_1 * 1.0 / (qz_0 + qz_1)
                if not math.isnan(temp):
                    self.total_changes += np.abs(self.qz[index] - temp)
                    self.qz[index] = temp

        
        
    def update_quv_mesh(self, w = 1.0):
    
        for (worker, list_il) in self.dic_wl.items():
              if worker in self.quv:
                #print worker; sys.stdout.flush()
                mean = self.quv[worker][0]
                v = self.quv[worker][1]
                a = []
                for x in get_mesh(mean, v):
                    val = self.eval_pdf(worker, 'sen', x, list_il)
                    a.append( (x, val) )

                (m, var) = self.fit_normal(a)
                if math.isnan(m)==False and math.isnan(var)==False:
                    new_0 = self.interpolate(self.quv[worker][0], m, w)
                    new_1 = self.interpolate(self.quv[worker][1], var, w)
                    self.total_changes += np.abs(self.quv[worker][0] - new_0) + np.abs(self.quv[worker][1] - new_1)
                    self.quv[worker][0] = new_0
                    self.quv[worker][1] = new_1

                mean = self.quv[worker][2]
                v = self.quv[worker][3]
                a = []
                for x in get_mesh(mean, v):
                    val = self.eval_pdf(worker, 'fpr', x, list_il)
                    a.append( (x, val) )

                (m, var) = self.fit_normal(a)
                if math.isnan(m)==False and math.isnan(var)==False:
                    new_2 = self.interpolate(self.quv[worker][2], m, w)
                    new_3 = self.interpolate(self.quv[worker][3], var, w)
                    self.total_changes += np.abs(self.quv[worker][2] - new_2) + np.abs(self.quv[worker][3] - new_3)
                    self.quv[worker][2] = new_2
                    self.quv[worker][3] = new_3
                    
        
       
        
    
    
    def finite_diff(self, worker, mu, C, W, eps = 0.000000001):
        
        Wx = W + np.asarray([eps, 0])
        Dx = ( eval_F([self], worker, mu, C, Wx) - eval_F([self], worker, mu, C, W) ) / eps
        
        Wy = W + np.asarray([0, eps])
        Dy = ( eval_F([self], worker, mu, C, Wy) - eval_F([self], worker, mu, C, W) ) / eps
        
        return [Dx, Dy]
        
        
        
    def check_der(self):
        for (worker, list_il) in self.dic_wl.items():
            mu = self.get_mu(worker)
            C = self.get_c(worker)
            W = np.asarray([2,-1])
            
            print worker
            print self.finite_diff(worker, mu, C, W)
            print eval_Grad_F([self], worker, mu, C, W)
            print "------------------------"
        
        
    def update(self, old, new):
        
        self.total_changes += np.abs(old - new)
        return new
        
    def update_quv_lap(self):
        """
        Update Quv using Laplace Approximation 
        to Variational Inference
        """
        for (worker, list_il) in self.dic_wl.items():
            mu = self.get_mu(worker)
            C = self.get_c(worker)
            
            f = lambda W: -eval_F([self], worker, mu, C, W)
            fp = lambda W: -eval_Grad_F([self], worker, mu, C, W)
            x0 = np.asarray(mu)
            
            res = scipy.optimize.minimize(f, x0, method='BFGS', jac=fp)

            #self.quv[worker][0] = self.update(self.quv[worker][0], res.x[0])
            #self.quv[worker][2] = self.update(self.quv[worker][2], res.x[1])
            self.quv[worker][0] = res.x[0]
            self.quv[worker][2] = res.x[1]
            
            hes = - np.linalg.inv ( eval_Grad2_F([self], worker, mu, C, res.x) )
            #hes = np.linalg.inv(res.hess_inv)
            #hes1 = res.hess_inv
            #print worker
            #print res.message
            #print hes1
            #print hes
            #print "-----------------------------"
            
            #self.quv[worker][1] = self.update(self.quv[worker][1], hes[0][0])
            #self.quv[worker][3] = self.update(self.quv[worker][3], hes[1][1])
            self.quv[worker][1] = max(min(hes[0][0], 100), 0.00000001)
            self.quv[worker][3] = max(min(hes[1][1], 100), 0.00000001)
            
            
            
        
        
    def e_step(self, num_it = 3, w = 1.0, id_range = None, use_lap = True):
        """
        Do variational inference to find the distribution over hidden vars
        P(u1, v1, u2, v2)
        """

        if id_range == None:
            labels_set = self.lc.crowd_labels
            left = 0; right = self.lc.n
        else:
            left = id_range[0]; right = id_range[1]
            labels_set = self.lc.crowd_labels[left:right]


        for it in range(num_it):
            #print 'iteration ', it;
            sys.stdout.flush()
            self.total_changes = 0

            self.update_qz(left, right)
            if use_lap: self.update_quv_lap()
            else: self.update_quv_mesh()
            
            #average_changes = self.total_changes * 1.0 / (len(self.qz) + len(self.quv)*4)
            #if average_changes < 0.01: break

        self.get_dic_ss()

    def fit_normal(self, a):
        """
        fit a normal distribution to [(x, p)]
        where p is in log scale
        """
        # normalize p:
        #print a
        list_p = []
        for (x, p) in a:  list_p.append(p)
        ps = scipy.misc.logsumexp(list_p)

        s  = 0
        ss = 0
        for (x, p) in a:
            s  += x * np.exp(p - ps)
            ss += x*x * np.exp(p - ps)

        var = ss - s*s
        ep = 1E-300
        if var < ep: var = ep
        return (s, var)

    def fit_normal2(self, a):
        """
        fit a normal distribution to [(x, p)]
        """
        # normalize p:
        #print a
        ps = 0
        for (x, p) in a:
            if p > 0:
                ps += p

        mu  = 0
        var = 0
        for (x, p) in a:
            if p > 0:
                mu  += x * (p * 1.0 / ps)


        for (x, p) in a:
            if p > 0:
                var += (x-mu)*(x-mu) * (p * 1.0 / ps)

        ep = 1E-300;
        if var <= ep : var = ep

        return (mu, var)
    def get_mu(self, worker):
        if worker.startswith('S'): return self.mu1
        else: return self.mu2

    def get_c(self, worker):
        if worker.startswith('S'): return self.C1
        else: return self.C2

    def Ber(self, p, l):

        if l == 1: return max(p, 1e-100)
        else: return  max(1 - p, 1e-100)

    def get_item_w(self, item):
        #if self.maj_lab[item] > 0.5:
        if self.qz[item] > 0.5:
            return self.balance_weight
        else:
            return 1.0
        

    def eval_pdf(self, worker, var, x, list_il):
        """
        Eval *LOG* of pdf of new qu or qv
        """
        if var == 'sen':
            res = expectation_binorm('v', self.quv[worker][2], self.quv[worker][3], \
                x, self.get_mu(worker), self.get_c(worker))
            #print res
            for (item, label) in list_il:
                item_w = self.get_item_w(item)
                res += item_w * self.qz[item] * np.log( self.Ber(S(x), label))
        else:
            res = expectation_binorm('u', self.quv[worker][0], self.quv[worker][1], \
                x, self.get_mu(worker), self.get_c(worker))
            for (item, label) in list_il:
                item_w = self.get_item_w(item)
                res += item_w * (1 - self.qz[item]) * np.log( self.Ber(S(x), label))

        return res



    def estimate_params(self, ess):
        """
        Estimate  Nomal param,
        given expecteed sufficient stats
        """
        n = len(ess)
        mu = np.asarray([0,0])
        for (m, c) in ess:
            mu = mu + m
        mu = mu * 1.0 / n

        C = np.asarray([[0,0],[0,0]])
        for (m, c) in ess:
            C = C + c

        C = C * 1.0 / n
        
        C = C - np.outer(mu, mu)

        if not self.full_cov:
            C = reduce_cov(C)

        return (mu, C)

    def m_step(self, w = 1.0, id_range = None, nsam = 1000):
        """
        Maximize params ( mu, C and theta)
        given the variational distribuiton q
        """
        # theta
        self.theta = np.sum(self.qz) * 1.0 / len(self.qz)

        # mu and C
        ess1 = []; ess2 = []
        for (worker, (sen, varsen, fpr, varfpr)) in self.quv.items():
            m = np.asarray([sen, fpr])
            c = np.asarray([[sen*sen + varsen, sen*fpr],[sen*fpr, fpr*fpr + varfpr]])
            #c = np.asarray([[sen*sen , sen*fpr],[sen*fpr, fpr*fpr]])
            if worker.startswith('S'):
                ess1.append((m, c))
            else:
                ess2.append((m, c))


        old_mu1 = self.mu1; old_mu2 = self.mu2
        old_C1  = self.C1 ; old_C2  = self.C2
        change = 0

        if len(ess1) > 0:
            (self.mu1, self.C1) = self.estimate_params(ess1)
            change += ( np.sum(np.abs(old_mu1 - self.mu1)) + np.sum(np.abs(old_C1 - self.C1)) ) / 6.0
        if len(ess2) > 0:
            (self.mu2, self.C2) = self.estimate_params(ess2)
            change += ( np.sum(np.abs(old_mu2 - self.mu2)) + np.sum(np.abs(old_C2 - self.C2)) ) / 6.0

        return change

    def get_dic_ss(self):
        """
        """
        self.dic_ss = {}
        for (worker, (sen, varsen, fpr, varfpr) ) in self.quv.items():
            self.dic_ss[worker] = (S(sen), 1 - S(fpr))


    def get_first_params(self):
        self.mu1 = np.zeros(2); self.mu2 = np.zeros(2);
        self.C1 = np.zeros((2,2)); self.C2 = np.zeros((2,2))
        self.m_step()

    def em(self, num_it = 5, use_lap = True):

        for it in range(num_it):
            if it == 0:
                self.e_step(3, use_lap = use_lap)
            else:
                self.e_step(use_lap = use_lap)
            avg_change = self.m_step()
            #if avg_change < 0.0001: break

    def online_em(self, new_labels, w = 0.1, num_it = 3, no_train = False):
        if len(new_labels) == 0: return

        n = self.lc.n
        l = len(new_labels)
        self.lc.add_labels(new_labels, l * [None])
        self.N = self.lc.n
        self.qz = np.hstack((self.qz, np.zeros(l)))
        self.maj_lab = np.hstack((self.maj_lab, np.zeros(l)))

        self.init_prob(id_range = (n, self.lc.n))

        if no_train: return

        self.m_step(id_range = (n, self.lc.n), w = w)

        for it in range(num_it):
            self.e_step(id_range = (n, self.lc.n), w = w)
            self.m_step(id_range = (n, self.lc.n), w = w)


def Ber(p, l):
        if l == 1: return max(p, 1e-100)
        else: return  max(1 - p, 1e-100)


def eval_F(list_data, worker, mu, C, W):
        """
        F = log N(T, mu, C) + SUM q() log Ber(L_ij | T)
        list_il can be from different datasets:
        list_il = list: each elem is a list from a dataset
        """    
            
        res = scipy.stats.multivariate_normal.logpdf(W, mean = mu, cov = C, allow_singular = True)
        
        for dt, data in enumerate(list_data):
            # dataset dt: sen = W[2*dt], fpr = W[2*dt+1]
            sen, fpr = W[2*dt], W[2*dt+1]
            if worker in data.dic_wl:
             for (item, label) in data.dic_wl[worker]:
                # prob of being 1
                qz = data.qz[item]
                res += qz     * np.log( Ber(S(sen), label) )
                #res += qz     * ( label*np.log(S(sen)) + (1-label)*np.log(S(sen))  )
                res += (1-qz) * np.log( Ber(S(fpr), label) )
                
                
        return res
        
        
def LogSp(x):
    return 1 / (1 + np.exp(x))

def Log1MSp(x):
    return -np.exp(x) / (1 + np.exp(x))

def LogSpp(x):
    return -np.exp(x) / pow(1 + np.exp(x), 2)
        
def eval_Grad_F(list_data, worker, mu, C, W):
        """
        Grad F = p(W) * C^-1 * (W - mu) + SUM q() ...
        """
        C = np.asarray(C)
        mu = np.asarray(mu)
        W = np.asarray(W)
        
        C_inv = np.linalg.inv(C)
        
        #res1 = - np.log(scipy.stats.multivariate_normal.pdf(W, mean = mu, cov = C, allow_singular = True)) \
        #    * C_inv.dot(W - mu)
            
        res1 = -C_inv.dot(W - mu)
        
        res2 = np.zeros_like(res1)
        
        for dt, data in enumerate(list_data):
            # dataset dt: sen = W[2*dt], fpr = W[2*dt+1]
            sen, fpr = W[2*dt], W[2*dt+1]
            if worker in data.dic_wl:
             for (item, label) in data.dic_wl[worker]:
                # prob of being 1
                qz = data.qz[item]
                res2[2*dt]   += qz     * ( label * LogSp(sen) + (1-label) * Log1MSp(sen) )
                res2[2*dt+1] += (1-qz) * ( label * LogSp(fpr) + (1-label) * Log1MSp(fpr) )
                
        return res1 + res2
        

def eval_Grad2_F(list_data, worker, mu, C, W):
        """
        Grad2 F = 
        """
        C = np.asarray(C)
        n = C.shape[0]
        mu = np.asarray(mu).reshape((n,1))
        W = np.asarray(W).reshape((n,1))
        #print  mu
        #print W
        
        
        C_inv = np.linalg.inv(C)
        
        D = W - mu
        E = C_inv.dot(D).dot( D.T ).dot(C_inv) - C_inv
        #E = C_inv.dot(D)
        #print E
        
        res1 = E
        
        res2 = np.zeros_like(res1)
        for dt, data in enumerate(list_data):
            # dataset dt: sen = W[2*dt], fpr = W[2*dt+1]
            sen, fpr = W[2*dt], W[2*dt+1]
            if worker in data.dic_wl:
             for (item, label) in data.dic_wl[worker]:
                # prob of being 1
                qz = data.qz[item]
                # LogSpp = Log1MSpp
                res2[2*dt][2*dt]     += qz     * ( label * LogSpp(sen) + (1-label) * LogSpp(sen) )
                res2[2*dt+1][2*dt+1] += (1-qz) * ( label * LogSpp(fpr) + (1-label) * LogSpp(fpr) )
                
        return res1 + res2



def  get_mesh(mean, v):
    """
    Provide mesh points
    for Normal estimation
    """
    std = pow(v, 0.5)
    #a = np.linspace(mean - 5*v, mean + 5*v, 50)
    b = np.linspace(mean - 3*std, mean + 3*std, 25)
    #c = np.linspace(mean - 1, mean + 1, 50)

    #return a
    #return np.hstack((a,b,c))
    return b

def S(x):
    return sigmoid(x)


def expectation_z(mu, var, L, w = 3):
    """
    USE LAPLACE approx
    Evaluate the expectation of log[ S(u)^L (1-S(u))^(1-L) ]
    when u ~ Normal(mu, var)
    """
    if L == 1:
        f = lambda u: np.log(S(u))
        fpp = lambda x: -np.exp(x) / pow(1 + np.exp(x), 2)
        return f(mu) + 0.5* fpp(mu)*var
    else:
        f = lambda u: np.log(1 - S(u))
        fpp = lambda x: -np.exp(x) / pow(1 + np.exp(x), 2)
        return f(mu) + 0.5* fpp(mu)*var

    # need: E[u~N](f)
    return scipy.integrate.quad(f, mu - w*std, mu + w*std)[0]


def expectation_z_quad(mu, var, L, w = 3):
    """
    USE QUAD (NUMERICAL INTEGRATION)
    Evaluate the expectation of log[ S(u)^L (1-S(u))^(1-L) ]
    when u ~ Normal(mu, var)
    """
    #U = np.random.normal(mu, np.sqrt(var), num)
    if L == 1:
        f = lambda u: scipy.stats.norm.pdf(u, loc = mu, scale = np.sqrt(var) ) * np.log(S(u))
    else:
        f = lambda u: scipy.stats.norm.pdf(u, loc = mu, scale = np.sqrt(var) ) * (np.log(1 - S(u)))

    #return f
    std = np.sqrt(var)
    #return scipy.integrate.quad(f, mu - w*std, mu + w*std, epsabs = 1.0e-4, epsrel = 1.0e-4, limit = 25)[0]
    return scipy.integrate.quad(f, mu - w*std, mu + w*std)[0]


def expectation_binorm(rv, mu, var, x, M, C, w = 3):
    """
    Evaluate the expectation of log Norm(uv| M, C)
         x = u, v ~ Norm(mu, var) rv == 'v'
         x = v, u ~ Norm(mu, var) rv == 'u'
    """
    #print rv, mu, var, x, M, C
    if rv == 'v':
      f = lambda v: scipy.stats.norm.pdf(v, loc = mu, scale = np.sqrt(var) ) * \
        np.log(scipy.stats.multivariate_normal.pdf([x, v], mean = M, cov = C, allow_singular = True))
    else:
      f = lambda u: scipy.stats.norm.pdf(u, loc = mu, scale = np.sqrt(var) ) * \
        np.log(scipy.stats.multivariate_normal.pdf([u, x], mean = M, cov = C, allow_singular = True))

    #return f
    #print f(mu)
    std = np.sqrt(var)
    return scipy.integrate.quad(f, mu - w*std, mu + w*std)[0]


class multitask():
    """
    """
    def __init__(self, datasets, inter_cor = 0.0):
        """
        inter_cor = correlation of a worker performance in different datasets
        datasets = a list of (two) vss_model
        """
        self.datasets = datasets
        
        self.workers = []
        for data in self.datasets:
            for w in data.dic_wl.keys():
                if w not in self.workers:
                    self.workers.append(w)
        
        # init global mu, C
        # W ~ N(mu, C) 
        self.n = len(datasets)
        n = self.n
        self.mu = np.zeros(n*2)
        self.C = np.zeros((n*2, n*2))
        for data in self.datasets:
            data.em(1)
            
        self.inter_cor = inter_cor
        self.set_params()
        
        
    def set_params(self):
        """
        Copy params of each dataset to global
        """
    
        for dt, data in enumerate(self.datasets):
            
            self.mu[2*dt]   = data.mu2[0]
            self.mu[2*dt+1] = data.mu2[1]
            
            self.C[2*dt: 2*dt+2, 2*dt: 2*dt+2] = data.C2
        
        # set correlation of a worker performance
        n = self.n
        for dt1 in range(n):
            for dt2 in range(n):
              if dt1 != dt2:
                self.C[dt1*2, dt2*2] = self.C[dt2*2, dt1*2] = self.inter_cor
                self.C[dt1*2+1, dt2*2+1] = self.C[dt2*2+1, dt1*2+1] = self.inter_cor
        
        
        
    def e_step(self, num_it = 3, estim_unseen_worker= False):
        """
        estim_unseen_worker = last step before evaluation
        estimate unseen workers
        """
        for it in range(num_it):
          # update Z
          for data in self.datasets:
            data.update_qz()
            
          # update W = (sen1, fpr1, sen2, fpr2, ... ) for each worker
        
          for worker in self.workers:
            f = lambda W: -eval_F(self.datasets, worker, self.mu, self.C, W)
            fp = lambda W: -eval_Grad_F(self.datasets, worker, self.mu, self.C, W)
            x0 = np.asarray(self.mu)
            
            res = scipy.optimize.minimize(f, x0, method='BFGS', jac=fp)
            #hes = -eval_Grad2_F(self.datasets, worker, self.mu, self.C, res.x)
            
            hes = - np.linalg.inv ( eval_Grad2_F(self.datasets, worker, self.mu, self.C, res.x) )
            for (dt, data) in enumerate(self.datasets):
              if worker in data.quv or estim_unseen_worker:
                data.quv[worker] = [1,1,-1,1]
                    
                data.quv[worker][0] = res.x[dt*2]
                data.quv[worker][2] = res.x[dt*2+1]
            
                
                #hes = np.linalg.inv(res.hess_inv)
                #print worker
                #print res.message
                #print hes1
                #print hes
                #print "-----------------------------"
            
                data.quv[worker][1] = max(min(hes[dt*2][dt*2], 100), 0.00000001)
                data.quv[worker][3] = max(min(hes[dt*2+1][dt*2+1], 100), 0.00000001)
        

    def prepare_eval(self):
        self.e_step(1, True)

    def m_step(self, learn_cor = True):
        for data in self.datasets:
            data.m_step()
            
        self.set_params()
        
        # learn intere_cor:
        if learn_cor:
         for (dt1, data1) in enumerate(self.datasets):
            for (dt2, data2) in enumerate(self.datasets):
              if dt1 < dt2:
                C0 = 0; C2 = 0
                m0_1 = 0; m0_2 = 0; m2_1 = 0; m2_2 = 0;
                m = 0
                X = np.empty((0,4))
                for worker in self.workers:
                    if worker in data1.quv and worker in data2.quv:
                        v = [data1.quv[worker][0], data1.quv[worker][2], data2.quv[worker][0], data2.quv[worker][2]]
                        X = np.vstack((X, v))
                        #m += 1
                        #C0 = C0 + data1.quv[worker][0] * data2.quv[worker][0]
                        #C2 = C2 + data1.quv[worker][2] * data2.quv[worker][2]
                        
                        #m0_1 += data1.quv[worker][0]
                        #m0_2 += data2.quv[worker][0]
                        
                        #m2_1 += data1.quv[worker][2]
                        #m2_2 += data2.quv[worker][2]
                    
                #m0_1 = m0_1*1.0 / m; m0_2 = m0_2*1.0 / m
                #m2_1 = m2_1*1.0 / m; m2_2 = m2_2*1.0 / m
                
                #C0 = C0*1.0/m
                #C2 = C2*1.0/m
                
                #print m, C2, m2
                #self.C[dt1*2, dt2*2] = self.C[dt2*2, dt1*2] = min( max( C0 - m0_1*m0_2, -5000), 5000)
                #self.C[dt1*2+1, dt2*2+1] = self.C[dt2*2+1, dt1*2+1] = min( max( C2 - m2_1*m2_2, -5000), 5000)
                
                C = np.cov(X.T, bias = 1)
                self.C[dt1*2, dt2*2] = self.C[dt2*2, dt1*2]         = C[0][2]
                self.C[dt1*2+1, dt2*2+1] = self.C[dt2*2+1, dt1*2+1] = C[1][3]
                
                self.C[dt1*2, dt2*2+1] = self.C[dt2*2+1, dt1*2]  = C[0][3]
                self.C[dt1*2+1, dt2*2] = self.C[dt2*2, dt1*2+1]  = C[1][2]
                #print "self.C = "
                #print self.C
                #print "cor(X) = "
                #print np.cov(X.T, bias = 1)
                #print "-------------------------"
                #self.C[dt1*2, dt2*2+1] = self.C[dt2*2+1, dt1*2] = 0.10
            
    def em(self, num_it = 3):
        for it in range(num_it):
            self.e_step()
            self.m_step()
            
        for data in self.datasets:
            data.get_dic_ss()
            
    
def setup_multitask(inter_cor = 0.0):
    import start
    start.main('proton-beam')
    lc = labels_collection(start.turk_data_id[:4000], 4000*[None])
    proton = vss_model(lc)
    
    #start.main('appendicitis')
    #lc = labels_collection(start.turk_data_id[:1000], 1000*[None])
    #appen = vss_model(lc)
    
    start.main('omega3')
    lc = labels_collection(start.turk_data_id[:250], 250*[None])
    omega3 = vss_model(lc)
    
    #start.main('dst')
    #lc = labels_collection(start.turk_data_id[:1000], 1000*[None])
    #dst = vss_model(lc)
    
    return multitask([proton, omega3], inter_cor)
    #return multitask([proton, appen, omega3, dst])
    
class mv_model():
    """
    """
    def __init__(self, lc):
        self.lc = lc
        self.N = lc.n
        self.mv_lab = np.zeros(self.N)

        self.dic_conf = {} # dic of confusion matrix
        self.dic_ss = {}

        self.prior_sen = global_psen
        self.prior_spe = global_pspe

        self.lab_per_w = {}

        self.init_prob()

    def init_prob(self, id_range = None):
        if id_range == None:
            left = 0; right = self.N
        else:
            left = id_range[0]; right = id_range[1]

        for (index, lw_set) in zip (range(left, right), self.lc.crowd_labels[left:right]):
          if self.lc.true_labels[index] == None:
            temp = 0
            nl = 0
            for lab in lw_set[0]:
                if lab >= 0:
                    temp += lab
                    nl += 1
            if nl == 0:
                #print index, lw_set
                self.mv_lab[index] = 0
            else:
                #self.mv_lab[index] = 1 if (temp*1.0/nl) >= 0.5 else 0
                self.mv_lab[index] = fix_r(temp*1.0/nl)
          else:
            self.mv_lab[index] = self.lc.true_labels[index]

        for (index, lw_set) in zip (range(left, right), self.lc.crowd_labels[left:right]):
            for (label, worker) in zip(*lw_set):
              if label >= 0:
                prob1 = self.mv_lab[index]

                if worker not in self.dic_conf:
                    self.dic_conf[worker] = [[0,0], [0,0]]
                    self.lab_per_w[worker] = 0
                self.dic_conf[worker][1][label] += prob1
                self.dic_conf[worker][0][label] += (1 - prob1)
                self.lab_per_w[worker] += 1

        psen = self.prior_sen; pspe = self.prior_spe

        for (worker, conf_mat) in self.dic_conf.items():
            try:
                sen = (psen[0] + conf_mat[1][1] * 1.0) / (conf_mat[1][1] + conf_mat[1][0] + psen[0] + psen[1])
                spe = (pspe[0] + conf_mat[0][0] * 1.0) / (conf_mat[0][0] + conf_mat[0][1] + pspe[0] + pspe[1])
            except ZeroDivisionError:
                sen = None
                spe = None

            if not sen == None:
                self.dic_ss[worker] = (sen, spe)
            else:
                self.dic_ss[worker] = (0.7, 0.05)

    def online(self, new_labels):
        if len(new_labels) == 0: return

        n = self.lc.n
        l = len(new_labels)
        self.lc.add_labels(new_labels, l * [None])
        self.N = self.lc.n
        self.mv_lab = np.hstack((self.mv_lab, np.zeros(l)))
        self.init_prob((n, self.N))



class hc_model():
    """
    Hybrid confusion model (Liu & Wang)
    """
    def __init__(self, lc):
        self.lc = lc
	self.prior_class = pymc.Dirichlet("prior class", [1,1], value = 0.5)
	self.nsam = 1000


    def build_model_def(self):
        """
        build model definition
        from the labels collection
        """
        # params:


        # for each item: a var of true label, hidden if no gold available
        N = self.lc.n
        items = np.empty(N, dtype=object)

        for i in range(N):
            if self.lc.true_labels[i] == None:
                items[i] = pymc.Bernoulli("Item_%i" % i, self.prior_class)
            else:
                l = self.lc.true_labels[i]
                items[i] = pymc.Bernoulli("Item_%i" % i, self.prior_class, value = (l==1), observed = True)


        # for each worker: 2 hidden var: sen and spe, drawn from Dirichlet 
        worker_dic = {}
        for (index, lw_set) in enumerate( self.lc.crowd_labels):
            for (label, worker) in zip(*lw_set):
                if not worker_dic.has_key(worker):
                    # create distribution for SS of the worker
                    ss = np.empty(2, dtype = object)
                    
                    psen = global_psen; pspe = global_pspe
                    sen = pymc.Dirichlet("sen_%s" % worker, [psen[0], psen[1]], value = psen[0] * 1.0/ (psen[0] + psen[1]) )
		    spe = pymc.Dirichlet("spe_%s" % worker, [pspe[0], pspe[1]], value = pspe[0] * 1.0/ (pspe[0] + pspe[1]) )

                    worker_dic[worker] = (sen, spe)

        # for each crowd label: a visible var
        wlabels = []
        list_prob = []
        list_p = []
        for (index, lw_set) in enumerate( self.lc.crowd_labels):
            for (label, worker) in zip(*lw_set):
                var_name = "x_" + worker + "_" + str(index)

                sen = worker_dic[worker][0]
		spe = worker_dic[worker][1]                

                #p = pymc.Lambda("prob_" + var_name, lambda trul = items[index]:
                #     prob_1 if trul == 1 else prob_0 )


                #def prob_eval(item_index, prob_0, prob_1):
                #    if item_index == 0:
                #        return prob_0
                #    else:
                #        return prob_1

                #p = pymc.Deterministic(eval = prob_eval,
                #  name = "prob_" + var_name,
                #  parents = {"Item_" + str(index): items[index], "prob_1_%s" %worker: prob_1, "prob_0_%s" %worker: prob_0},
                #  dtype=float)

		# prob of the worker returns 1 given the true label
                @pymc.deterministic(plot=False, name = "p_" + var_name)
                def p(item_index = items[index], p1 = sen, p0 = spe):
                    if item_index == 0: return 1 - p0
                    else: return p1

                
                list_p.append(p)
                wlabels.append( pymc.Bernoulli(var_name, p, value = label, observed = True) )
                #pymc.InvLogit()

        self.model_def = (items, worker_dic, wlabels, list_p, self.prior_class)


    def get_sample(self):
        self.S = pymc.MCMC(self.model_def)
        self.S.sample(20000, burn = 1000, thin = 10, progress_bar = False)

    def infer_dic_ss(self):
	self.get_sample()
	# get the sample trace, sen = average of trace

	mv = mv_model(self.lc)
        self.dic_ss = {}
	for w in mv.dic_ss:
            sen = np.mean(self.S.trace('sen_' + w)[:])
            spe = np.mean(self.S.trace('spe_' + w)[:]) 
	    self.dic_ss[w] = (sen, spe)

        del self.S
	    

