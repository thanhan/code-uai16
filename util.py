import xml.etree.ElementTree as ET
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import scipy

import csv

import os

def get_text(a):
    try:
        return a.text
    except AttributeError:
        return ''

def get_relevant():
    f = open('data/proton-beam-relevant.txt')
    res = np.zeros(4751)
    for line in f:
        x = int(line)
        res[x-1] = 1
    f.close()
    return res

def get_pub_dic_xml(file_name = 'data/proton-beam-all.xml'):
    tree = ET.parse(file_name)
    root = tree.getroot()[0]

    # Create dic of : id -> text features
    pub_dic = {}
    for pub in root:
        rec_number = int (get_text (pub.find('rec-number')))
        abstract   = get_text (pub.find('abstract'))
        title      = get_text (pub.find('titles')[0])
        text = title + abstract
        for kw in pub.find('keywords'):
            text = text + kw.text + ' '
        pub_dic[rec_number] = text

    return pub_dic




def get_pub_dic_csv(dataset):
    filename = "data/" + dataset + "-text.csv"
    f = open(filename)
    f.readline()
    csv_reader = csv.reader(f)

    # Create dic of : id -> text features
    pub_dic = {}

    for row in csv_reader:
        if dataset.startswith("RCT"):
            (abstract_id, abstract, title) = tuple(row)[0:3]
        else:
            (abstract_id, title, publisher, abstract) = tuple(row)[0:4]

        abstract_id = int(abstract_id)
        text = title + abstract

        pub_dic[abstract_id] = text

    return pub_dic


def new_lab(l):
    l = int(l)
    if l == 2: return 0
    if l == 3: return -1
    return l

def get_turk_data(dataset):
    filename = "data/" + dataset + "-turk.csv"
    f = open(filename)
    first_line = f.readline()
    csv_reader = csv.reader(f)

    turk_dic = {}
    rel_dic = {}
    for row in csv_reader:
        #print len(row)
        if dataset.startswith ('omega3'):
            (AssignmentId, WorkerId, HITId, AcceptTime, SubmitTime, ApprovalTime, TimeToComplete, PMID, AbstractId, Question2, Question3, Question4, Relevant, Honeypot) = tuple(row)
        elif dataset.startswith ('proton-beam'):
            (AssignmentId, WorkerId, HITId, AcceptTime, SubmitTime, ApprovalTime, TimeToComplete, PMID, AbstractId, Question1, Question2, Question3, Question4, Relevant) = tuple(row)
        elif dataset.startswith("RCT"):
            (row, abs_id, labeler_type, date, label, final_decision, labeler_id) = tuple(row)
        else:
            (AssignmentId, WorkerId, HITId, AcceptTime, SubmitTime, ApprovalTime, TimeToComplete, PMID, AbstractId, Question1, Question2, Question3, Question4, Relevant, Honeypot) = tuple(row)

        if dataset.startswith("RCT"):
            abs_id = int(abs_id)
            if abs_id not in turk_dic: turk_dic[abs_id] = []
            # final ID = First char of type + ID
            if new_lab(label) > -1:
                turk_dic[abs_id].append(( new_lab(label), labeler_type[0] + labeler_id)  )
            else:
                turk_dic[abs_id].append(( 0, labeler_type[0] + labeler_id)  )
                turk_dic[abs_id].append(( 1, labeler_type[0] + labeler_id)  )
            rel_dic[abs_id] = new_lab(final_decision)
        else:
            AbstractId = int(AbstractId)
            if AbstractId not in turk_dic: turk_dic[AbstractId] = []
            turk_dic[AbstractId].append( (Question3, Question4, WorkerId) )
            rel_dic[AbstractId] = Relevant

    return (turk_dic, rel_dic)

# Global vars:
#   mat: features matrix
#   rel: true relevance (label)
#   turk_dic: id -> (q3, q4, WorkerId)
mat = None
rel = None
turk_dic = None

def main(dataset = 'proton-beam-xml'):
    csv.field_size_limit(430000)
    global mat, rel, turk_dic

    if dataset == 'proton-beam-xml':
        pub_dic_tmp = get_pub_dic_xml()
        # pub_dic_items are already sorted by key
        [rec_nums, texts] = zip(*pub_dic.items())
        rel = get_relevant()
    else:
        pub_dic_tmp = get_pub_dic_csv(dataset)
        #[rec_nums, texts] = zip(*pub_dic.items())
        (turk_dic_tmp, rel_dic_tmp) = get_turk_data(dataset)

        texts = []
        pub_dic = {}; turk_dic = {}; rel_dic = {}

        for i in sorted(pub_dic_tmp.keys()):
            if pub_dic_tmp.has_key(i) and turk_dic_tmp.has_key(i) and rel_dic_tmp.has_key(i):
                texts.append(pub_dic_tmp[i])
                pub_dic[i] = pub_dic_tmp[i]
                turk_dic[i] = turk_dic_tmp[i]
                rel_dic[i] = rel_dic_tmp[i]
            #else:
            #    if pub_dic.has_key(i): pub_dic.pop(i)
            #    if turk_dic.has_key(i): turk_dic.pop(i)
            #    if rel_dic.has_key(i): rel_dic.pop(i)

        (_,rel) = zip(*sorted(rel_dic.items()))
        rel = map(int, rel)

    vectorizer = TfidfVectorizer()
    #save_texts = texts
    mat = vectorizer.fit_transform(texts)
    return (pub_dic, texts)


def write_aggregate_init(directory = 'temp/'):
    """
    Write files for aggregation
    --categories <categoriesfile>         : The <categoriesfile> can also be used
                                         to define the prior values for the
                                         different categories, instead of
                                         letting the priors be defined by the
                                         data. In that case, it becomes a
                                         tab-separated file and each line has
                                         the form <category><tab><prior>
     --cost <costfile>                     : A tab-separated text file. Each line
                                         has the form <from_class><tab><to_class
                                         ><tab><classification_cost> and
                                         records the classification cost of
                                         classifying an object thatbelongs to
                                         the `from_class` into the `to_class`.
    """
    f = open(directory + 'categories.txt', 'w')
    f.write("1\t0.5\n")
    f.write("0\t0.5\n")
    f.close()

    f = open(directory + 'costs.txt', 'w')
    f.write("1\t0\t10\n") # recall loss
    f.write("0\t1\t1\n")
    f.close()

    #create empty gold and labels files:
    f = open(directory + 'gold.txt', 'w')
    f.close()

    f = open(directory + 'labels.txt', 'w')
    f.close()



def write_aggregate_next(directory = 'temp/', golds = [], crowds = []):
    """
    Incrementally write new crowd/expert labels to files
     --gold <goldfile>                     : A tab-separated text file. Each line
                                         has the form <objectid><tab><gold_label
                                         > and records the gold label for
                                         whatever objects we have them.
 --input <inputfile>                   : A tab-separated text file. Each line
                                         has the form <workerid><tab><objectid><
                                         tab><assigned_label> and records the
                                         label that the given worker gave to
                                         that object
    """

    f = open(directory + 'gold.txt', 'a')
    for gold in golds:
        f.write(str(gold[0]) + '\t' + str(gold[1]) + '\n')
    f.close()


    f = open(directory + 'labels.txt', 'a')
    for crowd in crowds:
        f.write(str(crowd[0]) + '\t' + str(crowd[1]) + '\t' + str(crowd[2]) + '\n')
    f.close()


def get_conf(s):
    """
    extract number from confution mat of get-another-l:
    """
    #x = float(s.split()[0].split('=')[1].split('%')[0]) / 100
    #y = float(s.split()[1].split('=')[1].split('%')[0]) / 100

    res = [[0,0],[0,0]]
    for x in s.split():
        i = int(x[2])
        j = int(x[5])
        res[i][j] = float(x.split('=')[1].split('%')[0]) / 100
    return (res[0][0], res[0][1], res[1][0], res[1][1])


def aggregate(directory = "temp/", has_gold = True):
    """
    Do crowd aggregation
    """
    import subprocess
    #call("java -Xmx2048m -ea -cp lib/jblas-1.2.3.jar:lib/log4j-1.2.14.jar:qa-1.0.jar  org.square.qa.analysis.Main")

    #subprocess.check_call("pwd", shell  = True)
    #subprocess.check_call("wc temp/*", shell  = True)
    subprocess.check_call("sh get-another-label/bin/get-another-label.sh --categories " + directory +
      "categories.txt --input " + directory + "labels.txt " + " --cost " + directory + "costs.txt" + (" --gold " + directory + "gold.txt") if has_gold else "" , shell = True)


    #"--gold " + directory + "gold.txt
    #subprocess.check_call("ls datac", shell = True)



    # read results
    dic_ds = {}
    dic_mv = {}
    with open("results/object-probabilities.txt") as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            i = int(row["Object"])
            prob_ds = float(row["DS_Pr[1]"])
            prob_mv = float(row["MV_Pr[1]"])
            dic_ds[i] = prob_ds
            dic_mv[i] = prob_mv
    ds = zip(*sorted(dic_ds.items()))[1]
    mv = zip(*sorted(dic_mv.items()))[1]

    f = open('results/worker-statistics-detailed.txt','r')
    l = list(f)
    wid = None
    dic_ss = {}
    i = 0
    while True:
        line = l[i]
        if line.startswith('Worker:'):
            wid = line.split()[1].strip()
        elif line.startswith('Estimated Confusion Matrix'):
            #(n00, n01) = get_conf(l[i+1])
            #(n10, n11) = get_conf(l[i+2])
            (n00, n01, n10, n11) = get_conf(l[i+1] + l[i+2])
            i+= 2
            try:
                sen = n11 * 1.0 / (n10 + n11)
                spe = n00 * 1.0 / (n00 + n01)
            except ZeroDivisionError:
                sen = None
                spe = None
            dic_ss[wid] = (sen, spe)
        i += 1
        if i >= len(l): break

    f.close()

    # delete files:
    #for fi in os.listdir(directory):
    #    os.remove(directory + fi)
    #os.remove(directory + "labels.txt")
    os.remove("results/object-probabilities.txt")
    return (ds, mv, dic_ss)



def classify(n = 50):
    #clf = MultinomialNB(fit_prior=False)
    #clf = SVC(gamma=2, C=1, class_weight = {0.0:0.063829777, 1.0:1.0})
    clf = SGDClassifier(loss="log", penalty="l1", class_weight = {0.0:0.022, 1.0:1.0})

    clf.fit(mat[:n], rel[:n])
    return clf


def confu_mat(rel, turk_rel):
    m = [[0,0],[0,0]]
    for i in range(len(rel)):
        m[rel[i]][turk_rel[i]] += 1
    return m

def plot_pr(gold, predicted_prob, lb):
    pp1 = predicted_prob[:,1] # prob for class 1
    p, r, th = precision_recall_curve(gold, pp1)
    ap = average_precision_score(gold, pp1)
    plt.plot(r, p, label= lb + ' (area = {0:0.2f})'
                   ''.format(ap))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision and Recall')
    plt.legend(loc="upper right")
    #plt.show()

def eval_clf(gold, clf, mat, start = 0):
    pp = clf.predict_proba(mat[start:,:])
    pp1 = pp[:,1]
    ap = average_precision_score(gold[start:], pp1)
    return ap

def train_and_plot(ex = [50,100,200]):
    """
    train the classifier with ex[i] examples
    Plot
    """

    for num in ex:
        clf = classify(num)
        pp = clf.predict_proba(mat)
        plot_pr(rel[2000:], pp[2000:], str(num))


def get_balance_data(mat, rel):
    mat_1 = mat[ np.nonzero(rel == 1)[0] ]
    mat_0 = mat[ np.nonzero(rel == 0)[0] ]

    #print mat_1.shape, mat_0.shape

    n = min(mat_1.shape[0], mat_0.shape[0])

    #shuffle mat_0
    index = np.arange( mat_0.shape[0] )
    np.random.shuffle(index)
    mat_0 = mat_0[index]

    #print mat_0.shape

    new_mat = scipy.sparse.vstack([mat_1[:n], mat_0[:n]], 'csr')
    new_rel = np.hstack([np.ones((n,)), np.zeros((n,))] )

    #print new_mat, new_rel.shape

    #shuffle new mat and rel
    index = np.arange(new_mat.shape[0])
    np.random.shuffle(index)

    new_mat = new_mat[index]
    new_rel = new_rel[index]

    return (new_mat, new_rel)


    #s = [0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20, 37, 44, 68, 71, 118, 141, 162,183, 189, 248, 249, 255, 267, 268, 324]

    #
    #from sklearn.cross_validation import KFold
    #kf = KFold(n, n_folds=10)
    #acc_list = []
    #for train, test in kf:
    #    clf.fit(mat[train], rel[train])
    #    predicted = clf.predict(mat[test])
    #    acc = sum(predicted == rel[test]) * 1.0 / len(rel[test])
    #    acc_list.append(acc)

    #print 'average accuracy: ', np.average(acc_list)

    #for i in range(20, 1000, 20):
    #    clf.fit(mat[0:i], rel[0:i])
    #    predicted = clf.predict(mat[1000:])
    #    acc = sum(predicted == rel[1000:]) * 1.0 / len(rel[1000:])
    #    print i, acc
    #from sklearn.svm import SVC

    #clf = SVC()

    #clf.fit(mat, rel)
