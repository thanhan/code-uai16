import csv
import run
# raw turk to label


def read_turk_dic_proton():
    """
    dic of abs_id to answers to (wid, q3,q4) for each worker
    """
    f = open("data/proton-beam-RawTurkResults.csv")

    first_line = f.readline()

    csv_reader = csv.reader(f)
    
    turk_dic = {}
    for row in csv_reader:        
        (AssignmentId, WorkerId, HITId, AcceptTime, SubmitTime, ApprovalTime, TimeToComplete, AbstractId, Question1, Question2, Question3, Question4, Relevant) = tuple(row)
        AbstractId = int(AbstractId)
        if AbstractId not in turk_dic: turk_dic[AbstractId] = []
        turk_dic[AbstractId].append( (Question3, Question4) )
        
    return turk_dic
    
def get_answer(q3, q4, dataset):
    if dataset == 'proton-beam':
        ans = 1
        if (q3 == 'No' or (not q4.isdigit())):
            ans = 0
    else:
        ans = 0
        if (q4 == 'CantTell' or q4 == 'Yes'):
            ans = 1
            
    return ans
    

def rawturk_to_label(turk_dic, dataset, fix = True):
    """
    Create data for training classifier
    from raw turk data
    """
    new_turk_dic = {}
    for AbstractId in turk_dic:
        turk_ans = turk_dic[AbstractId]
        anss = []
        for ans in turk_ans:
            infer_ans = get_answer(ans[0], ans[1], dataset)
            anss.append(infer_ans)
        final_ans = 0 if sum(anss) < 3 else 1
        new_turk_dic[AbstractId] = final_ans

    #Fix missing data for 996 and 3491 for proton-beam
    if fix:
        new_turk_dic[996] = 0
        new_turk_dic[3491] = 0    
                
    (key, turk_data) = zip(*new_turk_dic.items())  
    return turk_data
    
def rawturk_to_uncer_label(turk_dic, dataset, fix = True):
    """
    Read rawturk data
    Return (#no, #yes) for each query
    """
    new_turk_dic = {}
    for AbstractId in turk_dic:
        turk_ans = turk_dic[AbstractId]
        #print AbstractId, turk_ans
        anss = []
        for ans in turk_ans:
            infer_ans = get_answer(ans[0], ans[1], dataset)
            anss.append(infer_ans)
        num_ans_1 = sum(anss)
        num_ans_0 = len(anss) - num_ans_1
        new_turk_dic[AbstractId] = (num_ans_0, num_ans_1)
    
    if fix:
        new_turk_dic[996] = (5,0)
        new_turk_dic[3491] = (5,0) 
    
    (key, turk_data) = zip(*new_turk_dic.items())  
    return turk_data
    
    
def rawturk_to_id_label(turk_dic, dataset, fix = False):
    """
    Return turk labels with worker id
    
    """
    new_turk_dic = {}
    for AbstractId in turk_dic:
        turk_ans = turk_dic[AbstractId]
        #print AbstractId, turk_ans
        anss = []
        wid  = []
        for ans in turk_ans:
            infer_ans = get_answer(ans[0], ans[1], dataset)
            anss.append(infer_ans)
            wid.append(ans[2])
        new_turk_dic[AbstractId] = (anss, wid)
    
    if fix:
        new_turk_dic[996] = (5,0)
        new_turk_dic[3491] = (5,0) 
    
    (key, turk_data) = zip(*new_turk_dic.items())  
    return list(turk_data)
    
    
    
turk_dic = None
turk_data = None
turk_dic_uncer = None
turk_data_uncer = None
turk_data_id = None
    
def main(dataset ='proton-beam-xml', given_turk_dic = None):
    global turk_dic, turk_data, turk_dic_uncer, turk_data_uncer, turk_data_id
    
    if dataset == 'proton-beam-xml':
        turk_dic = read_turk_dic_proton()
        turk_data = rawturk_to_label(turk_dic, dataset)
        turk_data_uncer = rawturk_to_uncer_label(turk_dic, dataset)
    else:
        turk_dic = given_turk_dic
        turk_data = rawturk_to_label(turk_dic, dataset, fix = False)
        turk_data_uncer = rawturk_to_uncer_label(turk_dic, dataset, fix = False)
        turk_data_id = rawturk_to_id_label(turk_dic, dataset, fix = False)
        #print len(turk_data_id)
    #return turk_dic
    
    
        
        
