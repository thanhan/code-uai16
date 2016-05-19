import csv


def get_answer(q3, q4):
    """
    Return the answer of a worker about relevance
    """
    #ans = 0
    #if (q3 == 'Yes' and q4.isdigit()):
    #    if int(q4) > 0: ans = 1
    ans = 1
    if (q3 == 'No' or (not q4.isdigit())):
        ans = 0
    return ans
    
def get_answer2(q3, q4):
    if (q3 == 'Yes' or q3 == 'NoInfo' or q3 == 'CantTell'):
        return 1
    return 0
    
    

    
def add_answer(aid, ans, ans_dic):
    if aid not in ans_dic:
        ans_dic[aid] = [0,0]
    ans_dic[aid][ans] += 1
    
    
    
def main():

    f = open("RawTurkResults.csv")

    first_line = f.readline()

    csv_reader = csv.reader(f)
    cm = [[0, 0], [0, 0]]

    ans_dic = {}
    rel = {}
    for row in csv_reader:
        aid = row[7]
        q3 = row[10]
        q4 = row[11]
        r =  row[12]
        
        ans = get_answer(q3, q4)
        
        rel[aid] = int(r)
        add_answer(aid, ans, ans_dic)
        

        #cm[int(r)][ans] += 1
        
    for aid in ans_dic:
        r = rel[aid]
        ans_0 = ans_dic[aid][0]
        ans_1 = ans_dic[aid][1]
        if ans_0 + ans_1 != 5:
            print "something not 5"
        mar_vote = 0 if ans_0 > ans_1 else 1
        at_least_1 = 0 if ans_0 == 5 else 1
        unanimous = 0 if ans_0 > 0 else 1
        cm[r][unanimous] += 1
        if r == 0 and unanimous==1:
            print aid
        
    print cm
    f.close()
