#from allennlp.predictors.predictor import Predictor
#import allennlp_models.coref
import tokenizations
import copy
from nltk.tokenize import sent_tokenize
import json
import random
import csv
class Tree:

    def __init__(self, val):
        self.val = val
        self.nodes = []

    def add_node(self, val):
        self.nodes.append(Tree(val))
    def ana(self, maxh=0,height=1,maxv=0,arcs=0,arcc=0,lfc=0):
        if len(self.nodes)==0:
            lfc=lfc+1
        if height>maxh:
            maxh=height
        for c in self.nodes:
            maxh,_,maxv,arcs,arcc,lfc=c.ana(maxh,height+1,maxv,arcs+c.val-self.val,arcc+1,lfc)

        return maxh,height,maxv,arcs,arcc,lfc
            
    def addn(self, l):
        for e in range(len(l)):
            if l[e]==self.val:
                self.nodes.append(Tree(e+2))
                self.nodes[-1].addn(l)
    def add(self, n,a):
        for nn in self.nodes:
            nn.add(n,a)
        if self.val==n:
            self.nodes.append(Tree(a))
    def avg(self,l,s1,s2):
        s1=s1+l
        s2=s2+1
        for nn in self.nodes:
            s1,s2=nn.avg(l+1,s1,s2)
            #s1=s1+s01
            #s2=s2+s02
        return s1,s2
    def rb(self,s1,s2):
         
        s2=s2+1
        for nn in self.nodes:
            s1,s2=nn.rb(s1,s2)
            #s1=s1+s01
            #s2=s2+s02
            if nn.val==self.val+1:
                s1=s1+1

        return s1,s2
        
    def __repr__(self):
        return f"NonBinTree({self.val}): {self.nodes}"
    def display(self,indent=0):
        print( ('  '*indent)+str(self.val))        
        for c in self.nodes:
            c.display(indent+1)
random.seed(0)
with open('/scratch/cluster/wjko/ProfileREG/transformers/discourse/predictions_20000.json') as fa:
    faa=fa.readlines()
faa.pop(0)
j=dict()
jj=dict()
akk=0
akkk=0
with open('/scratch/cluster/wjko/parseturk/qa/B3.csv') as csvfile:
#with open('/scratch/cluster/wjko/parseturk/qa/Batch_315619_batch_results.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        if row[0] =='HITId':
            for aa in range(len(row)):
                jj[row[aa]]=aa
        if not row[0] =='HITId':
            if int(float(row[68])) in j:
                j[int(float(row[68]))].append(row)
            else:
                j[int(float(row[68]))]=[row]

f=open('/scratch/cluster/wjko/squad/train-v2.0.json')
y = json.load(f)
yy=copy.deepcopy(y)
qtempl=copy.deepcopy(yy['data'][0]["paragraphs"][0]["qas"][0])
lsh=0
ddd=[]
ts=dict()
a0=0
a1=0
a2=0
a3=0
a4=0
ac=0
for k in j:
    asa=copy.deepcopy(y['data'][0])
    asa["paragraphs"]=[asa["paragraphs"][0]]
    asa["paragraphs"][0]["qas"]=[]
    rare=[]
    asa["paragraphs"][0]["context"]=''
    for dd in range(len(j[k])):
        for cc in range(20):
            j[k][dd][jj['Input.ac'+str(cc)]]=' '.join(j[k][dd][jj['Input.ac'+str(cc)]].split(' ')[1:])
    for cc in range(20):
        asa["paragraphs"][0]["context"]=asa["paragraphs"][0]["context"]+' XT'+str(cc+1).zfill(2)+' '+j[k][0][jj['Input.ac'+str(cc)]]     
                 
    for dd in range(len(j[k])):
        print(k)
        tr=Tree(1)
        for cc in range(19):
            if (j[k][dd][jj['Answer.teg'+str(cc)]]).isdigit():
                qtt=copy.deepcopy(qtempl)
        #svv=qss[2].split()
                sn=j[k][dd][jj['Answer.teg'+str(cc)]]
                qtt['question']=j[k][dd][jj['Input.ac'+str(int(float(cc))+1)]]
                qtt['id']=str(lsh).zfill(24)
                lsh=lsh+1

                if int(float(sn))<=20:
                    if False:
                        qtt['is_impossible']=True
                        qtt['answers']=[]
                    
                    else:
                        qtt['is_impossible']=False

                        q11=1
                        for awa in range(int(float(sn))-1):
                            q11=q11+len(j[k][dd][jj['Input.ac'+str(awa)]])+6        
                        qtt['answers']=[{"text": 'XT'+sn.zfill(2), "answer_start":q11}]
                        er=int(float(faa.pop(0)[35:37]))
                        tr.add(er,cc+2)
                        
                    #if (not int(float(sn))-cc==1) or random.random()<0.2:
                    rare.append(qtt)                                    
        tr.display()
        
        if True:
            
            if k not in [1492,1427]:
                maxh,height,maxv,arcs,arcc,lfc=tr.ana()
                s01,s02=tr.avg(1,0,0)
                ss01,ss02=tr.rb(0,0)
                arc=arcs/arcc/arcc
                lf=lfc/(arcc+1)
                ac=ac+1
                a0=a0+maxh
                a1=a1+arc
                a2=a2+lf
                a3=a3+s01/s02
                a4=a4+ss01/ss02
                if k in ts:
                    ts[k].append(tr)
                else:
                    ts[k]=[tr]
    for qqes in rare:
        asa["paragraphs"][0]["qas"].append(qqes)
    ddd.append(asa)
    
yy['data']=ddd
print(a0/ac)
print(a1/ac)
print(a2/ac)

print(a3/ac)
print(a4/ac)
#with open('/scratch/cluster/wjko/squad/discoursetest.json', 'w') as outfile:
#    json.dump(yy, outfile) 

                              