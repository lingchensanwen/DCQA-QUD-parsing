import os
import copy
import json
import random
import csv
random.seed(0)

anchor_info=open('./ao/predictions_.json','r')
anchor_info.readline()
question_file=open('./questions.txt','r')
fw=open('./glue/glue_data/WNLI/dev.tsv','w')
fw.write('index\tsentence1\tsentence2\tlabel\n')
top_question_num=1
directory='./inputa/'
articles=[]
# save file information in files
for filename in sorted(os.listdir(directory)):
    article = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(article):
        each_article=[]
        file=open(article,'r')
        for line in file:
            #get sentence(remove sentence number)
            each_article.append(line.strip().split('\t')[1])
        articles.append(each_article)

f=open('./train-v2.0.json')
train_file = json.load(f)
train_file_copy=copy.deepcopy(train_file)
first_question=copy.deepcopy(train_file_copy['data'][0]["paragraphs"][0]["qas"][0])
question_num=0
new_articles_info=[]

for file_num in range(len(articles)):

    #get essay related information
    essay=copy.deepcopy(train_file['data'][0])
    #get all paragraphs related information
    essay["paragraphs"]=[essay["paragraphs"][0]]
    #get questions related to the paragraphs
    essay["paragraphs"][0]["qas"]=[]
    rare=[]
    essay["paragraphs"][0]["context"]=''
    current_article = articles[file_num]
    for sentence_num in range(len(current_article)):
        essay["paragraphs"][0]["context"]=essay["paragraphs"][0]["context"]+' XT'+str(sentence_num+1).zfill(2)+' '+current_article[sentence_num] 
             

    for cc in range(len(current_article)-1):
        questions=copy.deepcopy(first_question)
        anchor_sentence_pos=anchor_info.readline()[35:37]
        if(len(anchor_sentence_pos)==0):
            continue
        question_num=question_num+1

        questions['is_impossible']=False
        q11=1
        for awa in range(cc+1):
            q11=q11+len(current_article[awa])+6        
        questions['answers']=[{"text": 'XT'+str(cc+2).zfill(2), "answer_start":q11}]
        rare.append(questions) 
                
        for top_question_num in range(10):
            generated_question=question_file.readline().strip() 
            fw.write(str(top_question_num)+'\t'+generated_question+'\t'+current_article[int(float(anchor_sentence_pos))-1]+' | '+current_article[cc+1]+'\t0\n')       
        

    for qqes in rare:
        essay["paragraphs"][0]["qas"].append(qqes)
    new_articles_info.append(essay)
    
train_file_copy['data']=new_articles_info


                              