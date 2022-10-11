import os
import copy
import json
import random
import csv
random.seed(0)

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
train_file_copy =copy.deepcopy(train_file)
#get first question related information(question, question_id, answer, answer_start) in paragraphs
#later use this one to create a question format as this
first_question=copy.deepcopy(train_file_copy['data'][0]["paragraphs"][0]["qas"][0])

question_num=0
new_articles_info=[]

for k in range(len(articles)):
    #get essay related information
    essay=copy.deepcopy(train_file['data'][0])
    #get all paragraphs related information
    essay["paragraphs"]=[essay["paragraphs"][0]]
    #get questions related to the paragraphs
    essay["paragraphs"][0]["qas"]=[]
    questions=[]

    essay["paragraphs"][0]["context"]=''
    current_article = articles[k]

    # add sentence id before each sentence
    for sentence in range(len(current_article)):
        essay["paragraphs"][0]["context"]=essay["paragraphs"][0]["context"]+' XT'+str(sentence+1).zfill(2)+' '+ current_article[sentence]     

    for sentence in range(len(current_article)-1):
        sn=1 #placeholder for sentence number
        q11=1 #placeholder for answer start place
        question=copy.deepcopy(first_question) #copy the question format from first question in train.json
        question['question']=articles[k][sentence+1] #the answer sentence here is our question
        #pad question id with 0000.. and make it length of 24
        question['id']=str(question_num).zfill(24)
        question_num=question_num+1

        question['is_impossible']=False
        question['answers']=[{"text": 'XT'+str(sn).zfill(2), "answer_start":q11}]
        questions.append(question)  

    for question in questions:
        essay["paragraphs"][0]["qas"].append(question)
    new_articles_info.append(essay)
    
train_file_copy['data']=new_articles_info
with open('a.json', 'w') as outfile:
    json.dump(train_file_copy, outfile) 

                              