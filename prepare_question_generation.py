#from allennlp.predictors.predictor import Predictor
#import allennlp_models.coref
import os

import copy
import json
import random
import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

anchor_info=open('./ao/predictions_.json','r')
anchor_info.readline()
random.seed(0)

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

fw=open('./ner.txt','w')
info_mapping_file = open("extra_info_mapping.txt", "w")
info_mapping_file.write("answer_id\tanswer_sentence\tanchor_id\tanchor_sentence\n")
directory='./inputa/'
articles=[]
for filename in sorted(os.listdir(directory)):
    article = os.path.join(directory, filename)
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
    questions=[]
    essay["paragraphs"][0]["context"]=''

    current_article = articles[file_num]
    sentences=[]
    sentences_after_ner=[]
    context=''

    for sentence_num in range(len(current_article)):
        essay["paragraphs"][0]["context"]=essay["paragraphs"][0]["context"]+' XT'+str(sentence_num+1).zfill(2)+' '+articles[file_num][sentence_num] 
        sentence=articles[file_num][sentence_num]
        sentences.append(sentence)
        ner_results = nlp(sentence)
        sentence_copy=sentence
        sentence=sentence.split()

        for item_entity in ner_results:
            if not item_entity['word'][:2]=='##':
                item_entity_new=item_entity
            else:
                item_entity_new['word']=item_entity_new['word']+item_entity['word'][2:]

        for item_entity in ner_results:
            if not item_entity['word'][:2]=='##':
                for word_num in range(len(sentence)):
                    if item_entity['word'] in sentence[word_num]:
                        sentence[word_num]=item_entity['entity'].split('-')[-1]

        context=context+sentence_copy
        sentence=' '.join(sentence)
        sentences_after_ner.append(sentence)

    anchor_dict=dict()
    dd = 1
    for cc in range(len(current_article)-1):
        anchor_dict[(dd,cc)]=anchor_info.readline()   

    for cc in range(len(current_article)-1):
        #cc is the answer sentence(each sentence in original eassy)
        question=copy.deepcopy(first_question)
        sentence_anchor_pair=anchor_dict[(dd,cc)]
        if(len(sentence_anchor_pair[35:37])==0):
            continue
        anchor_sentence_pos=int(sentence_anchor_pair[35:37])
        question['question']=current_article[cc]+' | '+current_article[int(float(anchor_sentence_pos))-1]
        if int(float(anchor_sentence_pos))<(len(current_article)+1):
            info_mapping_file.write(str(cc+2)+"\t"+current_article[cc+1]+"\t"+str(anchor_sentence_pos)+"\t"+sentences[int(float(anchor_sentence_pos))-1]+"\n")
            fw.write(' '.join(sentences[:int(float(anchor_sentence_pos))-1]) if not len(sentences[:int(float(anchor_sentence_pos))-1])==1 else sentences[:int(float(anchor_sentence_pos))-1][0])
            fw.write(' <@ '+sentences[int(float(anchor_sentence_pos))-1]+' (> ')
            fw.write(' '.join(sentences[int(float(anchor_sentence_pos)):cc+1]) if not len(sentences[int(float(anchor_sentence_pos)):cc+1])==1 else sentences[int(float(anchor_sentence_pos)):cc+1][0])
            fw.write(' || '+sentences[int(float(anchor_sentence_pos))-1]+' || '+sentences_after_ner[cc+1])
            fw.write(' | '+current_article[cc]+'\n') #placeholder for question
        question['id']=str(question_num).zfill(24)
        question_num=question_num+1

        question['is_impossible']=False
        q11=1
        for awa in range(cc+1):
            q11=q11+len(current_article[awa])+6        
        question['answers']=[{"text": 'XT'+str(cc+2).zfill(2), "answer_start":q11}]
        questions.append(question)                                    
    for qqes in questions:
        essay["paragraphs"][0]["qas"].append(qqes)
    new_articles_info.append(essay)
    
train_file_copy['data']=new_articles_info