from numpy import number
NUMBER_QUESTION_SELECTION = 10

f=open("./output.txt",'r')
fw=open("./results.txt",'w')
f2=open("./questions.txt",'r')
number_of_questions = sum(1 for _ in open('questions.txt'))

for i in range(int(number_of_questions/NUMBER_QUESTION_SELECTION)):
    maxx=-99999999999999
    for j in range(NUMBER_QUESTION_SELECTION):
        l=f2.readline()
        try:
            score = f.readline().strip()
            ll=float(score)
            if ll>maxx:
                maxx=ll
                fo=l
        except ValueError:
            print("the result cannot be coverted float: "+score)
            exit()
    fw.write(fo)