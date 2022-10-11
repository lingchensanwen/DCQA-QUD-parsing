f=open("./output.txt",'r')
fw=open("./results.txt",'w')
f2=open("./questions.txt",'r')
for i in range(19*3):
    maxx=-99999999999999
    for j in range(10):
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