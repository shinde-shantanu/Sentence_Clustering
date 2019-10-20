from ast import literal_eval
import pickle
import json
print("clustering")
#!/usr/bin/python
import sklearn
import sys
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import HashingVectorizer
#if sys.version_info[0]>=3: raw_input=input
####
transformer=HashingVectorizer(stop_words='english')
###
model=LinearSVC()
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
train=[]
trainLabel=[]
##
##f=open('trivia_questions2.json','r')
##for x in f:
##        print(type(x))
##        d=json.loads(x)

##l=literal_eval({"_id":1000000,"cat":["TELEVISION"],"question":["'This Sunday night series is subtitled \"The New Adventures of Superman\"'"],"ans":["Lois \u0026 Clark","Santa Barbara","Mimi","Friends"],"d1":["TELEVISION"]})
##print(l)

lab={'"ANIMALS"':0,
     '"MUSIC"':1,
     '"FOOD"':2,
     '"OPERA"':3,
     '"BALLET"':4,
     '"BUSINESS & INDUSTRY"':5,
     '"SPORTS"':6,
     '"TELEVISION"':7,
     '"WORLD CITIES"':8,
     '"ISLANDS"':9,
     '"ART"':10,
     '"AMERICANA"':11,
     '"MUSICAL INSTRUMENTS"':12,
     '"WORD ORIGINS"':13}
###
f=open('trivia_questions2.txt','r')
import requests

i=0
from pprint import pprint
try:
        for i in range(0,204580):
                x=f.readline()
                #print(x)
                try:
                        q=x.split('"question":["')[1].split('"],"ans"')[0]
                        c=x.split('"cat":[')[1].split('],"ques')[0]
                ##        x=x.split("""_id":""")[1]
                ##        x=x.split("")
                ##        print(x)
                ##        me="1939907"
                ##        a= requests.get('https://api.mlab.com/api/1/databases/questions/collections/jpd_question?q={"_id":'+str(me)+'}&apiKey=TV6msbUHF5n72wwMQchCn94kPUi2f5SH')
                ##        dat=json.loads(x)
                ##	#s=tfile.readline().rstrip()
                ##	#s=str(s)
                ##	#var=s.split(' ')
                ##	#ans=""
                ####	for x in var[:len(var)-2]:
                ####		ans=ans+x
                        train.append(q)
                        trainLabel.append(lab[c])
                except:
                        pass
        x=f.readline()
        for i in range(304588,400000):
                x=f.readline()
                #print(x)
                try:
                        q=x.split('"question":["')[1].split('"],"ans"')[0]
                        c=x.split('"cat":[')[1].split('],"ques')[0]
                ##        x=x.split("""_id":""")[1]
                ##        x=x.split("")
                ##        print(x)
                ##        me="1939907"
                ##        a= requests.get('https://api.mlab.com/api/1/databases/questions/collections/jpd_question?q={"_id":'+str(me)+'}&apiKey=TV6msbUHF5n72wwMQchCn94kPUi2f5SH')
                ##        dat=json.loads(x)
                ##	#s=tfile.readline().rstrip()
                ##	#s=str(s)
                ##	#var=s.split(' ')
                ##	#ans=""
                ####	for x in var[:len(var)-2]:
                ####		ans=ans+x
                        train.append(q)
                        trainLabel.append(lab[c])
                except:
                        pass
except Exception as e:
        print(e)
        print(i)
TRAIN = transformer.fit_transform(train)
SVM=LinearSVC()
try:
	SVM.fit(TRAIN,trainLabel)
except Exception as e:
	print(e)

###
test=[]
print("Done")
r=int(input("Enter: "))
for i in range(r):
        s=input("IP: ")
test.append(s)
###
TEST = transformer.transform(test)
testLabel=SVM.predict(TEST)
###
for v in testLabel: print(v)
