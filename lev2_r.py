import pickle
from sklearn.feature_extraction.text import HashingVectorizer

transformer=HashingVectorizer(stop_words='english')

filename = 'Models\lev2_ballet_kmean.sav'
SVM = pickle.load(open(filename, 'rb'))

print(SVM.labels_)

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
