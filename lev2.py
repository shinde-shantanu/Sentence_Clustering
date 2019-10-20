from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import pickle
from sklearn.feature_extraction.text import HashingVectorizer

transformer=HashingVectorizer(stop_words='english')

f=open('Text_files\ballet.txt','r')

train=[]
trainLabel=[]

for x in f:
    q=x.split('"question":["')[1].split('"],"ans"')[0]
    c=x.split('"cat":[')[1].split('],"ques')[0]
    train.append(q)
    #trainLabel.append(lab[c])

TRAIN = transformer.fit_transform(train)
print(type(TRAIN[1]))
model = KMeans()

try:
	model.fit(TRAIN)
except Exception as e:
	print(e)

test=[]
print("Done")
r=int(input("Enter: "))
for i in range(r):
        s=input("IP: ")
test.append(s)
#####
TEST = transformer.transform(test)
testLabel=model.predict(TEST)

for v in testLabel: print(v)

filename = 'Models\lev2_ballet_kmean.sav'
pickle.dump(model, open(filename, 'wb'))
##print(type(TRAIN))
##print("hw")
##lab=['"ANIMALS"',
##     '"MUSIC"',
##     '"FOOD"',
##     '"OPERA"',
##     '"BALLET"',
##     '"BUSINESS \u0026 INDUSTRY"',
##     '"SPORTS"',
##     '"TELEVISION"',
##     '"WORLD CITIES"',
##     '"ISLANDS"',
##     '"ART"',
##     '"AMERICANA"',
##     '"MUSICAL INSTRUMENTS"',
##     '"WORD ORIGINS"']
##
##mergings = linkage(train, method='complete')
##dendrogram(mergings,
##           labels=lab,
##           leaf_rotation=90,
##           leaf_font_size=6,
##           )
##
##plt.show()
##
##filename = 'lev2_ballet_hierarchy.sav'
##pickle.dump(mergings, open(filename, 'wb'))
