import pickle
from sklearn.feature_extraction.text import HashingVectorizer

transformer=HashingVectorizer(stop_words='english')

lab={0:'"ANIMALS"',
     1:'"MUSIC"',
     2:'"FOOD"',
     3:'"OPERA"',
     4:'"BALLET"',
     5:'"BUSINESS \u0026 INDUSTRY"',
     6:'"SPORTS"',
     7:'"TELEVISION"',
     8:'"WORLD CITIES"',
     9:'"ISLANDS"',
     10:'"ART"',
     11:'"AMERICANA"',
     12:'"MUSICAL INSTRUMENTS"',
     13:'"WORD ORIGINS"'}

options={0:["the giant panda","whales","Bottlenose dolphin","Migrate"],
         1:["bows","Sir Edward Elgar","Chorister","Operetta"],
         2:["croutons","Egg Yolks","Mint","Doritos"],
         3:["Carmen","Marian Anderson","Meistersinger","Verdi"],
         4:["Vienna","Rodeo","Leonard Bernstein","Agnes de Mille"],
         5:["Disney","Maytag","United Fruit Company","Apple"],
         6:["Chris Evert","basketball","the foil","George Foreman"],
         7:["Mission: Impossible","Our brains","Joan Rivers","David Duchovny"],
         8:["London","Vienna","Baghdad","Rio de Janeiro"],
         9:["the Shetlands","Vancouver","the Antilles","the Bahamas"],
         10:["eggs","the Renaissance","Albrecht Durer","Édouard Manet"],
         11:["the Liberty Bell","Atlantic City","AstroTurf","Colorado"],
         12:["Viola","a ukelele","the tambourine","Fife"],
         13:["devious","Mascot","a termite","rhinestones"]}

filename = 'finalized_model.sav'
SVM = pickle.load(open(filename, 'rb'))

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
for v in testLabel:
        print(lab[v])
        print(options[v])
