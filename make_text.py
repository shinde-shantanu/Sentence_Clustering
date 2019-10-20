import pandas as pd
s=""
features={"GPE":1,
            "MONEY":2,
            "ORG":3,
            "DATE":4,
            "PERSON":5,
            "EVENT":6,
            "LOC":7,
            "PRODUCT":8,
            "LANGUAGE":9,
            "TIME":10,
            "PERCENT":11,
            "QUANTITY":12,
            "WORK_OF_ART":13,
            "CARDINAL":14,
            "NNP":15,
            "CD":16,
            "NN":17,
            "NOUN":18,
            "NUM":19,
            "RB":20}
df=pd.read_csv('op.csv')
for i,x in df.iterrows():
    s=s+x['answer']+" "+str(features[x['features from sentense(NER/POS)']])+"\n"
    
with open('trainingdata.txt','a') as f:
    f.write(s+str(i))
