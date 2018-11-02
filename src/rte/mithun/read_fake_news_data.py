import csv,os
from random import shuffle
from retrieval.read_claims import UOFADataReader
from tqdm import tqdm
from processors import ProcessorsBaseAPI

class load_fever_DataSet():
    def __init__(self, cwd,bodies, stances):
        API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)

        self.path = cwd+"/data/fnc/"

        #read the stances into a dictionary. Note that stances are in the format: Headline,Body ID,Stance
        self.stances = self.read(stances)
        articles = self.read(bodies)

        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

       #
    def read(self,filename):
        rows = []
        with open(self.path  + filename, encoding='utf8') as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]

        shuffle(rows)

        return rows


    def annotate_fnc(self, data):

        ann_head_tr = "ann_head_tr.json"
        ann_body_tr = "ann_body_tr.json"

        try:
            os.remove(ann_head_tr)
            os.remove(ann_body_tr)

        except OSError:
            print("not able to find file")
        objUOFADataReader = UOFADataReader()

        for s in (tqdm.tqdm(data.stances)):

            headline = s['Headline']
            bodyid = s['Body ID']
            actualBody = data.articles[bodyid]

            hypothesis = headline
            premise = actualBody



            objUOFADataReader.annotate_and_save_doc(hypothesis, premise,bodyid, self.API, ann_head_tr, ann_body_tr, logger)

