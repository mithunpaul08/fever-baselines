import csv,os,sys
from random import shuffle
from retrieval.read_claims import UOFADataReader
from tqdm import tqdm
from processors import ProcessorsBaseAPI

class load_fever_DataSet():




    def read_parent(self,cwd,bodies,stances):
        path = cwd + "/data/fnc/"

        # read the stances into a dictionary. Note that stances are in the format: Headline,Body ID,Stance
        stances = self.read(path,stances)
        articles = self.read(path,bodies)

        # articles_dict = dict()
        #
        # # make the body ID an integer value
        # for s in self.stances:
        #     s['Body ID'] = int(s['Body ID'])
        #
        # # copy all bodies into a dictionary
        # for article in articles_list:
        #     articles_dict[int(article2['Body ID'])] = article['articleBody']

        return stances,articles

    def read(self,path,filename):

        rows = []
        with open(path  + filename, encoding='utf8') as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]

        shuffle(rows)

        return rows


    def annotate_fnc(self, stances,articles,logger):



        API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)

        ann_head_tr = "ann_head_tr.json"
        ann_body_tr = "ann_body_tr.json"

        try:
            os.remove(ann_head_tr)
            os.remove(ann_body_tr)

        except OSError:
            print("not able to find file")
        objUOFADataReader = UOFADataReader()

        for index,s in enumerate(tqdm(stances,total=len(stances),desc="for each stance:")):

            headline = s['Headline']
            bodyid = int(s['Body ID'])
            print(f"index:{index}")
            print(f"Headline:{headline}")
            print(f"bodyid:{bodyid}")
            actualBody = articles[bodyid]
            #actualBody=dump["articleBody"]
            hypothesis = headline
            premise = actualBody
            print(f"dump:{dump}")
            print(f"actualBody:{actualBody}")
            print(f"hypothesis:{hypothesis}")
            print(f"premise:{premise}")
            objUOFADataReader.annotate_and_save_doc(hypothesis, premise,bodyid,API, ann_head_tr, ann_body_tr, logger)

