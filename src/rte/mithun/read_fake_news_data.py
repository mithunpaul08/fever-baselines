

class load_fever_DataSet():
    def __init__(self, cwd,bodies, stances):

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


