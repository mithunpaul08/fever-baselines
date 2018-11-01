

class load_fever_DataSet():
    def __init__(self, cwd,bodies, stances):

        self.path = cwd+"/data/"

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

    def read(self, file_path: str):

        instances = []

        ds = FEVERDataSet(file_path, reader=self.reader, formatter=self.formatter)
        ds.read()

        for instance in tqdm.tqdm(ds.data):
            if instance is None:
                continue

            if not self._sentence_level:
                pages = set(ev[0] for ev in instance["evidence"])
                premise = " ".join([self.db.get_doc_text(p) for p in pages])
            else:
                lines = set([self.get_doc_line(d[0], d[1]) for d in instance['evidence']])
                premise = " ".join(lines)

            if len(premise.strip()) == 0:
                premise = ""

            hypothesis = instance["claim"]
            label = instance["label_text"]
            instances.append(self.text_to_instance(premise, hypothesis, label))
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)



