
# UOFA- Fact Extraction and VERification

This code is built on top of sheffield's fever baseline. So we assume you have installed all the required documents they mention in their [readme file](https://github.com/sheffieldnlp/fever-baselines)


Apart from that you will need PyProcessors over Docker. After you have installed [Docker](https://www.docker.com/), do:


- `Docker pull myedibleenso/processors-server:latest`

- `docker run -d -e _JAVA_OPTIONS="-Xmx3G" -p 127.0.0.1:8886:8888 --name procserv myedibleenso/processors-server`

note: the docker run command is for the very first time you create this container. 

Second time onwards use: `docker start procserv`




## In our project we are experimenting with fact verification but unlexicalized.
## As of Nov 2018 there are two main lines of develpment
1. With decomposable attention + hand crafted features of NER replacement
2. With handcrafted features + SVM


To run the decomposable attention model, with Smart NER (replace tokens with NER tags but checking if they exists in the claim) use `./run_oracle_decomp.sh`


To run the With handcrafted features + SVM pipe line (from fever baseline model) + SVM (our model): use `./runner_ir.sh`

note that: these shell scripts will run the whole training and testing on dev pipeline. If you want just pieces, comment out accordingly
    
note that :the IR part is in oracle mode.-i.e there is no Information retrieval being done on the fly. instead we rely on the gold data annotators found for ecah of the training data. However, do note that, the above statement is true only for classes SUPPORTS and REFUTES. For the class NOT ENOUGH INFO, there are two methods of retrieving evidences. via either using nearest neighbor or random. We are using the fever baseline's nearest neighbor methods


# Versions
These are the various versions in the fact verification code development cycle (and what they do) at University of Arizona. Note, there must be only one version of this document and preferably exists in the master branch

| Date of modification | name of the branch| git SHA| change made | New F1 score | New average Accuracy | New average Precision| Notes |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| | Nov 8th 2018  | Content Cell  | 9f20b8b8e3e79c6b3410b51c3905f58042d42d28  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | 
| Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  |
