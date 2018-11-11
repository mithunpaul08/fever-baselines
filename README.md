
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


## to run the entire pipe line with IR (from fever baseline model) + SVM (our model): use ./runner_ir.sh (or the commands within)


## To run instead, the decomposable attention model, with Smart NER (replace tokens with NER tags but checking if they exists in the claim) use ./run_oracle_decomp.sh- note that 
##the IR part is in oracle mode.-i.e there is no Information retrieval being done on the fly. instead we rely on the gold data annotators found for ecah of the training data. However, do note that, the above statement is true only for classes SUPPORTS and REFUTES. For the class NOT ENOUGH INFO, there are two methods of retrieving evidences. via either using nearest neighbor or random. We are using the fever baseline's nearest neighbor methods
 
    

- source activate fever

## to run training from my_fork folder on jenny
`PYTHONPATH=src python src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file data/fever-data/train.jsonl --out-file data/fever/train.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode train --lmode WARNING`


## to run training from another folder on jenny
PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/train.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/train.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode train --lmode WARNING

## to run training on a smaller data set from another folder on jenny
PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db
--model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/train.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/train.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode small  --dynamic_cv True


 ## To run our entailment trainer on training data alone :

data_root="/work/mithunpaul/fever/my_fork/fever-baselines/data"

## To run on dev

`PYTHONPATH=src python src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file data/fever-data/dev.jsonl --out-file data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode dev --lmode WARNING`

## to run dev in a  folder branch_myfork in server but feeding from same data fold
`PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/dev.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode dev --lmode INFO`

## to run testing
`PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever-data/dev.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --mode test --dynamic_cv True`

## to run dev after running the nearest neighbors algo for not enough info class (note that this assumes that you have run the NEI code mentioned below by sheffield)
`PYTHONPATH=src python src/scripts/retrieval/ir.py --db /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/fever.db --model /work/mithunpaul/fever/my_fork/fever-baselines/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --in-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.ns.pages.p1.jsonl --out-file /work/mithunpaul/fever/my_fork/fever-baselines/data/fever/dev.sentences.p5.s5.jsonl  --max-page 5 --max-sent 5 --mode dev --lmode INFO`


## nstructions from sheffield :might not be updated. use their instructions [page](https://github.com/sheffieldnlp/fever-baselines#evaluation)

# These are the various versions in the fact verification code development cycle (and what they do) at University of Arizona
# note, there must be only one version of this and preferably exists in the master branch

| Date of modification | name of the branch| git SHA| change made | New F1 score | New average Accuracy | New average Precision| Notes |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | 
| Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  |
