
# UOFA- Fact Extraction and VERification

# What is it?
This is a fact verification system that will be unlexicalized. The advantage of this system is that it is domain transferable. 
This is a follow up on the recently concluded Fact Verification(FEVER) challenge and workshop. Details can be found at the fever [homepage](http://fever.ai/) 



## Instructions to run the trained model on the FEVER dev data:

- `mkdir mithun_hw4`
- `cd  mithun_hw4`
- `git clone git@github.com:mithunpaul08/fever-baselines.git .`
- `git checkout v1.9`
- `mkdir  -p data/fever-data-ann`
- `conda env create -f environment.yml`
- `source activate hw4mithun`
- Download devhw4.zip from [this](https://drive.google.com/file/d/1rH5p_euolj2lmsbdHHUVA03rghGT7msb/view) link 
   - (curl or wget should work on based on your OS)
   - if curl or wget doesn't work please read [this](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive) solution
   - Please unzip so that a new folder called dev should be created as: data/fever-data-ann/dev Eg: `unzip devhw4.zip data/fever-data-ann/dev`
- `pip install -r requirements.txt`
    -(run export LANG=C.UTF-8 if installation of DrQA fails)
    - Also ignore incompatibility warnings please
- `bash scripts/download-processed-wiki.sh`
- `mkdir -p data/models`
- Download the trained model from [this] (https://drive.google.com/file/u/1/d/1D8syoDID3btYTlYt2zpnKrVZLLMGKu8n/view?usp=sharing)google drive link to the data/models folder
- `export CUDA_DEVICE=0`
    -Assuming you have a GPU, if you don't please do: export CUDA_DEVICE=1
- `PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db  data/fever/dev.ns.pages.p1.jsonl --param_path config/fever_nn_ora_sent_hw4.json --randomseed 1234 --slice 100`
    - Run this command in the home directory (the one which has requirements.txt) 


# Instructions to train  trained model on the FEVER dev data:


## Software Dependencies

code is built on top of sheffield's fever baseline. So we assume you have installed all the required documents they mention in their [readme file](https://github.com/sheffieldnlp/fever-baselines)

Apart from that you will need PyProcessors over Docker. After you have installed [Docker](https://www.docker.com/), do:


- `Docker pull myedibleenso/processors-server:latest`

- `docker run -d -e _JAVA_OPTIONS="-Xmx3G" -p 127.0.0.1:8886:8888 --name procserv myedibleenso/processors-server`

note: the docker run command is for the very first time you create this container. 

Second time onwards use: `docker start procserv`


#### In this project we are experimenting with fact verification but unlexicalized.As of Nov 2018 there are two main lines of development
1. With decomposable attention + hand crafted features of NER replacement
2. With handcrafted features + SVM


A sample command to run the training and dev together decomposable attention model, with Smart NER (replace tokens with NER tags but checking if they exists in the claim) looks like:

`time ./runner_oracle_decomp_args.sh -m smartner_tr-fever -l log_smartner_tr-fever -r train  -c config/fever_nn_ora_sent_updateEmbeds.json`
`PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/smartner_tr-fever.tar.gz data/fever/dev.ns.pages.p1.jsonl --param_path config/fever_nn_ora_sent_updateEmbeds.json  --randomseed 123123 --slice 10`


Note: To run dev alone, comment out the lines corresponding to training in the shell script. You will have to still provide -m and -c command line inputs

Eg:`./runner_oracle_decomp_args.sh -m smartner_tr-fever -c config/fever_nn_ora_sent_updateEmbeds.json -n 23232 -slice 10`

To run the with handcrafted features + SVM pipe line (from fever baseline model) + SVM (our model): 
 
 `./runner_ir.sh`

note that: these shell scripts will run the whole training and testing on dev pipeline. If you want just pieces, comment out accordingly
    
note that :the IR part is in oracle mode.-i.e there is no Information retrieval being done on the fly. instead we rely on the gold data annotators found for ecah of the training data. However, do note that, the above statement is true only for classes SUPPORTS and REFUTES. For the class NOT ENOUGH INFO, there are two methods of retrieving evidences. via either using nearest neighbor or random. We are using the fever baseline's nearest neighbor methods


## version tracker is kept [here](https://github.com/mithunpaul08/fever-baselines/blob/master/versions.md)