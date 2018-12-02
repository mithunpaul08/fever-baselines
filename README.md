
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


A sample command to run the training and dev together decomposable attention model, with Smart NER (replace tokens with NER tags but checking if they exists in the claim) looks like:

#`time ./runner_oracle_decomp_args.sh -m smartner_tr-fever -l log_smartner_tr-fever -r train  -c config/fever_nn_ora_sent_updateEmbeds.json`
`PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/smartner_tr-fever.tar.gz data/fever/dev.ns.pages.p1.jsonl --param_path config/fever_nn_ora_sent_updateEmbeds.json  --randomseed 123123 --slice 10`


Note: To run dev alone, comment out the lines corresponding to training in the shell script. You will have to still provide -m and -c command line inputs

Eg:`./runner_oracle_decomp_args.sh -m smartner_tr-fever -c config/fever_nn_ora_sent_updateEmbeds.json -n 23232 -slice 10`

To run the With handcrafted features + SVM pipe line (from fever baseline model) + SVM (our model): use `./runner_ir.sh`

note that: these shell scripts will run the whole training and testing on dev pipeline. If you want just pieces, comment out accordingly
    
note that :the IR part is in oracle mode.-i.e there is no Information retrieval being done on the fly. instead we rely on the gold data annotators found for ecah of the training data. However, do note that, the above statement is true only for classes SUPPORTS and REFUTES. For the class NOT ENOUGH INFO, there are two methods of retrieving evidences. via either using nearest neighbor or random. We are using the fever baseline's nearest neighbor methods


## version tracker is kept [here](https://github.com/mithunpaul08/fever-baselines/blob/master/versions.md)

## details of config file entries

`
"datasets_to_work_on": [
"fnc"
],
`

#### other options include: fever. Can add more than one like this: "fnc,fever" 
    
 `
"list_of_runs": [
"dev"
],
`       
### here other options include: "test/train/dev/annotation". 
- Can add more than one like this: "dev,annotation". 
- Note that it means, corresponding data set and corresponding runs. i.e if you add "fever,fnc" in datasets and "train,dev" on runs, it means, it will run train on fever and dev on fnc.
- Whenever you run dev, make sure you have `name_of_trained_model_to_use` filled up under `dev_partition_details` in the corresponding dataset details tag. Eg: `fever_dataset_details`
 
    