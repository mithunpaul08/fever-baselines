
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

# All code can be run with the below command (but with different config file values)
`PYTHONPATH=src python run_fact_verif.py -p config/fever_nn_ora_sent_updateEmbeds.json`

# details of config file entries
- The name/path of the config file file is: config/fever_nn_ora_sent_updateEmbeds.json
- Almost always/mostly you will have to change only two entries `datasets_to_work_on` and `list_of_runs`
- With `datasets_to_work_on` you tell the machine which all data sets you want to work on. 
- Combined with `list_of_runs` it is an indication of what kind of process to run on what type of data set.

Eg: 
`
"datasets_to_work_on": [
"fnc"
],
`

- other options for value of datasets_to_work_on include: fever. Can add more than one like this: "fnc,fever" 

-

 `
"list_of_runs": [
"dev"
],
`       
- here other options include: "test/train/dev/annotation". 
- Can add more than one like this: "dev,test,annotation". 
- Note that it means, corresponding data set and corresponding runs. i.e if you add "fever,fnc" in datasets and "train,dev" on runs, it means, it will run train on fever and dev on fnc.
- Whenever you run dev, make sure you have `name_of_trained_model_to_use` filled up under `dev_partition_details` in the corresponding dataset details tag. Eg: `fever_dataset_details`
 
### Example:

So if you want to run just dev on fake news data set this is how your config will look like:
```
"datasets_to_work_on": [
       "fnc"
     ],
     "list_of_runs": [
       "dev"
     ],
```

### Example:

Instead if you want to first train the code on fever and then test on fnc,  your config will look like:
```
"datasets_to_work_on": [
       "fever",
       "fnc"
     ],
     "list_of_runs": [
       "train",
       "dev"
     ],
```

### Example:

If you want to annotated the fever data set with pyprocessors your config will look like:
```
"datasets_to_work_on": [
       "fever"
     ],
     "list_of_runs": [
       "annotation"
     ],

```

## version tracker is kept [here](https://github.com/mithunpaul08/fever-baselines/blob/master/versions.md)

