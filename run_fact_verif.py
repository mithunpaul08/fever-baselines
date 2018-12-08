from allennlp.common import Params
from allennlp.models import Model, archive_model, load_archive
from allennlp.data import Vocabulary, Dataset, DataIterator, DatasetReader, Tokenizer, TokenIndexer
import argparse
import sys,os
from src.rte.mithun.log import setup_custom_logger
from types import *
from src.scripts.rte.da.train_da import train_da
from src.scripts.rte.da.eval_da import eval_model
from src.scripts.rte.da.train_da import train_model_uofa_version
from src.scripts.rte.da.eval_da import convert_fnc_to_fever_and_annotate
from rte.parikh.reader_uofa import FEVERReaderUofa
from tqdm import tqdm
from rte.mithun.trainer import UofaTrainTest
from retrieval.fever_doc_db import FeverDocDB
from subprocess import call

"""takes a data set and a dictionary of features and generate features based on the requirement. 
EG: take claim evidence and create smartner based replaced text
Eg: take claim evidence and create feature vectors for word overlap
Parameters
    ----------   

        """

#todo: eventually when you merge hand crafted features + text based features, you will have to make both the functions return the same thing

def generate_features(zipped_annotated_data,feature,feature_details,reader,mithun_logger,objUofaTrainTest,dataset,length_data):
    mithun_logger.info(f"got inside generate_features")
    mithun_logger.info(f"value of feature  is:{feature}")
    mithun_logger.info(f"value of dataset  is:{dataset}")
    instances = []
    for index, (he, be, hl, bl, hw, bw, ht, hd, hfc) in enumerate(zipped_annotated_data):

        new_label =""
        label = hfc

        if(dataset == "fnc"):

            if  (label == "unrelated"):
                continue
            else:
                if (label == 'discuss'):
                    new_label = "NOT ENOUGH INFO"
                if (label == 'agree'):
                    new_label = "SUPPORTS"
                if (label == 'disagree'):
                    new_label = "REFUTES"

        else :
            new_label=label


        he_split = he.split(" ")
        be_split = be.split(" ")
        hl_split = hl.split(" ")
        bl_split = bl.split(" ")
        hw_split = hw.split(" ")
        bw_split = bw.split(" ")

        premise_ann=""
        hypothesis_ann=""

        if (feature=="plain_NER"):
            premise_ann, hypothesis_ann = objUofaTrainTest.convert_NER_form_per_sent_plain_NER(he_split, be_split, hl_split,
                                                                                               bl_split, hw_split, bw_split,mithun_logger)
        else:
            if (feature == "smart_NER"):
                premise_ann, hypothesis_ann, found_intersection = objUofaTrainTest.convert_SMARTNER_form_per_sent(he_split,
                                                                                                                      be_split,
                                                                                                                      hl_split,
                                                                                                                      bl_split,hw_split, bw_split)
        if(index %100==0):
            mithun_logger.info(f"value of old label is:{label}")
            mithun_logger.info(f"value of new label is:{new_label}")
            mithun_logger.info(f"value of claim before annotation is:{hw}")
            mithun_logger.info(f"value of evidence before anntoation is is:{bw}")
            mithun_logger.info(f"value of premise_ann is:{premise_ann}")
            mithun_logger.info(f"value of hypothesis_ann is:{hypothesis_ann}")

        mithun_logger.debug(f"value of old label is:{label}")
        mithun_logger.debug(f"value of new label is:{new_label}")
        mithun_logger.debug(f"value of claim before annotation is:{hw}")
        mithun_logger.debug(f"value of evidence before anntoation is is:{bw}")
        mithun_logger.debug(f"value of premise_ann is:{premise_ann}")
        mithun_logger.debug(f"value of hypothesis_ann is:{hypothesis_ann}")

        #todo: fixe me. not able to cleanly retrieve boolean values from the config file
        # person_c1 = feature_details.pop('person_c1', {})
        # lower_case_tokens= feature_details.pop('lower_case_tokens', {})
        # update_embeddings= feature_details.pop('update_embeddings', {})
        # assert type(person_c1) is str
        # assert type(lower_case_tokens) is bool
        # assert type(update_embeddings) is bool
        #
        # if(lower_case_tokens):
        #     premise_ann=premise_ann.lower(),
        #     hypothesis_ann=hypothesis_ann.lower()
        #     mithun_logger.debug(f"value of premise_ann after lower case token is:{premise_ann}")
        #     mithun_logger.debug(f"value of label after lower case token  is:{hypothesis_ann}")


        instances.append(reader.text_to_instance(premise_ann, hypothesis_ann, new_label))


    if len(instances)==0:
        mithun_logger.error("No instances were read from the given filepath {}. ""Is the path correct?")
        sys.exit(1)
    mithun_logger.info(f"type of instances is :{type(instances)}")
    return Dataset(instances)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--param_path',
                           type=str,
                           help='path to parameter file describing the model to be trained')
    parser.add_argument('-o', '--overrides',
                        type=str,
                        default="",
                        help='a HOCON structure used to override the experiment configuration')

    args = parser.parse_args()




    '''All of this must be done in this file run_fact_verify.py
    1.1  Get list of data sets
    1.2 get list of runs (eg: train,dev)
    1.3 for zip (eacha of data-run combination)
    
    Step2:
    
    - decide what kinda data it is eg: fnc or ever
    - extract corresponding data related details from config file Eg: path to annotated folder
    - find is it dev or train that must be run
    - if dev, extract trained model path
    - if train , nothing
    - create a logger
    
    
   
   - what kinda classifier to run?
   
    3. read data (with input/data folder path from 2.1)
    4. create features
    4.1 get corresponding details for features from config file
    4.2 create features (based on output from 4.1)
     
    
    8.1 call the corresponding function with input (features) and trained model (if applicable)- return results
    9. print results
    '''

    params = Params.from_file(args.param_path)
    uofa_params = params.pop('uofa_params', {})
    datasets_to_work_on = uofa_params.pop('datasets_to_work_on', {})
    list_of_runs = uofa_params.pop('list_of_runs', {})
    assert len(datasets_to_work_on) == len(list_of_runs)

    path_to_trained_models_folder = uofa_params.pop('path_to_trained_models_folder', {})
    cuda_device = uofa_params.pop('cuda_device', {})
    random_seed = uofa_params.pop('random_seed', {})
    assert type(path_to_trained_models_folder) is not Params
    assert type(cuda_device) is not Params
    assert type(random_seed) is not Params

    # step 2.1- create a logger
    logger_details = uofa_params.pop('logger_details', {})
    print(f"value of logger_details is {logger_details}")
    print(type(logger_details))
    assert type(logger_details) is Params
    logger_mode = logger_details.pop('logger_mode', {})
    assert type(logger_mode) is not Params
    mithun_logger = setup_custom_logger('root', logger_mode, "general_log.txt")

    #all one time used config values move outside for loop. This has to be done because the allennlp pop function clears out the dictionary if its read once.
    path_to_saved_db = uofa_params.pop("path_to_saved_db")
    # step 4 - generate features
    features = uofa_params.pop("features", {})
    assert type(features) is not Params
    type_of_classifier = uofa_params.pop("type_of_classifier", {})
    assert type(type_of_classifier) is  str
    name_of_trained_model_to_use = uofa_params.pop('name_of_trained_model_to_use', {})
    mithun_logger.info((f"value of name_of_trained_model_to_use is: {name_of_trained_model_to_use}"))
    assert type(name_of_trained_model_to_use) is str
    serialization_dir_base = uofa_params.pop("serialization_dir", {})
    assert type(name_of_trained_model_to_use) is str


    for (dataset, run_name) in (zip(datasets_to_work_on, list_of_runs)):

        #Step 2.2- get relevant config details from config file
        mithun_logger.info((f"value of dataset is: {dataset}"))
        mithun_logger.info((f"value of run_name is: {run_name}"))
        fds= dataset + "_dataset_details"
        mithun_logger.info((f"value of fds is: {fds}"))
        dataset_details = uofa_params.pop(fds, {})
        mithun_logger.info((f"value of dataset_details is: {dataset_details}") )
        assert type(dataset_details) is  Params
        frn= run_name + "_partition_details"
        mithun_logger.info((f"value of frn is: {frn}"))
        data_partition_details = dataset_details.pop(frn, {})
        mithun_logger.info((f"value of data_partition_details is: {data_partition_details}"))
        assert type(data_partition_details) is  Params
        path_to_pyproc_annotated_data_folder = data_partition_details.pop('path_to_pyproc_annotated_data_folder', {})
        mithun_logger.info(
            (f"value of path_to_pyproc_annotated_data_folder is: {path_to_pyproc_annotated_data_folder}"))
        assert type(path_to_pyproc_annotated_data_folder) is not Params
        slice_percent = data_partition_details.pop("slice_percent", {})
        mithun_logger.info(
            (f"value of slice_percent is: {slice_percent}"))
        assert type(slice_percent) is int



        serialization_dir= serialization_dir_base+dataset+"_"+run_name + "_"+str(slice_percent)
        bashCommand = "mkdir -p " + serialization_dir
        os.system(bashCommand)
        mithun_logger.info(
            (f"just finished creating a serialization_dir with path:{serialization_dir}"))
        sys.exit(1)

        # Step 2.6 - find is it dev or train that must be run
        # - if dev, extract trained model path
        # - if train , nothing
        # update: the feverdatareader we are using from the fever code needs the name of trained model. EVen for training. wtf..
        # update: so moved it to outside this for loop, since we are accessing it only once using uofa_params.pop anyway








        #step 3 -read data

        objUofaTrainTest = UofaTrainTest()

        if (run_name == "annotation" and dataset == "fnc"):
            path_to_trained_models=path_to_trained_models_folder+ name_of_trained_model_to_use
            convert_fnc_to_fever_and_annotate(FeverDocDB, path_to_trained_models,  mithun_logger,cuda_device,path_to_pyproc_annotated_data_folder)



        db = FeverDocDB(path_to_saved_db)
        archive = load_archive(path_to_trained_models_folder + name_of_trained_model_to_use, cuda_device)
        config = archive.config
        ds_params = config["dataset_reader"]
        model = archive.model
        model.eval()
        mithun_logger.info(f"going to initiate FEVERReaderUofa.")
        fever_reader = FEVERReaderUofa(db,
                             sentence_level=ds_params.pop("sentence_level", False),
                             wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                             claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                             token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})))

        cwd=os.getcwd()
        mithun_logger.info(f"going to start reading data.")
        zipped_annotated_data,length_data = fever_reader.read(mithun_logger, cwd+path_to_pyproc_annotated_data_folder)

        mithun_logger.info(f"done with reading data. going to generate features.")





        data = None
        for feature in features:
            # todo: right now there is only one feature, NER ONE, so you will get away with data inside this for loop. However, need to dynamically add features
            fdl= feature + "_details"
            mithun_logger.info(f"value of fdl is:{fdl}")
            mithun_logger.info(f"value of feature is:{feature}")
            feature_details=uofa_params.pop("fdl", {})

            data=generate_features(zipped_annotated_data, feature, feature_details, fever_reader, mithun_logger,objUofaTrainTest,dataset,length_data)

        if(type_of_classifier=="decomp_attention"):
            mithun_logger.info(f"found that the type_of_classifier is decomp attention")
            if(run_name== "train"):
                mithun_logger.info(f"found that the run_name is train. Going to get into  is train_model_uofa_version attention")
                train_model_uofa_version(params, cuda_device, serialization_dir, slice_percent , mithun_logger,data)
            else:
                if(run_name== "dev"):
                    eval_model(data,mithun_logger,path_to_trained_models_folder,name_of_trained_model_to_use,cuda_device)



