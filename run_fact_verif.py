from allennlp.common import Params
from allennlp.models import Model, archive_model, load_archive
from allennlp.data import Vocabulary, Dataset, DataIterator, DatasetReader, Tokenizer, TokenIndexer
import argparse
import sys,os
from src.rte.mithun.log import setup_custom_logger
from types import *
from src.scripts.rte.da.train_da import train_da
from src.scripts.rte.da.eval_da import eval_model
from rte.parikh.reader_uofa import FEVERReaderUofa
from tqdm import tqdm
from rte.mithun.trainer import UofaTrainTest
from retrieval.fever_doc_db import FeverDocDB


"""takes a data set and a dictionary of features and generate features based on the requirement. 
EG: take claim evidence and create smartner based replaced text
Eg: take claim evidence and create feature vectors for word overlap
Parameters
    ----------   

        """

#todo: eventually when you merge hand crafted features + text based features, you will have to make both the functions return the same thing

def generate_features(zipped_annotated_data,feature,feature_details,reader,mithun_logger,objUofaTrainTest):
    mithun_logger.debug(f"got inside generate_features")
    instances = []
    for index, (he, be, hl, bl, hw, bw, ht, hd,hfc,label) in enumerate(zipped_annotated_data):
            #tqdm(,total=len(he), desc="reading annotated data"):

        new_label =""
        label = str(label)
        mithun_logger.debug(f"value of label is:{label}")

        he_split = he.split(" ")
        be_split = be.split(" ")
        hl_split = hl.split(" ")
        bl_split = bl.split(" ")
        hw_split = hw.split(" ")
        bw_split = bw.split(" ")

        if not (label == "unrelated"):

            if (label == 'discuss'):
                new_label = "NOT ENOUGH INFO"
            if (label == 'agree'):
                new_label = "SUPPORTS"
            if (label == 'disagree'):
                new_label = "REFUTES"

            mithun_logger.debug(f"value of new_label is:{new_label}")

            premise_ann=""
            hypothesis_ann=""



            if (feature=="plain_NER"):
                premise_ann, hypothesis_ann = objUofaTrainTest.convert_NER_form_per_sent_plain_NER(he_split, be_split, hl_split,
                                                                                                   bl_split, hw_split, bw_split)
            else:
                if (feature == "smart_NER"):
                    premise_ann, hypothesis_ann, found_intersection = objUofaTrainTest.convert_SMARTNER_form_per_sent(he_split,
                                                                                                                          be_split,
                                                                                                                          hl_split,
                                                                                                                          bl_split,
                                                                                                                          hw_split,
                                                                                                                          bw_split)
            if (index%100) == 0:
                mithun_logger.info(f"value of premise_ann is:{premise_ann}")
                mithun_logger.info(f"value of hypothesis_ann is:{hypothesis_ann}")


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
        raise ConfigurationError("No instances were read from the given filepath {}. "
                                 "Is the path correct?")
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

    for (dataset, run_name) in (zip(datasets_to_work_on, list_of_runs)):
        # step 2.1- create a logger
        logger_details = uofa_params.pop('logger_details', {})
        # print(f"value of logger_details is {logger_details}")
        # print(type(logger_details))
        assert type(logger_details) is Params
        logger_mode = logger_details.pop('logger_mode', {})
        assert type(logger_mode) is not Params
        mithun_logger = setup_custom_logger('root', logger_mode, "general_log.txt")

        #Step 2.2- get relevant config details from config file
        fds= dataset + "_dataset_details"
        mithun_logger.debug(fds)
        dataset_details = uofa_params.pop(fds, {})
        assert type(dataset_details) is  Params
        frn= run_name + "_partition_details"
        data_partition_details = dataset_details.pop(frn, {})
        assert type(data_partition_details) is  Params
        path_to_pyproc_annotated_data_folder = data_partition_details.pop('path_to_pyproc_annotated_data_folder', {})
        assert type(path_to_pyproc_annotated_data_folder) is not Params





        # Step 2.6 - find is it dev or train that must be run
        # - if dev, extract trained model path
        # - if train , nothing

        name_of_trained_model_to_use = ""

        if (run_name == "dev"):
            name_of_trained_model_to_use = data_partition_details.pop('name_of_trained_model_to_use', {})
            assert type(name_of_trained_model_to_use) is str






        #step 3 -read data
        # todo: this is a hack where we are reading the labels of fnc data from a separate labels only file.
        # However, this should have been written along with the pyproc annotated data, so that it can be r
        # ead back inside the generate features function, just like we do for fever data.
        all_labels=None
        objUofaTrainTest = UofaTrainTest()
        if (dataset == "fnc"):
            label_dev_file = data_partition_details.pop('label_dev_file', {})
            mithun_logger.debug(f"value of label_dev_file is:{label_dev_file}")
            assert type(label_dev_file) is not Params
            label_folder = data_partition_details.pop('label_folder', {})
            assert type(label_folder) is str
            mithun_logger.debug(f"value of label_folder is:{label_folder}")
            assert type(label_dev_file) is str
            lbl_file = os.getcwd()+label_folder + label_dev_file
            mithun_logger.debug(f"value of lbl_file is:{lbl_file}")
            assert type(lbl_file) is str
            all_labels = objUofaTrainTest.read_csv_list(lbl_file)

        path_to_saved_db = uofa_params.pop("path_to_saved_db")
        db = FeverDocDB(path_to_saved_db)
        archive = load_archive(path_to_trained_models_folder + name_of_trained_model_to_use, cuda_device)
        config = archive.config
        ds_params = config["dataset_reader"]
        model = archive.model
        model.eval()
        fever_reader = FEVERReaderUofa(db,
                             sentence_level=ds_params.pop("sentence_level", False),
                             wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                             claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                             token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})))

        cwd=os.getcwd()
        zipped_annotated_data = fever_reader.read(mithun_logger, cwd+path_to_pyproc_annotated_data_folder,all_labels)

        mithun_logger.debug(f"done with reading data. going to generate features")

        #step 4 - generate features
        features = uofa_params.pop("features", {})
        assert type(features) is not Params

        data = None
        for feature in features:
            # todo: right now there is only one feature, NER ONE, so you will get away with data inside this for loop. However, need to dynamically add features
            fdl= feature + "_details"
            mithun_logger.debug(f"value of fdl is:{fdl}")
            mithun_logger.debug(f"value of feature is:{feature}")
            feature_details=uofa_params.pop("fdl", {})

            data=generate_features(zipped_annotated_data, feature, feature_details, fever_reader, mithun_logger,objUofaTrainTest).instances







        type_of_classifier = uofa_params.pop("type_of_classifier", {})

        if(type_of_classifier=="decomp_attention"):
            if(run_name== "train"):
                train_da( ds, operation, logger_mode)
            if(run_name== "dev"):
                eval_model(data,mithun_logger,path_to_trained_models_folder,name_of_trained_model_to_use,cuda_device)



