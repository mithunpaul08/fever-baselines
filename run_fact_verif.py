from allennlp.common import Params
import argparse
import sys,os
from src.rte.mithun.log import setup_custom_logger
from types import *
from src.scripts.rte.da.train_da import train_da
from src.scripts.rte.da.eval_da import eval_model
from rte.parikh.reader_uofa import FEVERReaderUofa
from tqdm import tqdm
from rte.mithun.trainer import UofaTrainTest


"""takes a data set and a dictionary of features and generate features based on the requirement. 
EG: take claim evidence and create smartner based replaced text
Eg: take claim evidence and create feature vectors for word overlap
Parameters
    ----------   

        """

#todo: eventually when you merge hand crafted features + text based features, you will have to make both the functions return the same thing

def generate_features(zipped_annotated_data,feature,feature_detail_dict,reader,mithun_logger):
    mithun_logger.debug(f"got inside generate_features")
    instances = []
    for he, be, hl, bl, hw, bw, ht, hd,label in zipped_annotated_data:
            #tqdm(,total=len(he), desc="reading annotated data"):


        label = str(label)
        mithun_logger.debug(f"value of label is:{label}")

        he_split = he.split(" ")
        be_split = be.split(" ")
        hl_split = hl.split(" ")
        bl_split = bl.split(" ")
        hw_split = hw.split(" ")
        bw_split = bw.split(" ")

        #
        # if not (label == "unrelated"):
        #
        #     if (label == 'discuss'):
        #         new_label = "NOT ENOUGH INFO"
        #     if (label == 'agree'):
        #         new_label = "SUPPORTS"
        #     if (label == 'disagree'):
        #         new_label = "REFUTES"
        premise_ann=""
        hypothesis_ann=""

        objUofaTrainTest = UofaTrainTest()


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
        mithun_logger.debug(f"value of premise_ann is:{premise_ann}")
        mithun_logger.debug(f"value of label is:{hypothesis_ann}")

        person_c1 = feature_detail_dict.pop('person_c1', {})
        lower_case_tokens= feature_detail_dict.pop('lower_case_tokens', {})
        update_embeddings= feature_detail_dict.pop('update_embeddings', {})
        assert type(person_c1) is not Params
        assert type(lower_case_tokens) is not Params
        assert type(update_embeddings) is not Params

        if(lower_case_tokens):
            premise_ann=premise_ann.lower(),
            hypothesis_ann=hypothesis_ann.lower()


        instances.append(reader.text_to_instance(premise_ann, hypothesis_ann, new_label))

    if len(instances)==0:
        raise ConfigurationError("No instances were read from the given filepath {}. "
                                 "Is the path correct?".format(file_path))
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






    #if annotate:annotation_on_the_fly(self, file_path, run_name, objUOFADataReader):


    '''All of this must be done in this file run_fact_verify.py
    1.1  Get list of data sets
    1.2 get list of runs (eg: train,dev)
    1.3 for zip (eacha of data-run combination)
    2.0 decide what kinda data it is eg: fnc or ever
    2.1 extract corresponding data related details from config file Eg: path to annotated folder
    
    2.1.1 create a logger
    
    3. read data (with input/data folder path from 2.1)
    4. create features
    4.1 get corresponding details for features from config file
    4.2 create features (based on output from 4.1)
     
    5. find is it dev or train that must be run
    6. if dev, extract trained model path
    7. if train , nothing
   
    8. what kinda classifier to run?
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

    # step 2.1.1
    logger_details = uofa_params.pop('logger_details', {})
    # print(f"value of logger_details is {logger_details}")
    # print(type(logger_details))
    assert type(logger_details) is Params
    logger_mode = logger_details.pop('logger_mode', {})
    assert type(logger_mode) is not Params

    mithun_logger = setup_custom_logger('root', logger_mode, "general_log.txt")


    for (dataset, run_name) in (zip(datasets_to_work_on, list_of_runs)):
        mithun_logger.debug(dataset)
        #Step 2
        fds= dataset + "_dataset_details"
        mithun_logger.debug(fds)
        dataset_details = uofa_params.pop(fds, {})
        assert type(dataset_details) is  Params
        frn= run_name + "_partition_details"
        data_partition_details = dataset_details.pop(frn, {})
        assert type(data_partition_details) is  Params
        path_to_pyproc_annotated_data_folder = data_partition_details.pop('path_to_pyproc_annotated_data_folder', {})
        assert type(path_to_pyproc_annotated_data_folder) is not Params



        log_file_base_name = logger_details.pop('log_file_base_name', {})
        assert type(log_file_base_name) is not Params

        # todo: this is a hack where we are reading the labels of fnc data from a separate labels only file.
        # However, this should have been written along with the pyproc annotated data, so that it can be r
        # ead back inside the generate features function, just like we do for fever data.

        all_labels=None
        if (dataset == "fnc"):
            mithun_logger.debug(f"value of dataset is:{dataset}")
            fds = dataset + "_dataset_details"
            mithun_logger.debug(f"value of dataset is:{dataset}")
            mithun_logger.debug(f"value of fdsfds is:{fds}")
            dataset_details = uofa_params.pop(fds, {})
            mithun_logger.debug(f"value of frn is:{frn}")
            data_partition_details = dataset_details.pop(frn, {})
            mithun_logger.debug(f"value of data_partition_details is:{data_partition_details}")
            #assert type(data_partition_details) is  Params
            path_to_pyproc_annotated_data_folder = data_partition_details.pop('path_to_pyproc_annotated_data_folder',
                                                                              {})
            #assert type(path_to_pyproc_annotated_data_folder) is not Params

            label_dev_file = data_partition_details.pop('label_dev_file', {})
            mithun_logger.debug(f"value of label_dev_file is:{label_dev_file}")
            #assert type(label_dev_file) is not Params
            label_folder = data_partition_details.pop('label_folder', {})
            mithun_logger.debug(f"value of label_folder is:{label_folder}")
            #assert type(label_folder) is not Params
            lbl_file = label_folder + label_dev_file
            mithun_logger.debug(f"value of lbl_file is:{lbl_file}")
            all_labels = objUofaTrainTest.read_csv_list(lbl_file)

        #step 3
        reader = FEVERReaderUofa()
        cwd=os.getcwd()
        zipped_annotated_data = reader.read(mithun_logger, cwd+path_to_pyproc_annotated_data_folder,all_labels)

        #step 4
        features = uofa_params.pop("features", {})
        assert type(features) is not Params





        #todo: right now there is only one feature, NER ONE, so you will get away with data inside this for loop. However, need to dynamically add features
        data = None
        for feature in features:
            fdl= feature + "_details"
            feature_details=uofa_params.pop("fdl", {})
            data=generate_features(zipped_annotated_data, feature, feature_details,reader,mithun_logger).instances



        name_of_trained_model_to_use=""

        #step 5
        if (run_name == "dev"):
            frn = run_name + "_partition_details"
            data_partition_details = dataset_details.pop(frn, {})
            assert type(data_partition_details) is not Params
            name_of_trained_model_to_use = data_partition_details.pop('name_of_trained_model_to_use',{})
            assert type(name_of_trained_model_to_use) is not Params



        type_of_classifier = uofa_params.pop("type_of_classifier", {})

        if(type_of_classifier=="decomp_attention"):
            if(run_name== "train"):
                train_da( ds, operation, logger_mode)
            if(run_name== "dev"):
                eval_model(data,mithun_logger,path_to_trained_models_folder,name_of_trained_model_to_use,cuda_device)



