from allennlp.common import Params
import argparse
import sys
from src.rte.mithun.log import setup_custom_logger
from types import *
from src.scripts.rte.da.train_da import train_da
from src.scripts.rte.da.eval_da import eval_da
from rte.parikh.reader_uofa import FEVERReaderUofa





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
    8.1 call the corresponding function with input  (features) and data model (if applicable)- return results
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




    for (d,r) in (zip(datasets_to_work_on,list_of_runs)):

        #Step 2
        fds=d+"_dataset_details"
        dataset_details = uofa_params.pop(fds, {})
        frn=r+"_partition_details"
        data_partition_details = dataset_details.pop(frn, {})
        path_to_pyproc_annotated_data_folder = data_partition_details.pop('path_to_pyproc_annotated_data_folder', {})

        #step 2.1.1
        logger_details = uofa_params.pop('logger_details', {})
        logger_mode = logger_details.pop('logger_mode', {})
        log_file_base_name = logger_details.pop('log_file_base_name', {})
        assert type(log_file_base_name) is not Params
        log_file_name = d+"_"+r+"_feverlog.txt" +  "_" + str(random_seed)
        mithun_logger = setup_custom_logger('root', debug_mode, log_file_name)

        #step 3
        reader = FEVERReaderUofa()
        data = reader.read(mithun_logger, path_to_pyproc_annotated_data_folder).instances


    #     name_of_trained_model_to_use = dev_partition_details.pop('name_of_trained_model_to_use', {})
    #     path_to_pyproc_annotated_data_folder = dev_partition_details.pop('path_to_pyproc_annotated_data_folder', {})
    #     debug_mode = uofa_params.pop('debug_mode', {})
    #
    #     path_to_fnc_annotated_data = dev_partition_details.pop('path_to_pyproc_annotated_data_folder', {})
    #
    #     dev_partition_details = fever_dataset_details.pop('dev_partition_details', {})
    #     name_of_trained_model_to_use = dev_partition_details.pop('name_of_trained_model_to_use', {})
    #
    #
    #     print(f"value of dataset_to_work_on:{datasets_to_work_on}")
    #     print(f"value of name_of_trained_model_to_use:{name_of_trained_model_to_use}")
    #     print(f"value of list_of_runs:{list_of_runs}")
    #
    #     print(path_to_pyproc_annotated_data_folder)
    #
    #
    #
    #
    #
    # fever_dataset_details = uofa_params.pop('fever_dataset_details', {})
    #
    #
    #
    # operation = list_of_runs[index]
    #
    # #for each of this data set, what should we doing in it, eg: train, dev-i.e take the first data set, run train on it, and  take the second data set and run dev on it.
    # for index,ds in enumerate(datasets_to_work_on):
    #
    #     logfile_full_name=log_file_base_name+"_"+ds+"_"+operation+".log"
    #     print(f"value of logfile_full_name:{logfile_full_name}")
    #     general_log = setup_custom_logger('root', logger_mode, logfile_full_name)
    #     general_log.debug(f"going to run {operation} on dataset {ds}")
    #
    #     if()
    #
    #     if operation=="train" :
    #         train_da(general_log, ds, operation,logger_mode)
    #     else:
    #         if operation == "dev":
    #             eval_da(ds,args,operation)


