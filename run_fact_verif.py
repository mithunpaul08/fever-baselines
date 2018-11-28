from allennlp.common import Params
import argparse
import sys
from src.rte.mithun.log import setup_custom_logger
from types import *

if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument('--param_path',
                           type=str,
                           help='path to parameter file describing the model to be trained')

    args = parser.parse_args()
    params = Params.from_file(args.param_path)
    uofa_params = params.pop('uofa_params', {})

    datasets_to_work_on = uofa_params.pop('datasets_to_work_on', {})
    logger_details = uofa_params.pop('logger_details', {})
    logger_mode = logger_details.pop('logger_mode', {})
    log_file_base_name = logger_details.pop('log_file_base_name', {})

    assert type(log_file_base_name) is not Params

    # first find the list of data sets we will be working on. eg: fever, fnc
    fever_dataset_details = uofa_params.pop('fever_dataset_details', {})
    list_of_runs = uofa_params.pop('list_of_runs', {})

    dev_partition_details = fever_dataset_details.pop('dev_partition_details', {})
    name_of_trained_model_to_use = dev_partition_details.pop('name_of_trained_model_to_use', {})
    path_to_pyproc_annotated_data_folder = dev_partition_details.pop('path_to_pyproc_annotated_data_folder', {})

    print(f"value of dataset_to_work_on:{datasets_to_work_on}")
    print(f"value of name_of_trained_model_to_use:{name_of_trained_model_to_use}")
    print(f"value of list_of_runs:{list_of_runs}")

    print(path_to_pyproc_annotated_data_folder)




    assert len(datasets_to_work_on) == len(list_of_runs)

    #for each of this data set, what should we doing in it, eg: train, dev-i.e take the first data set, run train on it, and  take the second data set and run dev on it.
    for index,ds in enumerate(datasets_to_work_on):
        operation=list_of_runs[index]
        logfile_full_name=log_file_base_name+"_"+ds+"_"+operation+".log"
        print(f"value of logfile_full_name:{logfile_full_name}")
        mithun_logger = setup_custom_logger('root', logger_mode, logfile_full_name)
        mithun_logger.debug("going to run {operation} on dataset {ds}")

