from allennlp.common import Params
import argparse
import sys
from src.rte.mithun.log import setup_custom_logger
from types import *
from src.scripts.rte.da.train_da import train_da
from src.scripts.rte.da.eval_da import eval_model
from rte.parikh.reader_uofa import FEVERReaderUofa


"""takes a data set and a dictionary of features and generate features based on the requirement. 
EG: take claim evidence and create smartner based replaced text
Eg: take claim evidence and create feature vectors for word overlap
Parameters
    ----------   

        """

#todo: eventually when you merge hand crafted features + text based features, you will have to make both the functions return the same thing

def generate_features(zipped_annotated_data,feature,feature_detail_dict):
    instances = []
    for he, be, hl, bl, hw, bw, ht, hd, hfc in \
            tq(zipped_annotated_data,total=len(heads_complete_annotation), desc="reading annotated data"):

        premise = heads_words
        hypothesis = bodies_words

        label = str(hfc)

        instances.append(self.text_to_instance(premise, hypothesis, label))

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
        person_c1 = feature_detail_dict.pop('person_c1', {})
        lower_case_tokens= feature_detail_dict.pop('lower_case_tokens', {})
        update_embeddings= feature_detail_dict.pop('update_embeddings', {})
        assert type(person_c1) is not Params
        assert type(lower_case_tokens) is not Params
        assert type(update_embeddings) is not Params

        if(lower_case_tokens):
            premise_ann=premise_ann.lower(),
            hypothesis_ann=hypothesis_ann.lower()


        instances.append(text_to_instance(premise_ann, hypothesis_ann, new_label))

    if len(instances)==0:
        raise ConfigurationError("No instances were read from the given filepath {}. "
                                 "Is the path correct?".format(file_path))
    return Dataset(instances)

def text_to_instance(premise: str,
                     hypothesis: str,
                     label: str = None) -> Instance:

    fields : Dict[str, Field] = {}
    premise_tokens = self._wiki_tokenizer.tokenize(premise) if premise is not None else None
    hypothesis_tokens = self._claim_tokenizer.tokenize(hypothesis)
    fields['premise'] = TextField(premise_tokens, self._token_indexers) if premise is not None else None
    fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
    if label is not None:
        fields['label'] = LabelField(label)
    return Instance(fields)


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




    for (d,r) in (zip(datasets_to_work_on,list_of_runs)):

        #Step 2
        fds=d+"_dataset_details"
        dataset_details = uofa_params.pop(fds, {})
        assert type(dataset_details) is not Params
        frn=r+"_partition_details"
        data_partition_details = dataset_details.pop(frn, {})
        assert type(data_partition_details) is not Params
        path_to_pyproc_annotated_data_folder = data_partition_details.pop('path_to_pyproc_annotated_data_folder', {})
        assert type(path_to_pyproc_annotated_data_folder) is not Params

        #step 2.1.1
        logger_details = uofa_params.pop('logger_details', {})
        assert type(logger_details) is not Params
        logger_mode = logger_details.pop('logger_mode', {})
        assert type(logger_mode) is not Params

        log_file_base_name = logger_details.pop('log_file_base_name', {})
        assert type(log_file_base_name) is not Params
        log_file_name = d+"_"+r+"_feverlog.txt" +  "_" + str(random_seed)
        mithun_logger = setup_custom_logger('root', debug_mode, log_file_name)

        #step 3
        reader = FEVERReaderUofa()
        zipped_annotated_data = reader.read(mithun_logger, path_to_pyproc_annotated_data_folder).instances

        #step 4
        features = uofa_params.pop("features", {})
        assert type(features) is not Params

        #todo: right now there is only one feature, NER ONE, so you will get away with data inside this for loop. However, need to dynamically add features
        data = None
        for feature in features:
            fdl= feature + "_details"
            feature_details=uofa_params.pop("fdl", {})
            data=generate_features(zipped_annotated_data, feature, feature_details)


        #step 5
        if (r == "dev"):
            frn = r + "_partition_details"
            data_partition_details = dataset_details.pop(frn, {})
            assert type(data_partition_details) is not Params
            name_of_trained_model_to_use = data_partition_details.pop('name_of_trained_model_to_use',{})
            assert type(name_of_trained_model_to_use) is not Params


        type_of_classifier = uofa_params.pop("type_of_classifier", {})

        if(type_of_classifier=="decomp_attention"):
            if(r=="train"):
                train_da(general_log, ds, operation, logger_mode)
            if(r=="dev"):
                eval_model(data,mithun_logger,path_to_trained_models_folder,name_of_trained_model_to_use)


