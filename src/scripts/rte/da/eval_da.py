import os
import os,csv,sys
from copy import deepcopy
from typing import List, Union, Dict, Any
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from allennlp.common import Params
from allennlp.common.tee_logger import TeeLogger
#from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary, Dataset, DataIterator, DatasetReader, Tokenizer, TokenIndexer
from allennlp.models import Model, archive_model, load_archive
from allennlp.service.predictors import Predictor
from allennlp.training import Trainer
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from rte.parikh.reader import FEVERReader
from rte.mithun.read_fake_news_data import load_fever_DataSet
from tqdm import tqdm

import argparse
import logging
import sys
import json
import numpy as np
from sklearn.externals import joblib
from collections import defaultdict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def eval_model(db: FeverDocDB, args) -> Model:
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)

    config = archive.config
    ds_params = config["dataset_reader"]

    model = archive.model
    model.eval()

    reader = FEVERReader(db,
                                 sentence_level=ds_params.pop("sentence_level",False),
                                 wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                                 claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                                 token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})))

    logger.info("Reading training data from %s", args.in_file)

    # do annotation on the fly  using pyprocessors. i.e creating NER tags, POS Tags etcThis takes along time.
    #  so almost always we do it only once, and load it from disk . Hence do_annotation_live = False
    do_annotation_live = False
    data = reader.read(args.in_file,"dev",do_annotation_live).instances
    joblib.dump(data, "fever_dev_dataset_format.pkl")

    actual = []

    predicted = []

    if args.log is not None:
        f = open(args.log,"w+")
    if_ctr, else_ctr = 0, 0
    pred_dict = defaultdict(int)

    for item in tqdm(data):
        if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
            cls = "NOT ENOUGH INFO"
            if_ctr += 1
        else:
            else_ctr += 1

            prediction = model.forward_on_instance(item, args.cuda_device)
            cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]


        if "label" in item.fields:
            actual.append(item.fields["label"].label)
        predicted.append(cls)
        pred_dict[cls] += 1

        if args.log is not None:
            if "label" in item.fields:
                f.write(json.dumps({"actual":item.fields["label"].label,"predicted":cls})+"\n")
            else:
                f.write(json.dumps({"predicted":cls})+"\n")
    # print(f'if_ctr = {if_ctr}')
    # print(f'else_ctr = {else_ctr}')
    # print(f'pred_dict = {pred_dict}')


    if args.log is not None:
        f.close()


    if len(actual) > 0:
        print(accuracy_score(actual, predicted))
        print(classification_report(actual, predicted))
        print(confusion_matrix(actual, predicted))

    return model

def eval_model_fnc(db: FeverDocDB, args) -> Model:
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)

    config = archive.config
    ds_params = config["dataset_reader"]

    model = archive.model
    model.eval()


    logger.info("Reading training data from %s", args.in_file)

    reader = FEVERReader(db,
                         sentence_level=ds_params.pop("sentence_level", False),
                         wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                         claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                         token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})))

    # do annotation on the fly  using pyprocessors. i.e creating NER tags, POS Tags etcThis takes along time.
    #  so almost always we do it only once, and load it from disk . Hence do_annotation_live = False
    do_annotation_live = False
    data = reader.read_annotated_fnc_and_do_ner_replacement(args.in_file, "dev", do_annotation_live).instances
    joblib.dump(data, "fever_dev_dataset_format.pkl")
    #
    ###################end of running model and saving

    path=os.getcwd()

    data=joblib.load(path+"fever_dev_dataset_format")

    actual = []

    predicted = []

    if args.log is not None:
        f = open(args.log,"w+")
    if_ctr, else_ctr = 0, 0
    pred_dict = defaultdict(int)

    for item in tqdm(data):
        if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
            cls = "NOT ENOUGH INFO"
            if_ctr += 1
        else:
            else_ctr += 1

            prediction = model.forward_on_instance(item, args.cuda_device)
            cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]


        if "label" in item.fields:
            actual.append(item.fields["label"].label)
        predicted.append(cls)
        pred_dict[cls] += 1

        if args.log is not None:
            if "label" in item.fields:
                f.write(json.dumps({"actual":item.fields["label"].label,"predicted":cls})+"\n")
            else:
                f.write(json.dumps({"predicted":cls})+"\n")
    print(f'if_ctr = {if_ctr}')
    print(f'else_ctr = {else_ctr}')
    print(f'pred_dict = {pred_dict}')


    if args.log is not None:
        f.close()


    if len(actual) > 0:
        print(accuracy_score(actual, predicted))
        print(classification_report(actual, predicted))
        print(confusion_matrix(actual, predicted))

    return model


def convert_fnc_to_fever_and_annotate(db: FeverDocDB, args, logger) -> Model:
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)
    config = archive.config
    ds_params = config["dataset_reader"]
    model = archive.model
    model.eval()


    cwd = os.getcwd()
    fnc_data_set = load_fever_DataSet()

    #to annotate with pyprocessors- comment this part out if you are doing just evaluation
    # stances,articles= fnc_data_set.read_parent(cwd, "train_bodies.csv", "train_stances_csc483583.csv")
    #
    # dict_articles = {}
    # # copy all bodies into a dictionary
    # for article in articles:
    #     dict_articles[int(article['Body ID'])] = article['articleBody']
    #
    # load_fever_DataSet.annotate_fnc(cwd, stances,dict_articles,logger)
    # print("done with annotation. going to exit")
    # sys.exit(1)

    ###########end of pyprocessors annotation

    data=reader.read_fnc(fnc_data_set).instances



    actual = []
    predicted = []

    if args.log is not None:
        f = open(args.log,"w+")

    for item in data:
        if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
            cls = "NOT ENOUGH INFO"
        else:
            prediction = model.forward_on_instance(item, args.cuda_device)
            cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]

        actual.append(item.fields["label"].label)
        predicted.append(cls)

        if args.log is not None:
            f.write(json.dumps({"actual":item.fields["label"].label,"predicted":cls})+"\n")

    if args.log is not None:
        f.close()

    print(accuracy_score(actual, predicted))
    print(classification_report(actual, predicted))
    print(confusion_matrix(actual, predicted))

    return model



if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    LogHelper.get_logger(__name__)


    parser = argparse.ArgumentParser()

    parser.add_argument('db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('archive_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('in_file', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--log', required=False, default=None,  type=str, help='/path/to/saved/db.db')

    parser.add_argument("--cuda-device", type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')



    args = parser.parse_args()
    db = FeverDocDB(args.db)

    #this will ideally be used only once. i.e when you convert the fake news data into fever format and annotate it with pyprocessors
    #convert_fnc_to_fever_and_annotate(db, args, logger)
    eval_model_fnc(db,args)
   

    #eval_model(db,args)


