import argparse
import json
from multiprocessing.pool import ThreadPool
import tqdm
import os,sys
import logging
from common.util.log_helper import LogHelper
from tqdm import tqdm

from retrieval.top_n import TopNDocsTopNSents
from retrieval.fever_doc_db import FeverDocDB
from common.dataset.reader import JSONLineReader
from rte.riedel.data import FEVERGoldFormatter, FEVERLabelSchema
from retrieval.read_claims import uofa_training,uofa_testing,uofa_dev
from rte.mithun.log import setup_custom_logger
from fever.scorer import fever_score



def process_line(method,line):
    sents = method.get_sentences_for_claim(line["claim"])
    pages = list(set(map(lambda sent:sent[0],sents)))
    line["predicted_pages"] = pages
    line["predicted_sentences"] = sents
    return line


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_map_function(parallel):
    return p.imap_unordered if parallel else map

if __name__ == "__main__":
    #setup_custom_logger
    # LogHelper.setup()
    # logger = LogHelper.get_logger(__name__)
    logger = setup_custom_logger('root')
    logger.debug('main message')

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, help='drqa doc db file')
    parser.add_argument('--model', type=str, help='drqa index file')
    parser.add_argument('--in-file', type=str, help='input dataset')
    parser.add_argument('--out-file', type=str, help='path to save output dataset')
    parser.add_argument('--max-page',type=int)
    parser.add_argument('--max-sent',type=int)
    parser.add_argument('--parallel',type=str2bool,default=True)
    parser.add_argument('--mode', type=str, help='do training or testing' )
    parser.add_argument('--load_feat_vec', type=str2bool,default=False)
    parser.add_argument('--pred_file', type=str, help='path to save predictions',default="predictions.jsonl")


    args = parser.parse_args()

    db = FeverDocDB(args.db)
    jlr = JSONLineReader()
    formatter = FEVERGoldFormatter(set(), FEVERLabelSchema())

    method = TopNDocsTopNSents(db, args.max_page, args.max_sent, args.model)


    processed = dict()



    # #JUST GET CLAIMS FROM DEV OR TRAINING
    # with open(args.in_file,"r") as f, open(args.out_file, "w+") as out_file:
    #     lines = jlr.process(f)
    #     logger.info("Processing lines")
    #
    #
    #
    #     with ThreadPool() as p:
    #             for line in tqdm(get_map_function(args.parallel)(lambda line: process_line(method,line),lines), total=len(lines)):
    #                 #at this point the line thing has list of sentences it think is evidence for the given claim
    #                 line["predicted_pages"] = pages
    #                 line["predicted_sentences"] = sents
    #                 return line
    #                 processed[line["id"]] = line
    #
    #
    #
        # for line in lines:
        #         out_file.write(json.dumps(processed[line["id"]]) + "\n")

    #     logger.warning("Done, writing IR data to disk.")
    #
    #
    #
    #
    # with open(args.out_file,"r") as f:

    if(args.mode=="train"):
        uofa_training(args,jlr,method,logger)
    else:
        if(args.mode=="dev"):
            uofa_dev(args,jlr,method,logger)
            logger.info("Done, testing ")

        else:
            if(args.mode=="test"):
                uofa_testing(args,jlr,method,logger)
                logger.info("Done, testing ")

#                     {
#     "id": 78526,
#     "predicted_label": "REFUTES",
#     "predicted_evidence": [
#         ["Lorelai_Gilmore", 3]
#     ]
#https://github.com/sheffieldnlp/fever-scorer
#  }

