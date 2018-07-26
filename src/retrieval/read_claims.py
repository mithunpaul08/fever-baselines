from common.util.log_helper import LogHelper
from rte.mithun.ds import indiv_headline_body
from processors import ProcessorsBaseAPI
from tqdm import tqdm
from processors import Document
import logging
from rte.mithun.trainer import read_json_create_feat_vec,do_training,do_testing,load_model
import numpy as np
import os,sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
ann_head_tr = "ann_head_tr.json"
ann_body_tr = "ann_body_tr.json"
API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
logger=None
load_ann_corpus=True
data_folder_dev="/data/fever/"
from sklearn.externals import joblib
predicted_results="predicted_results.pkl"

#for each claim, get the evidence sentences, annotate and write to disk
def read_claims_annotate(args,jlr,logger,method):
    # try:
    #     os.remove(ann_head_tr)
    #     os.remove(ann_body_tr)
    #
    # except OSError:
    #     logger.error("not able to find file")

    logger.debug("inside read_claims_annotate")
    logger.debug("name of out file is:"+str(args.out_file))
    #the outfile from evidence prediction/IR phase becomes the in file/ file which contains all evidences
    with open(args.out_file,"r") as f:
        logger.debug("inside open with:")
        all_claims = jlr.process(f)
        obj_all_heads_bodies=[]
        ver_count=0
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_claim_ev:"):
            logger.debug("entire claim_full is:")
            logger.debug(claim_full)
            claim=claim_full["claim"]
            logger.debug("just claim alone is:")
            logger.debug(claim)
            x = indiv_headline_body()
            evidences=claim_full["evidence"]
            label=claim_full["label"]
            if not (label=="NOT ENOUGH INFO"):
                ver_count=ver_count+1
                logger.debug("length of evidences for this claim_full  is:" + str(len(evidences)))
                logger.debug("length of evidences for this claim_full  is:" + str(len(evidences[0])))
                ev_claim=[]
                for evidence in evidences[0]:
                    t=evidence[2]
                    l=evidence[3]
                    logger.debug(t)
                    logger.debug(l)
                    sent=method.get_sentences_given_claim(t,logger,l)
                    ev_claim.append(sent)
                all_evidences=' '.join(ev_claim)
                annotate_and_save_doc(claim, all_evidences,index, API, ann_head_tr, ann_body_tr, logger)

        return obj_all_heads_bodies


def read_test_data_annotate(args,jlr,logger,method):
    # try:
    #     os.remove(ann_head_tr)
    #     os.remove(ann_body_tr)
    #
    # except OSError:
    #     logger.error("not able to find file")


    logger.debug("inside read_claims_annotate")
    logger.debug("name of out file is:"+str(args.out_file))
    #the outfile from evidence prediction/IR phase becomes the in file/ file which contains all evidences
    cwd=os.getcwd()
    path =cwd+"/"+args.out_file
    logger.debug("path is:"+str(path))
    with open(path,"r") as f:
        logging.debug("inside read_json")
        l = []
        counter=0

        for eachline in (f):
            logging.debug(eachline)
            claim_full = json.loads(eachline)
            claim=claim_full["claim"]
            id=claim_full["id"]
            logger.debug("just claim alone is:")
            logger.debug(claim)
            predicted_pages=claim_full["predicted_pages"]
            predicted_sentences=claim_full["predicted_sentences"]
            logger.debug("predicted_sentences:" + str(predicted_sentences))
            logger.debug("predicted_pages:" + str(predicted_pages))
            ev_claim=[]
            for x in predicted_sentences:
                page=x[0]
                line_no=x[1]
                logger.debug("page is:" + str(page))
                logger.debug("line_no is:" + str(line_no))
                sent=method.get_sentences_given_claim(page,logger,line_no)
                logger.debug("evidences for this claim_full  is:" + str(sent))
                ev_claim.append(sent)
            all_evidences=' '.join(ev_claim)
            annotate_and_save_doc(claim, all_evidences,id, API, ann_head_tr, ann_body_tr, logger)

        return

def print_cv(combined_vector,gold_labels_tr):
    logging.debug(gold_labels_tr.shape)
    logging.debug(combined_vector.shape)
    x= np.column_stack([gold_labels_tr,combined_vector])
    np.savetxt("cv.csv", x, delimiter=",")
    sys.exit(1)


def uofa_training(args,jlr,method,logger):
    logger.warning("got inside uofatraining")

    #this code annotates the given file using pyprocessors. Run it only once in its lifetime.
    tr_data=read_test_data_annotate(args,jlr,logger,method)
    logger.info(
        "Finished read_claims_annotate")
    sys.exit(1)

    gold_labels_tr = get_gold_labels(args, jlr)
    logging.info("number of rows in label list is is:" + str(len(gold_labels_tr)))
    combined_vector = read_json_create_feat_vec(load_ann_corpus,args)

    logging.warning("done with generating feature vectors. Model training next")
    logging.info("gold_labels_tr is:" + str((gold_labels_tr)))
    do_training(combined_vector, gold_labels_tr)
    logging.warning("done with training. going to exit")
    sys.exit(1)

def uofa_dev(args, jlr, method, logger):
    logger.warning("got inside uofa_testing")
    gold_labels = get_gold_labels(args, jlr)
    logging.info("number of rows in label list is is:" + str(len(gold_labels)))
    combined_vector= read_json_create_feat_vec(load_ann_corpus,args)
    #print_cv(combined_vector, gold_labels)
    logging.warning("done with generating feature vectors. Model loading and predicting next")
    trained_model=load_model()
    logging.debug("weights:")
    #logging.debug(trained_model.coef_ )
    pred=do_testing(combined_vector,trained_model)
    logging.debug(str(pred))
    logging.debug("and golden labels are:")
    logging.debug(str(gold_labels))
    logging.warning("done testing. and the accuracy is:")
    acc=accuracy_score(gold_labels, pred)*100
    logging.warning(str(acc)+"%")
    logging.debug(classification_report(gold_labels, pred))
    logging.debug(confusion_matrix(gold_labels, pred))

    # get number of support vectors for each class
    #logging.debug(trained_model.n_support_)
    logging.info("done with testing. going to exit")
    sys.exit(1)


def uofa_testing(args, jlr, method, logger):
    logger.warning("got inside uofa_testing")
    combined_vector= read_json_create_feat_vec(load_ann_corpus,args)
    logging.warning("done with generating feature vectors. Model loading and predicting next")
    trained_model=load_model()
    logging.debug("weights:")
    pred=do_testing(combined_vector,trained_model)
    write_pred_str_disk(args,jlr,pred)
    logging.debug(str(pred))
    logging.info("done with testing. going to exit")
    sys.exit(1)

def annotate_save_quit(test_data,logger):

    for i, d in tqdm(enumerate(test_data), total=len(test_data),desc="annotate_json:"):
        annotate_and_save_doc(d, i, API, ann_head_tr, ann_body_tr,logger)


    sys.exit(1)

def read_json_alllines(json_file):
    l=[]
    with open(json_file) as f:
            for eachline in (f):
                d = json.loads(eachline)
                l.append(d)

    return l

#load predictions, convert it based on label and write it as string.
def write_pred_str_disk(args,jlr,pred):
    logging.debug("here1"+str(args.out_file))
    final_predictions=[]
    #pred=joblib.load(predicted_results)
    with open(args.out_file,"r") as f:
        ir = jlr.process(f)
        logging.debug("here2"+str(len(ir)))

        # for index,q in enumerate(ir):
        #     logging.debug("here3")
        #
        #     line=dict()
        #     label="not enough info"
        #     if(index%2 ==0):
        #         logging.debug("here4")
        #         label="supports"
        #     else:
        #         label="refutes"
        #
        #     line["id"]=q["id"]
        #     line["predicted_label"]=label
        #     line["predicted_evidence"]=q["predicted_sentences"]
        #     logging.debug(q["predicted_sentences"])
        #     logging.debug(q["id"])
        #     logging.debug(label)
        #
        #     final_predictions.append(line)

        for p,q in zip(pred,ir):
            line=dict()
            logger.debug("p")
            label="not enough info"
            if(p==0):
                label="supports"
            else:
                if(p==1):
                    label="refutes"

            line["id"]=q["id"]
            line["predicted_label"]=label
            line["predicted_evidence"]=q["predicted_sentences"]

            final_predictions.append(line)

    logging.info(len(final_predictions))
    with open(args.pred_file, "w+") as out_file:
        for x in final_predictions:

            out_file.write(json.dumps(x)+"\n")
    return


def annotate_and_save_doc(headline,body, index, API, json_file_tr_annotated_headline,json_file_tr_annotated_body,
                          logger):
    logger.debug("got inside annotate_and_save_doc")
    logger.debug("headline:"+headline)
    logger.debug("body:" + body)
    doc1 = API.fastnlp.annotate(headline)
    doc1.id=index
    with open(json_file_tr_annotated_headline, "a") as out:
      out.write(doc1.to_JSON())
      out.write("\n")


    doc2 = API.fastnlp.annotate(body)
    logger.debug(doc2)
    doc2.id = index

    with open(json_file_tr_annotated_body, "a") as out:
          out.write(doc2.to_JSON())
          out.write("\n")

    return

def get_gold_labels(args,jlr):
    labels = np.array([[]])

    with open(args.in_file,"r") as f, open(args.out_file, "w+") as out_file:
        all_claims = jlr.process(f)
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_gold_labels:"):
            label=claim_full["label"]
            if (label == "SUPPORTS"):
                labels = np.append(labels, 0)
            else:
                if (label == "REFUTES"):
                    labels = np.append(labels, 1)
                # else:
                #     if (label=="NOT ENOUGH INFO"):
                #         labels = np.append(labels, 2)

    return labels
