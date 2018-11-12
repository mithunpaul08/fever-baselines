import os
from shutil import copyfile
from git import Repo
from copy import deepcopy
from common.util.log_helper import LogHelper
from rte.mithun.ds import indiv_headline_body
from tqdm import tqdm
import logging
from rte.mithun.trainer import read_json_create_feat_vec,do_training,do_testing,load_model,print_missed
import numpy as np
import os,sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from rte.parikh.reader import FEVERReader
from scorer.src.fever.scorer import fever_score
import json,sys
from sklearn.externals import joblib
from allennlp.data import Vocabulary, Dataset, DataIterator, DatasetReader, Tokenizer, TokenIndexer
from processors import ProcessorsBaseAPI

ann_head_tr = "ann_head_tr.json"
ann_body_tr = "ann_body_tr.json"
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# logger=None
load_ann_corpus=True
#load_combined_vector=True

predicted_results="predicted_results.pkl"
snli_filename='snli_fever.json'
API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)

def read_claims_annotate(args,jlr,logger,method,db,params):

    archive_root= params.pop('archive_root')
    logger.info(f"archive_root:{archive_root}")
    cwd=os.getcwd()
    src_file_home_dir=cwd
    copy_file_to_archive(archive_root,args.mode,src_file_home_dir,ann_head_tr,ann_body_tr,logger)

    sys.exit(1)



    logger.error("Going to delete annotations output file if it exists")
    try:
        os.remove(ann_head_tr)
        os.remove(ann_body_tr)

    except OSError:
        logger.error("annotations output file  doesnt exist")

    logger.debug("inside read_claims_annotate")


    # #Looks like our code wasn't retreiving the nearest neighbor for NEI. So using the fever baseline code to do it
    os.makedirs(args.serialization_dir, exist_ok=True)
    serialization_params = deepcopy(params).as_dict(quiet=True)


    with open(os.path.join(args.serialization_dir, "model_params.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

        # Now we begin assembling the required parts for the Trainer.
    ds_params = params.pop('dataset_reader', {})
    dataset_reader = FEVERReader(db,
                                 sentence_level=ds_params.pop("sentence_level", False),
                                 wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                                 claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                                 token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})),
                                 filtering=args.filtering)

    validation_data_path = params.pop('validation_data_path')
    logger.info("Reading  data from %s", validation_data_path)
    data= dataset_reader.read_uofa(validation_data_path)

    logger.info(f"data is of type{type(data)}")

    counter=0
    for item in (tqdm(data)):
        claim = item["claim"]
        evidence = item["evidence"]
        label = item["label"]
        logger.debug(f"claim:{claim}")
        logger.debug(f"evidence:{evidence}")
        logger.debug(f"label:{label}")
        annotate_and_save_doc(claim, evidence, label, API, ann_head_tr, ann_body_tr, logger)

    archive_root= params.pop('archive_root')
    logger.info(f"archive_root:{archive_root}")
    cwd=os.getcwd()
    src_file_home_dir=cwd
    copy_file_to_archive(archive_root,args.mode,src_file_home_dir,ann_head_tr,logger)

    return data


def print_cv(combined_vector,gold_labels_tr):
    logging.debug(gold_labels_tr.shape)
    logging.debug(combined_vector.shape)
    x= np.column_stack([gold_labels_tr,combined_vector])
    np.savetxt("cv.csv", x, delimiter=",")
    sys.exit(1)


def uofa_training(args,jlr,params):
    #logger.warning("got inside uofatraining")

    #this code annotates the given file using pyprocessors. Run it only once in its lifetime.
    tr_data=read_claims_annotate(args,jlr,logger,method,params)
    logger.info(
        "Finished writing annotated json to disk . going to quit. names of the files are:" + ann_head_tr + ";" + ann_body_tr)
    sys.exit(1)
    logger.info(
        "Finished writing annotated json to disk . going to quit. names of the files are:" + ann_head_tr + ";" + ann_body_tr)

    gold_labels_tr =None
    if(args.mode =="small"):
        gold_labels_tr =get_gold_labels_small(args, jlr)
    else:
        gold_labels_tr = get_gold_labels(args, jlr)

    logging.info("number of rows in label list is is:" + str(len(gold_labels_tr)))
    combined_vector = read_json_create_feat_vec(load_ann_corpus,args)

    logging.warning("done with generating feature vectors. Model training next")
    logging.info("gold_labels_tr is:" + str(len(gold_labels_tr)))
    logging.info("shape of cv:" + str(combined_vector.shape))
    logging.info("above two must match")

    do_training(combined_vector, gold_labels_tr)

    logging.warning("done with training. going to exit")
    sys.exit(1)



def uofa_testing(args,jlr):


    #logger.warning("got inside uofa_testing")
    gold_labels = get_gold_labels(args, jlr)
    label_ev=get_gold_labels_evidence(args, jlr)




    combined_vector= read_json_create_feat_vec(load_ann_corpus,args)
    #print_cv(combined_vector, gold_labels)
    logging.info("done with generating feature vectors. Model loading and predicting next")
    logging.info("shape of cv:"+str(combined_vector.shape))
    logging.info("number of rows in label list is is:" + str(len(gold_labels)))
    logging.info("above two must match")
    assert(combined_vector.shape[0]==len(gold_labels))
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
    logging.info(classification_report(gold_labels, pred))
    logging.info(confusion_matrix(gold_labels, pred))



    # get number of support vectors for each class
    #logging.debug(trained_model.n_support_)
    logging.info("done with testing. going to exit")
    final_predictions=write_pred_str_disk(args,jlr,pred)
    fever_score(final_predictions,label_ev)
    sys.exit(1)

def annotate_save_quit(test_data,logger):

    for i, d in tqdm(enumerate(test_data), total=len(test_data),desc="annotate_json:"):
        annotate_and_save_doc(d, i, API, ann_head_tr, ann_body_tr,logger)


    sys.exit(1)


#load predictions, convert it based on label and write it as string.
def write_pred_str_disk(args,jlr,pred):
    logging.debug("here1"+str(args.out_file))
    final_predictions=[]
    #pred=joblib.load(predicted_results)
    with open(args.in_file,"r") as f:
        ir = jlr.process(f)
        logging.debug("here2"+str(len(ir)))

        for index,(p,q) in enumerate(zip(pred,ir)):
            line=dict()
            label="not enough info"
            if(p==0):
                label="supports"
            else:
                if(p==1):
                    label="refutes"

            line["id"]=q["id"]
            line["predicted_label"]=label
            line["predicted_evidence"]=q["predicted_sentences"]
            logging.debug(q["id"])
            logging.debug(label)
            logging.debug(q["predicted_sentences"])
            logging.debug(index)

            final_predictions.append(line)

    logging.info(len(final_predictions))

    with open(args.pred_file, "w+") as out_file:
        for x in final_predictions:
            out_file.write(json.dumps(x)+"\n")
    return final_predictions


def annotate_and_save_doc(headline,body, label, API, json_file_tr_annotated_headline,json_file_tr_annotated_body,
                          logger):
    logger.debug(f"got inside annotate_and_save_doc")
    logger.debug(f"headline:{headline}")
    logger.debug(f"body:{body}")
    doc1 = API.fastnlp.annotate(headline)
    doc1.id=label
    with open(json_file_tr_annotated_headline, "a") as out:
      out.write(doc1.to_JSON())
      out.write("\n")


    doc2 = API.fastnlp.annotate(body)
    logger.debug(doc2)
    doc2.id = label

    with open(json_file_tr_annotated_body, "a") as out:
          out.write(doc2.to_JSON())
          out.write("\n")

    return


def write_snli_format(headline,body,logger,label):

    logger.debug("got inside write_snli_format")
    #dictionary to dump to json for allennlp format
    snli={"annotator_labels": [""],
        "captionID": "",
    "gold_label": label,
     "pairID": "",
     "sentence1": headline,
     "sentence1_binary_parse": "",
     "sentence1_parse": "",
     "sentence2": body,
     "sentence2_binary_parse": "",
     "sentence2_parse": ""
             }

    logger.debug("headline:"+headline)
    logger.debug("body:" + body)

    if os.path.exists(snli_filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not


    with open(snli_filename, append_write) as outfile:
        json.dump(snli, outfile)
        outfile.write("\n")


    return

def get_gold_labels(validation_data_path,jlr):
    labels = np.array([[]])

    with open(validation_data_path,"r") as f, open(args.out_file, "w+") as out_file:
        all_claims = jlr.process(f)
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_gold_labels:"):
            label=claim_full["label"]
            if (label == "SUPPORTS"):
                labels = np.append(labels, 0)
            else:
                if (label == "REFUTES"):
                    labels = np.append(labels, 1)
                else:
                    if (label=="NOT ENOUGH INFO"):
                        labels = np.append(labels, 2)

    return labels

def get_gold_labels_evidence(args,jlr):
    evidences=[]
    with open(args.in_file,"r") as f:
        all_claims = jlr.process(f)
        gold=dict()
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_gold_labels_ev:"):
            label=claim_full["label"]
            if not (label.lower()=="NOT ENOUGH INFO"):
                gold["label"]=label
                gold["evidence"]=claim_full["evidence"]
                evidences.append(gold)

    return evidences

def get_claim_evidence_sans_NEI(args,jlr):
    claims=[]
    evidences=[]

    with open(args.in_file,"r") as f:
        all_claims = jlr.process(f)
        gold=dict()
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_gold_labels_ev:"):
            label=claim_full["label"]
            if not (label.lower()=="NOT ENOUGH INFO"):
                gold["label"]=label
                gold["evidence"]=claim_full["evidence"]
                evidences.append(gold)
                claims.append(claim_full)

    return claims,evidences

def get_gold_labels_small(args,jlr):
    labels = np.array([[]])

    counter=0
    with open(args.in_file,"r") as f, open(args.out_file, "w+") as out_file:
        all_claims = jlr.process(f)
        for index,claim_full in tqdm(enumerate(all_claims),total=len(all_claims),desc="get_gold_labels:"):
            counter+=1
            label=claim_full["label"]
            if (label == "SUPPORTS"):
                labels = np.append(labels, 0)
            else:
                if (label == "REFUTES"):
                    labels = np.append(labels, 1)
                else:
                    if (label=="NOT ENOUGH INFO"):
                        labels = np.append(labels, 2)
            logging.debug(index)
            if (counter==10):
                return labels
    return labels


def uofa_dev(args, jlr,method,db,params):



    logging.warning("got inside uofa_dev")

    #for annotation: you will probably run this only once in your lifetime.
    # data = read_claims_annotate(args, jlr, logger, method,db,params)
    # logger.info(
    #     "Finished writing annotated json to disk . going to quit. names of the files are:" + ann_head_tr + ";" + ann_body_tr)
    # sys.exit(1)

    combined_vector= read_json_create_feat_vec(load_ann_corpus,args)
    #print_cv(combined_vector, gold_labels)
    logging.info("done with generating feature vectors. Model loading and predicting next")
    logging.info("shape of cv:"+str(combined_vector.shape))

    logging.info("above two must match")
    trained_model=load_model()
    logging.debug("weights:")
    #logging.debug(trained_model.coef_ )
    pred=do_testing(combined_vector,trained_model)
    logging.debug(str(pred))


    logging.warning(str(acc)+"%")


    validation_data_path = params.pop('validation_data_path')
    logger.info("Reading  data from %s", validation_data_path)
    gold_labels = get_gold_labels(validation_data_path, jlr)
    logging.info("number of rows in label list is is:" + str(len(gold_labels)))
    logging.debug("and golden labels are:")
    logging.debug(str(gold_labels))
    logging.info(classification_report(gold_labels, pred))
    logging.warning("done testing. and the accuracy is:")
    acc = accuracy_score(gold_labels, pred) * 100

    logging.info(confusion_matrix(gold_labels, pred))



    # get number of support vectors for each class
    #logging.debug(trained_model.n_support_)
    logging.info("done with testing. going to exit")
    sys.exit(1)







def copy_file_to_archive(rootpath, mode, src_path, src_file_name1,src_file_name2, logger):
    repo =Repo(os.getcwd())
    branch=repo.active_branch.name
    sha=repo.head.object.hexsha

    full_path_branch = rootpath + branch
    full_path_branch_sha=full_path_branch+"/"+sha
    full_path_branch_sha_mode = full_path_branch_sha + "/"+mode


    src1= src_path +"/" + src_file_name1
    src2 = src_path + "/" + src_file_name2

    #create folder if it doesn't exist
    dest = full_path_branch_sha_mode + "/" + src_file_name1
    logger.debug(branch)
    logger.debug(full_path_branch_sha)
    logger.debug(full_path_branch_sha_mode)
    logger.debug(src1)
    logger.debug(dest)

    dir_check_create(full_path_branch,full_path_branch_sha,full_path_branch_sha_mode)
    #dir_create(full_path_branch_sha, full_path_branch_sha_mode)
    if_dir_create_file(full_path_branch_sha_mode,src1,dest)
    if_dir_create_file(full_path_branch_sha_mode, src2, dest)


def dir_check_create(parent,children):
    #if parent exists, create child, else create parent, then create child
    my_len=len(children)
    if(my_len==0):
        return;

    if os.path.isdir(parent):
        first_child=children[0]
        children=children[1:(my_len-1)]
        dir_check_create(first_child,children)
    else:
        os.mkdir(parent)
        first_child = children[0]
        children = children[1:(my_len - 1)]
        dir_check_create(first_child, children)



def if_dir_create_file(parent, child_src, child_dest):
    #if parent dir, create child file, else create parent, then create child file
    if os.path.isdir(parent):
        copyfile(child_src, child_dest)
    else:
        os.mkdir(parent)
        copyfile(child_src, child_dest)



def create_dirs_recursively(parent, children):
    #if parent exists, create child, else create parent, then create child
    my_len=len(children)
    print(f"my_len:{my_len}")
    print(f"parent:{parent}")

    if os.path.isdir(parent):
        print("inside if")

        if (my_len == 0):
            return;
        else:
            first_child = children[0]
            print(f"first_child:{first_child}")
            children = children[1:(my_len)]
            create_dirs_recursively(first_child, children)
    else:
        print("inside else. directory doesnt exist")
        os.mkdir(parent)
        if (my_len == 0):
            return;
        else:
            first_child = children[0]
            print(f"first_child:{first_child}")
            children = children[1:(my_len)]
            create_dirs_recursively(first_child, children)

