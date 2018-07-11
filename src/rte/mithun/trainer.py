from __future__ import division
from rte.mithun.log import setup_custom_logger
import sys
from sklearn import svm
import tqdm
import os
import numpy as np
from tqdm import tqdm
import time
from sklearn.externals import joblib
from processors import ProcessorsBaseAPI
from processors import Document
import json
API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
my_out_dir = "poop-out"
n_cores = 2
LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
RELATED = LABELS[0:3]
annotated_only_lemmas="ann_lemmas.json"
annotated_only_tags="ann_tags.json"
annotated_body_split_folder="split_body/"
annotated_head_split_folder="split_head/"
data_folder="/data/fever-data-ann/"


def read_json_feat_vec(load_ann_corpus_tr,gold_labels_tr,logger):


    logger.debug("value of load_ann_corpus_tph2:" + str(load_ann_corpus_tr))

    cwd=os.getcwd()
    bf=cwd+data_folder+annotated_body_split_folder
    bff=bf+annotated_only_lemmas
    bft=bf+annotated_only_tags

    hf=cwd+data_folder+annotated_head_split_folder
    hff=hf+annotated_only_lemmas
    hft=hf+annotated_only_tags


    logger.debug("hff:" + str(hff))
    logger.debug("bff:" + str(bff))
    logger.info("going to read heads_lemmas from disk:")

    heads_lemmas = read_json(hff,logger)
    bodies_lemmas = read_json(bff,logger)
    heads_tags = read_json(hft,logger)
    bodies_tags = read_json(bft,logger)


    logger.debug("size of heads_lemmas is: " + str(len(heads_lemmas)))
    logger.debug("size of bodies_lemmas is: " + str(len(bodies_lemmas)))


    if not (len(heads_lemmas) == len(bodies_lemmas)):
        logger.debug("size of heads_lemmas and bodies_lemmas dont match")
        sys.exit(1)


    combined_vector = create_feature_vec(heads_lemmas, bodies_lemmas, heads_tags,
                                         bodies_tags,logger)

    joblib.dump(combined_vector, 'combined_vector_testing_phase2.pkl')

    logger.debug("done generating feature vectors. Going to call classifier")

    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(combined_vector, gold_labels_tr.ravel())

    joblib.dump(clf, 'model_trained_phase2.pkl')

    return;


def read_json(json_file,logger):
    logger.debug("inside read_json_pyproc_doc")
    l = []
    counter=0

    with open(json_file) as f:
        for eachline in (f):
            d = json.loads(eachline)
            a=d["data"]
            just_lemmas=' '.join(str(r) for v in a for r in v)
            l.append(just_lemmas)
            logger.debug(counter)
            counter = counter + 1

    logger.debug("counter:"+str(counter))
    return l


def normalize_dummy(text):
    x = text.lower().translate(remove_punctuation_map)
    return x.split(" ")

def create_feature_vec(heads_lemmas,bodies_lemmas,heads_tags_related,bodies_tags_related,logger):
    word_overlap_vector = np.empty((0, 1), float)
    hedging_words_vector = np.empty((0, 30), int)
    refuting_value_matrix = np.empty((0, 16), int)
    noun_overlap_vector = np.empty((0, 2), int)

    for head_lemmas, body_lemmas,head_tags_related,body_tags_related in tqdm(zip(heads_lemmas, bodies_lemmas,heads_tags_related,bodies_tags_related),
                           total=len(bodies_tags_related), desc="feat_gen:"):

        lemmatized_headline = head_lemmas
        lemmatized_body=body_lemmas
        tagged_headline=head_tags_related
        tagged_body=body_tags_related

        # logger.debug(lemmatized_headline)
        # logger.debug(lemmatized_body)
        # logger.debug(tagged_headline)
        # logger.debug(tagged_body)



        word_overlap_array, hedge_value_array, refuting_value_array, noun_overlap_array = add_vectors(
            lemmatized_headline, lemmatized_body, tagged_headline, tagged_body,logger)

        word_overlap_vector = np.vstack([word_overlap_vector, word_overlap_array])
        hedging_words_vector = np.vstack([hedging_words_vector, hedge_value_array])
        refuting_value_matrix = np.vstack([refuting_value_matrix, refuting_value_array])
        noun_overlap_vector = np.vstack([noun_overlap_vector, noun_overlap_array])




    logger.debug("\ndone with all headline body.:")
    logger.debug("shape of  word_overlap_vector is:" + str(word_overlap_vector.shape))
    logger.debug("refuting_value_matrix.dtype=" + str(refuting_value_matrix.dtype))
    logger.debug("refuting_value_matrix is =" + str(refuting_value_matrix))

    combined_vector = np.hstack(
        [word_overlap_vector, hedging_words_vector, refuting_value_matrix, noun_overlap_vector])

    return combined_vector


def add_vectors(lemmatized_headline,lemmatized_body,tagged_headline,tagged_body,logger):
    word_overlap = word_overlap_features_mithun(lemmatized_headline, lemmatized_body)
    word_overlap_array = np.array([word_overlap])

    hedge_value = hedging_features_mithun(lemmatized_headline, lemmatized_body)
    hedge_value_array = np.array([hedge_value])

    refuting_value = refuting_features_mithun(lemmatized_headline, lemmatized_body)
    refuting_value_array = np.array([refuting_value])

    noun_overlap = noun_overlap_features(lemmatized_headline, tagged_headline, lemmatized_body, tagged_body)
    noun_overlap_array = np.array([noun_overlap])

    return word_overlap_array,hedge_value_array,refuting_value_array,noun_overlap_array


def word_overlap_features_mithun(clean_headline, clean_body):

    features = [
        len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]

    return features

def hedging_features_mithun(headline, clean_body):



    hedging_words = [
        'allegedly',
        'reportedly',
      'argue',
      'argument',
      'believe',
      'belief',
      'conjecture',
      'consider',
      'hint',
      'hypothesis',
      'hypotheses',
      'hypothesize',
      'implication',
      'imply',
      'indicate',
      'predict',
      'prediction',
      'previous',
      'previously',
      'proposal',
      'propose',
      'question',
      'speculate',
      'speculation',
      'suggest',
      'suspect',
      'theorize',
      'theory',
      'think',
      'whether'
    ]

    length_hedge=len(hedging_words)
    #logging.debug(length_hedge)
    hedging_body_vector = [0] * length_hedge


    #logging.debug("shape of hedging_body_vector is" + str(len(hedging_body_vector)))
    #logging.debug(hedging_body_vector)


    for word in clean_body:
        if word in hedging_words:
            index=hedging_words.index(word)
            #logging.debug(index)
            hedging_body_vector[index]=1

    #logging.debug("shape of hedging_body_vector is" + str(len(hedging_body_vector)))
    #logging.debug(hedging_body_vector)
    return hedging_body_vector

def refuting_features_mithun(clean_headline, clean_body):

    refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny',
        'denies',
        'refute',
        'not',
        'despite',
        'nope',
        'doubt',
        'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    length_hedge=len(refuting_words)
    refuting_body_vector = [0] * length_hedge

    # clean_headline = doAllWordProcessing(headline)
    # clean_body = doAllWordProcessing(body)

    for word in clean_body:
        if word in refuting_words:
            index=refuting_words.index(word)
            #logging.debug(index)
            refuting_body_vector[index]=1


    return refuting_body_vector

def noun_overlap_features(lemmatized_headline, headline_pos, lemmatized_body, body_pos):

        h_nouns = []
        b_nouns = []

        noun_count_headline = 0
        for word, pos in zip(lemmatized_headline, headline_pos):
            if pos.startswith('NN'):
                noun_count_headline = noun_count_headline + 1
                h_nouns.append(word)

        noun_count_body = 0
        for word, pos in zip(lemmatized_body, body_pos):
            if pos.startswith('NN'):
                noun_count_body = noun_count_body + 1
                b_nouns.append(word)

        overlap = set(h_nouns).intersection(set(b_nouns))

        overlap_noun_counter = len(overlap)

        features = [0, 0]

        if (noun_count_body > 0 and noun_count_headline > 0):
            prop_nouns_sent1 = overlap_noun_counter / (noun_count_body)
            prop_nouns_sent2 = overlap_noun_counter / (noun_count_headline)

            features = [prop_nouns_sent1, prop_nouns_sent2]

        return features