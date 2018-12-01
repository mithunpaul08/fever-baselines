import csv,sys
from random import shuffle

from typing import Dict
import json,os
import logging

from overrides import overrides
import tqdm
import sys
from tqdm import tqdm as tq
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from retrieval.read_claims import UOFADataReader
from rte.riedel.data import FEVERPredictions2Formatter, FEVERLabelSchema, FEVERGoldFormatter
from common.dataset.data_set import DataSet as FEVERDataSet
from rte.mithun.trainer import UofaTrainTest

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FEVERReaderUofa():
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------   
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 db: FeverDocDB,
                 sentence_level=False,
                 wiki_tokenizer: Tokenizer = None,
                 claim_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 filtering: str = None) -> None:
        self._sentence_level = sentence_level
        self._wiki_tokenizer = wiki_tokenizer or WordTokenizer()
        self._claim_tokenizer = claim_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.db = db

        self.formatter = FEVERGoldFormatter(set(self.db.get_doc_ids()), FEVERLabelSchema(), filtering=filtering)
        self.reader = JSONLineReader()



    def read(self, mithun_logger,data_folder,all_labels):
        mithun_logger.info("got inside read in file reader_uofa.py and class FEVERReaderUofa" )
        objUofaTrainTest = UofaTrainTest()
        mithun_logger.debug(f"data_folder: {data_folder}. Going to read data")

        bf = data_folder + objUofaTrainTest.annotated_body_split_folder
        bfl = bf + objUofaTrainTest.annotated_only_lemmas
        bfw = bf + objUofaTrainTest.annotated_words
        bfe = bf + objUofaTrainTest.annotated_only_entities

        hf = data_folder + objUofaTrainTest.annotated_head_split_folder
        hft = hf + objUofaTrainTest.annotated_only_tags
        hfd= hf + objUofaTrainTest.annotated_only_dep
        hfl = hf + objUofaTrainTest.annotated_only_lemmas
        hfw = hf + objUofaTrainTest.annotated_words
        hfe = hf + objUofaTrainTest.annotated_only_entities
        hfcomplete = hf + objUofaTrainTest.annotated_whole_data_head



        heads_lemmas = objUofaTrainTest.read_json(hfl)
        bodies_lemmas = objUofaTrainTest.read_json(bfl)
        heads_entities = objUofaTrainTest.read_json(hfe)
        bodies_entities = objUofaTrainTest.read_json(bfe)
        heads_words = objUofaTrainTest.read_json(hfw)
        bodies_words = objUofaTrainTest.read_json(bfw)
        heads_tags= objUofaTrainTest.read_json(hft)
        heads_deps = objUofaTrainTest.read_json_deps(hfd)
        heads_complete_annotation=objUofaTrainTest.read_id_field_json(hfcomplete)

        mithun_logger.debug(f"length of bodies_words:{len(bodies_words)}")

        assert len(all_labels) == len(heads_entities)

        data=zip(heads_entities, bodies_entities, heads_lemmas,
                                                    bodies_lemmas,
                                                      heads_words,
                                                      bodies_words,heads_tags,heads_deps,heads_complete_annotation,all_labels)



        return data


    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._wiki_tokenizer.tokenize(premise) if premise is not None else None
        hypothesis_tokens = self._claim_tokenizer.tokenize(hypothesis)
        fields['premise'] = TextField(premise_tokens, self._token_indexers) if premise is not None else None
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)


