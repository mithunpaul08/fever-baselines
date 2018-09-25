from typing import Dict
import json,sys,os
import logging

from overrides import overrides
import tqdm

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
from rte.riedel.data import FEVERPredictions2Formatter, FEVERLabelSchema, FEVERGoldFormatter
from common.dataset.data_set import DataSet as FEVERDataSet
from processors import Document
from processors import ProcessorsBaseAPI
from retrieval.read_claims import UOFADataReader
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("fever")
class FEVERReader(DatasetReader):
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
                 sentence_level = False,
                 wiki_tokenizer: Tokenizer = None,
                 claim_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 filtering: str = None) -> None:
        self._sentence_level = sentence_level
        self._wiki_tokenizer = wiki_tokenizer or WordTokenizer()
        self._claim_tokenizer = claim_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.db = db

        self.formatter = FEVERGoldFormatter(set(self.db.get_doc_ids()), FEVERLabelSchema(),filtering=filtering)
        self.reader = JSONLineReader()


    def get_doc_line(self,doc,line):
        lines = self.db.get_doc_lines(doc)
        if line > -1:
            return lines.split("\n")[line].split("\t")[1]
        else:
            non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
            return non_empty_lines[SimpleRandom.get_instance().next_rand(0,len(non_empty_lines)-1)]

    @overrides
    def read(self, file_path: str):

        instances = []

        ds = FEVERDataSet(file_path,reader=self.reader, formatter=self.formatter)
        ds.read()
        counter=0

        objUOFADataReader = UOFADataReader()
        # DELETE THE FILE IF IT EXISTS every time before the loop
        self.delete_if_exists(objUOFADataReader.ann_head_tr)
        self.delete_if_exists(objUOFADataReader.ann_body_tr)


        for instance in tqdm.tqdm(ds.data):
            counter=counter+1
            if instance is None:
                continue

            if not self._sentence_level:
                pages = set(ev[0] for ev in instance["evidence"])
                premise = " ".join([self.db.get_doc_text(p) for p in pages])
            else:
                lines = set([self.get_doc_line(d[0],d[1]) for d in instance['evidence']])
                premise = " ".join(lines)

            if len(premise.strip()) == 0:
                premise = ""

            hypothesis = instance["claim"]
            label = instance["label_text"]



            #call you pyprocessors annotator here, and write to disk

            self.uofa_annotate(hypothesis,premise,counter,objUOFADataReader)

            instances.append(self.text_to_instance(premise, hypothesis, label))
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
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

    @classmethod
    def from_params(cls, params: Params) -> 'FEVERReader':
        claim_tokenizer = Tokenizer.from_params(params.pop('claim_tokenizer', {}))
        wiki_tokenizer = Tokenizer.from_params(params.pop('wiki_tokenizer', {}))

        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        sentence_level = params.pop("sentence_level",False)
        db = FeverDocDB(params.pop("db_path","data/fever.db"))
        params.assert_empty(cls.__name__)
        return FEVERReader(db=db,
                           sentence_level=sentence_level,
                           claim_tokenizer=claim_tokenizer,
                           wiki_tokenizer=wiki_tokenizer,
                           token_indexers=token_indexers)

    def uofa_annotate(self,claim, evidence,index,objUOFADataReader):
        # logger.info(f' hypothesis is:{claim}')
        # logger.info(f'premise is:{evidence}')
        # logger.info(f' label is:{index}')
        # print(f' hypothesis is:{claim}')
        # print(f'premise is:{evidence}')
        # print(f' label is:{index}')
        head_ann,body_ann= objUOFADataReader.annotate_and_save_doc(claim, evidence, index, objUOFADataReader.API, objUOFADataReader.ann_head_tr, objUOFADataReader.ann_body_tr, logger)

        # heads_entities=head_ann.sentences[0].entities
        # heads_lemmas= head_ann.sentences[0].entities
        # heads_words=head_ann.sentences[0].words
        # bodies_entities = body_ann.sentences[0].entities
        # bodies_lemmas = body_ann.sentences[0].entities
        # bodies_words = body_ann.sentences[0].words
        #
        # print(f'{heads_entities}')
        # print(f'{heads_lemmas}')
        # print(f'{heads_words}')
        # print(f'{bodies_entities}')
        # print(f'{bodies_lemmas}')
        # print(f'{bodies_words}')
        # sys.exit(1)


    def delete_if_exists(self,name):

        if os.path.exists(name):
            append_write = 'w'  # make a new file if not
            with open(name, append_write) as outfile:
                outfile.write("")

    def convert_NER_form(self,heads_entities, bodies_entities, heads_lemmas,
                                                bodies_lemmas, heads_words, bodies_words, labels_no_nei):

        instances=[]

        for he, be, hl, bl, hw, bw, lbl in (zip(heads_entities, bodies_entities, heads_lemmas,
                                                bodies_lemmas, heads_words, bodies_words, labels_no_nei)):

            he_split_list = he.data.split(" ")
            hl_split_list = hl.data.split(" ")
            hw_split_list = hw.data.split(" ")

            be_split_list = be.data.split(" ")
            bl_split_list = bl.data.split(" ")
            bw_split_list = bw.data.split(" ")

            neutered_headline = []
            neutered_body = []

            for hee, hll, hww in zip(he_split_list, hl_split_list, hw_split_list):

                # if no NER tag exists, use the lemma itself, else use the NER tag
                if (hee == 'O'):
                    neutered_headline.append(hww)
                    # if NER tag exists use the NER tag
                else:
                    neutered_headline.append(hee)

            for bee, bll, bww in zip(be_split_list, bl_split_list, bw_split_list):

                # if no NER tag exists, use the lemma itself, else use the NER tag
                if (bee == 'O'):
                    neutered_body.append(bww)
                    # if NER tag exists use the NER tag
                else:
                    neutered_body.append(bee)

            premise = "".join(neutered_headline)
            hypothesis = "".join(neutered_body)
            label = lbl

            inst = self.text_to_instance(premise, hypothesis, label)

            instances.append(inst)

        return Dataset(instances)
