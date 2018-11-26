mkdir mithun_hw4
cd  mithun_hw4
git clone git@github.com:mithunpaul08/fever-baselines.git .
git checkout v1.9
mkdir  -p data/fever-data-ann
conda env create -f environment.yml
source activate hw4mithun
cp -r /net/kate/storage/work/mithunpaul/fever/dev_branch/fever-baselines/data/fever-data-ann/dev data/fever-data-ann/dev
pip install -r requirements.txt
cp -r /work/mithunpaul/fever/dev_branch/fever-baselines/data/fever data/
mkdir -p data/models
cp -r /work/mithunpaul/fever/dev_branch/fever-baselines/data/models/smartner_tr-fever_98467_100.tar.gz data/models/
export CUDA_DEVICE=0
PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db  data/fever/dev.ns.pages.p1.jsonl --param_path config/fever_nn_ora_sent_hw4.json --randomseed 1234 --slice 100

