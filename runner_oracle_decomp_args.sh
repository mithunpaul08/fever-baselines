POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -m|--modelfile)
    MODELFILE="$2"
    shift # past argument
    shift # past value
    ;;
    -l|--logsdir)
    LOGDIR="$2"
    shift # past argument
    shift # past value
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


#rm -rf logs/
PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db config/fever_nn_ora_sent.json logs/${LOGDIR} --cuda-device $CUDA_DEVICE
mkdir -p data/models
cp logs/${LOGDIR}/model.tar.gz data/models/${MODELFILE}.tar.gz
echo "Copied model file to data/models/${MODELFILE}.tar.gz"
PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db data/models/${MODELFILE}.tar.gz




