FROM python:3.6
ADD for_docker/ for_docker/
RUN cd /for_docker && pip install numpy
#ENTRYPOINT ["python","import numpy as np; np"]

CMD ["python", "./for_docker/src/scripts/retrieval/ir.py --mode dev --lmode WARNING"]
