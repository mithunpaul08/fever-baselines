FROM python:3.6
ADD for_docker/ for_docker/
RUN cd /for_docker && pip install numpy
