FROM wildflowerschools/wf-deep-docker:cuda10.2-alphapose-base-v19

RUN pip install retry pika click wf-minimal-honeycomb-python cython_bbox wildflower-honeycomb-sdk redis

RUN apt install -y redis-tools jq

ADD ./producer /opt/producer
ADD ./setup.py /opt/setup.py

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install gpu-utils

RUN cd /opt && pip install .

COPY ./scripts/inference.sh /usr/local/bin/inference.sh
# COPY ./scripts/demo_inference.py /build/AlphaPose/scripts/demo_inference.py

RUN chmod +x /usr/local/bin/inference.sh

RUN mkdir /data

WORKDIR /build/AlphaPose
