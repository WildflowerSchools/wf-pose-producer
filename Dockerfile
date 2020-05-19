FROM wildflowerschools/wf-deep-docker:cuda10.2-alphapose-base-v7

RUN pip install retry pika click wf-minimal-honeycomb-python cython_bbox wildflower-honeycomb-sdk redis

RUN apt install -y redis-tools jq

ADD ./producer /opt/producer
ADD ./setup.py /opt/setup.py

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install gpu-utils

RUN cd /opt && pip install .

COPY ./scripts/alphapose-runner.sh /usr/bin/alphapose-runner

RUN chmod +x /usr/bin/alphapose-runner

WORKDIR /build/AlphaPose
