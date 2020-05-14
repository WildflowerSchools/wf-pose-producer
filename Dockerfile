FROM wildflowerschools/wf-deep-docker:cuda10.2-alphapose-base-v7

RUN pip install retry pika click wf-minimal-honeycomb-python cython_bbox wildflower-honeycomb-sdk

ADD ./producer /opt/producer
ADD ./setup.py /opt/setup.py

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install gpu-utils

RUN cd /opt && pip install .

WORKDIR /build/AlphaPose
