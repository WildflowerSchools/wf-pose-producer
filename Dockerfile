FROM wildflowerschools/wf-deep-docker:cuda10.2-alphapose-base-v2

RUN pip install retry pika click wf-minimal-honeycomb-python

ADD ./producer /opt/producer
ADD ./setup.py /opt/setup.py

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN cd /opt && pip install .

WORKDIR /build/AlphaPose

CMD ["producer", "process-video"]
