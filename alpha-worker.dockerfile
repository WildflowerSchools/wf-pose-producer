FROM wildflowerschools/wf-deep-docker:cuda10.2-alphapose-base-v20

RUN pip install retry pika click wf-minimal-honeycomb-python wildflower-honeycomb-sdk

RUN apt install -y jq

ADD ./producer /opt/producer
ADD ./setup.py /opt/setup.py

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN cd /opt && pip install .

RUN mkdir /data

WORKDIR /build/AlphaPose

ENTRYPOINT ["python", "-m", "producer.theon"]
