FROM wildflowerschools/wf-deep-docker:cuda10.2-alphapose-base-v23

LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3-dev \
    jq \
    && rm -rf /var/lib/apt/lists/*


RUN pip install retry pika click wf-minimal-honeycomb-python wildflower-honeycomb-sdk \
    pytelegraf redis hiredis minio tenacity dateparser multi-model-server \
    sagemaker-inference retrying dataclasses_json PyYAML mxnet

#RUN pip install wf-video-io

COPY ./model/v2/data /build/AlphaPose/data
COPY ./model/v2/pretrained_models /build/AlphaPose/pretrained_models

COPY ./wf-video-io /build/wf-video-io
RUN cd /build/wf-video-io && pip install .

COPY ./producer /opt/producer
COPY ./modeltools /opt/modeltools
COPY ./setup.py /opt/setup.py

COPY ./sagemaker/wf_model_handler.py /opt/wf_model_handler.py
COPY ./sagemaker/model-entrypoint.py /opt/model-entrypoint.py

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN cd /opt && pip install .

RUN mkdir /data

WORKDIR /build/AlphaPose

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/jre/

ENTRYPOINT ["python", "/opt/model-entrypoint.py"]

CMD ["serve"]
