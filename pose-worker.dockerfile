FROM wildflowerschools/wf-deep-docker:cuda10.2-alphapose-base-v21


RUN pip install retry pika click wf-minimal-honeycomb-python wildflower-honeycomb-sdk \
    pytelegraf redis hiredis marshmallow

RUN apt install -y jq

COPY ./producer /opt/producer
COPY ./setup.py /opt/setup.py

# COPY ./model/v2/yolov4.wf.0.2.cfg
# COPY ./model/v2/yolov4.wf.0.2.weights
# COPY ./model/v2/alphapose-wf_res152_256x192.0.2.yolov4.yaml
# COPY ./model/v2/alphapose-wf_res152_256x192.0.2.yolov4.pth
# COPY ./model/v2/fast_421_res152_256x192.pth



ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN cd /opt && pip install .

RUN mkdir /data

WORKDIR /build/AlphaPose


ENTRYPOINT ["python"]
