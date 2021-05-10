FROM public.ecr.aws/lambda/python:3.7


RUN yum install -y jq opencv

RUN pip install wf-minimal-honeycomb-python wildflower-honeycomb-sdk dateparser msgpack


COPY ./wf-video-io /build/wf-video-io
RUN cd /build/wf-video-io && pip install .


COPY ./producer ./producer
COPY ./setup.py ./
COPY ./app.py ./

RUN pip install .

RUN pip install dataclasses-json

CMD ["app.frame_extraction_lambda_handler"]
