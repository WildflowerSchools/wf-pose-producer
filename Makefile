VERSION ?= 0

.PHONY: build

build:
	pylint producer
	docker build -t wildflowerschools/wf-deep-docker:pose-worker-v$(VERSION) -f pose-worker.dockerfile .
	docker push wildflowerschools/wf-deep-docker:pose-worker-v$(VERSION)

build-legacy:
	docker build -t wildflowerschools/wf-deep-docker:alphapose-producer-v$(VERSION) -f Dockerfile .
	docker push wildflowerschools/wf-deep-docker:alphapose-producer-v$(VERSION)
	sed -i -e 's/producer-v[0-9]*/producer-v$(VERSION)/' producer.Dockerfile
	docker build -t wildflowerschools/wf-deep-docker:alphapose-producer-utils-v$(VERSION) -f producer.Dockerfile .
	docker push wildflowerschools/wf-deep-docker:alphapose-producer-utils-v$(VERSION)
