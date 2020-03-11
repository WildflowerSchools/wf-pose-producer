VERSION ?= 0

.PHONY: build


build:
	docker build --no-cache -t wildflowerschools/wf-deep-docker:pose-producer-v$(VERSION) -f Dockerfile .
	docker push wildflowerschools/wf-deep-docker:pose-producer-v$(VERSION)

