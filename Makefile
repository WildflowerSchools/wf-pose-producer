VERSION ?= 0

.PHONY: build


build:
	docker build -t wildflowerschools/wf-deep-docker:alphapose-producer-v$(VERSION) -f Dockerfile .
	docker push wildflowerschools/wf-deep-docker:alphapose-producer-v$(VERSION)
