VERSION := "0"

ECR := "204031725010.dkr.ecr.us-east-2.amazonaws.com/wf-pose-producer"

build:
	pylint producer
	docker build -t wildflowerschools/wf-deep-docker:pose-worker-v{{VERSION}} -f ./docker/pose-worker.dockerfile .
	docker push wildflowerschools/wf-deep-docker:pose-worker-v{{VERSION}}

push-ecr:
	docker tag wildflowerschools/wf-deep-docker:pose-worker-v{{VERSION}} {{ECR}}:pose-worker-v{{VERSION}}
	docker push {{ECR}}:pose-worker-v{{VERSION}}


ECR-LOGIN:
	aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin {{ECR}}

build-frame-extract: ECR-LOGIN
	pylint producer
	docker build -t {{ECR}}:frame_extract-v{{VERSION}} -f ./docker/lambda-frame-extract.dockerfile .
	docker push {{ECR}}:frame_extract-v{{VERSION}}
