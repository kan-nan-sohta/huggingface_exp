FOLDER=$$(pwd)
IMAGE_NAME=test:latest

.PHONY: build
build: # Build docker image
	docker compose up -d

.PHONY: start
start: # Start docker container
	sudo docker run \
		-v /mnt/c/Users/ctddd/huggingface-exp:/home/kan_nan/huggingface-exp \
		--rm \
		--gpus all \
		-it huggingface-exp-core