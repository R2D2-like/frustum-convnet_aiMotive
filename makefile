DOCKER_CUDA:=11.7.1
LOCAL_BASE_IMAGE:=nvidia/cuda:$(DOCKER_CUDA)-devel-ubuntu22.04
DOCKERFILE_DIRECTORY := ./docker
DOCKERCOMPOSE_FILE := docker-compose.yml
DOCKER_IMAGE_NAME:=frustum-convnet
CONTAINER_NAME:=frustum-convnet
PROJECT:=det #現在いるディレクトリがデフォルト


.PHONY: build_docker
build_docker:
	docker-compose -p ${PROJECT} -f $(DOCKERFILE_DIRECTORY)/$(DOCKERCOMPOSE_FILE) build

.PHONY: run_docker
run_docker:
	docker-compose -p ${PROJECT} -f $(DOCKERFILE_DIRECTORY)/$(DOCKERCOMPOSE_FILE) up -d &&\
	docker exec -it ${CONTAINER_NAME} bash

