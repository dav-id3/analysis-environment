#!make

cmd=current

PROJECT_NAME := analysis-environment

#containers name
HOME_CONTAINER := $(PROJECT_NAME)

#path
HOME_FOLDER := $(CURDIR)
ENVFILE_FOLDER := $(CURDIR)/envfiles

##Network
.PHONY: network.create
network.create:
	@docker network create --driver bridge $(DOCKER_NETWORK)

.PHONY: network.remove
network.remove:
	@docker network rm $(DOCKER_NETWORK)

##Database
.PHONY: build
build:
	@docker build \
		-f $(HOME_FOLDER)/Dockerfile \
		-t $(HOME_CONTAINER) $(HOME_FOLDER) 

.PHONY: run
run:
	@docker run --rm -it \
		--env-file $(ENVFILE_FOLDER)/.env \
		--name $(HOME_CONTAINER) \
		$(HOME_CONTAINER) \
		/bin/bash