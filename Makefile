# Makefile for the Instrument Data Pipeline project

.PHONY: setup test run

setup:
	@echo "Setting up the environment..."
	pip install -r requirements.txt

run:
	@echo "Starting the application..."
	python dash_app/app.py

build:
	@echo "Building Docker image..."
	docker build -t instrument-data-pipeline .

start:
	@echo "Starting Docker container..."
	docker run -p 80:80 instrument-data-pipeline

stop:
	@echo "Stopping Docker container..."
	docker stop $(docker ps -q --filter ancestor=instrument-data-pipeline)

docker-clean:
	@echo "Removing stopped containers..."
	docker rm $(docker ps -a -q)

clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
