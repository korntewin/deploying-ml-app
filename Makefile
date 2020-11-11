HEROKU_APP_NAME=my-ml-app-udemy

build-docker-mlapi:
	docker build -t registry.heroku.com/$(HEROKU_APP_NAME)/web .

push-docker-mlapi:
	docker push registry.heroku.com/${HEROKU_APP_NAME}/web:latest