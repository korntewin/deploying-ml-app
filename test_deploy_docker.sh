
docker login --username=$HEROKU_EMAIL --password=$HEROKU_API_KEY registry.heroku.com
curl https://cli-assets.heroku.com/install-ubuntu.sh | sh

PYTHONPATH='./packages/lasso' python3 packages/lasso/lasso/train.py
python3 -m pip install -r packages/ml-app/requirements.txt

build_docker_ml_api() {
    docker build -t registry.heroku.com/${HEROKU_APP_NAME}/web:latest .
}

push_docker_ml_api() {
    docker push registry.heroku.com/${HEROKU_APP_NAME}/web:latest
}

build_docker_ml_api
push_docker_ml_api

heroku container:release web --app $HEROKU_APP_NAME