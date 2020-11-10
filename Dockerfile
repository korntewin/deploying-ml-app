FROM python:3.7.9

ENV FLASK_APP run.py

ADD packages/ml-app /app/packages/ml-app
ADD packages/lasso /app/packages/lasso

WORKDIR /app

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r packages/ml-app/requirements.txt

RUN chmod +x packages/ml-app/run.sh

EXPOSE 5000
CMD ["bash", "./packages/ml-app/run.sh"]