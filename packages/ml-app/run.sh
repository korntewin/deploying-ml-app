export IS_DEBUG=${DEBUG:-false}
gunicorn -b :5000 --pythonpath packages/ml-app --access-logfile - --error-logfile - run:application