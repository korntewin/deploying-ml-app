from myapi.app import create_app
from myapi import config


application = create_app(config_object=config.DevelopmentConfig)


if __name__ == '__main__':
    application.run()