import json
import logging
import logging.config


LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        "ugly": {"format": "%(asctime)s %(levelname)s: %(module)s.%(funcName)s:%(lineno)d  [%(message)s]"}
    },
    'handlers': {
        'ugly': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'ugly',
            'stream': 'ext://sys.stdout'
        },
    },
    'loggers': {
        'ugly': {'level': 'DEBUG', 'handlers': ['ugly']},
    }
}


def get_logger():
    logging.config.dictConfig(LOGGING)
    return logging.getLogger('ugly')


def get_json(outdir):
    logger = get_logger()
    logger.info('outdir: {}'.format(outdir))
    with open(os.path.join(outdir, 'alphapose-results.json')) as fh:
        ap_json = json.load(fh)
    return ap_json
