import os
import logging
import logging.config

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": os.getenv("LOG_LEVEL", "DEBUG"),
            "stream": "ext://sys.stdout",
            "formatter": "standard"
        }
    },
    "root": {
        "level": os.getenv("LOG_LEVEL", "DEBUG"),
        "handlers": ["console"],
        "propagate": True
    }
})

from label_studio_ml.api import init_app
from model import GLiNERModel

app = init_app(model_class=GLiNERModel)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 9090)), debug=True)
