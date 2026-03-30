import os
from label_studio_ml.api import init_app
from .model import PneumoniaNERModel

app = init_app(model_class=PneumoniaNERModel)

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 9090)),
        debug=bool(os.getenv("DEBUG", False)),
    )
