"""Render a review payload using the maintained adjudication template."""
import json
from pathlib import Path


def build_html(p):
    data=json.dumps(p,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return (Path(__file__).parent/'templates/double_annotation_review.html').read_text('utf-8').replace('__PAYLOAD__',data)
