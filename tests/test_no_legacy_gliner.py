"""Keep retired GLiNER code out of the retained data/model pipeline."""
import ast
import os
from pathlib import Path
import re
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LegacyGLiNERRetirementTests(unittest.TestCase):
    def test_retained_code_and_dependencies_do_not_use_gliner(self):
        legacy = {'gliner', 'gliner2', 'ml_backend'}
        for folder in ['modernbert_ml_backend', 'scripts', 'tests', 'annotation_agent_workflow/scripts']:
            for directory, children, files in os.walk(ROOT / folder, followlinks=False):
                # Local runtime Junctions are dependencies, not retained project source.
                children[:] = [name for name in children if name not in {'.venv', '__pycache__'}
                               and not (Path(directory) / name).is_junction()]
                for name in files:
                    if not name.endswith('.py'):
                        continue
                    file = Path(directory) / name
                    for node in ast.walk(ast.parse(file.read_text(encoding='utf-8'))):
                        names = []
                        if isinstance(node, ast.Import):
                            names = [item.name for item in node.names]
                        elif isinstance(node, ast.ImportFrom):
                            names = [node.module or '']
                        self.assertFalse(legacy.intersection(name.split('.')[0] for name in names), str(file))
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        dependencies = {re.split(r'[\[<>=!~; ]', dep)[0] for dep in project['project']['dependencies']}
        self.assertFalse(legacy & dependencies)
        self.assertTrue({'torch', 'transformers', 'sentencepiece'} <= dependencies)
        locked = {p['name'] for p in tomllib.loads((ROOT / 'uv.lock').read_text())['package']}
        self.assertFalse({'gliner', 'gliner2'} & locked)
        self.assertTrue({'torch', 'transformers', 'sentencepiece'} <= locked)


if __name__ == '__main__':
    unittest.main()
