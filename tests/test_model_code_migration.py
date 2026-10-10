"""Durable offline import/CLI contracts; independent of local model artifacts.

One-time historical AST/hash audits are recorded in model-migration.json,
not enforced here against future valid implementation changes.
"""
import ast
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "modernbert_ml_backend"

# Minimal dependency modules allow the actual implementation imports to execute
# without torch, model weights, SDK clients or site packages.
STUBS = r'''
import contextlib, importlib, sys, types
def fake(name, **attrs):
    value = types.ModuleType(name)
    value.__dict__.update(attrs)
    sys.modules[name] = value
    return value
fake('torch', Tensor=type('Tensor', (), {}), optim=types.SimpleNamespace(Optimizer=object), device=object,
     cuda=types.SimpleNamespace(amp=types.SimpleNamespace(GradScaler=object)))
fake('torch.nn', Module=object)
fake('torch.utils')
fake('torch.utils.data', Dataset=object, DataLoader=object)
fake('transformers', AutoModel=object, get_linear_schedule_with_warmup=object)
fake('sklearn')
fake('sklearn.metrics', precision_recall_fscore_support=object)
fake('tqdm', tqdm=object)
cfg = types.SimpleNamespace(device='cpu', base_model_name='synthetic', model_scope=None)
fake('config', config=cfg)
'''
SERVICE_STUBS = r'''
fake('label_studio_ml', __path__=[])
fake('label_studio_ml.model', LabelStudioMLBase=object)
fake('label_studio_ml.response', ModelResponse=object)
fake('label_studio_sdk', __path__=[])
fake('label_studio_sdk.label_interface', __path__=[])
fake('label_studio_sdk.label_interface.objects', PredictionValue=object)
'''


def run(code):
    script = "import sys; sys.path.insert(0, " + repr(str(BACKEND)) + ");\n" + code
    return subprocess.run(
        [sys.executable, "-S", "-c", script], cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True,
        text=True, encoding="utf-8", timeout=30,
    )


class ModelCodeMigrationTests(unittest.TestCase):
    def assert_runs(self, code):
        result = run(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_packages_import_without_ml_config_service_or_network(self):
        self.assert_runs("""
import importlib, socket
socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network'))
for name in ('modeling', 'label_studio_backend', 'backend', 'training'):
    importlib.import_module(name)
assert not ({'torch','transformers','config','label_studio_ml','label_studio_sdk'} & set(sys.modules))
""")

    def test_legacy_alias_identity_shared_config_and_patch_in_both_orders(self):
        for legacy_first in (False, True):
            with self.subTest(legacy_first=legacy_first):
                self.assert_runs(STUBS + SERVICE_STUBS + f"""
pairs = [('model','label_studio_backend.service'),
         ('backend.service','label_studio_backend.service'),
         ('train_ner_re','training.trainer')]
for old, new in pairs:
    first, second = (old,new) if {legacy_first!r} else (new,old)
    a, b = importlib.import_module(first), importlib.import_module(second)
    assert a is b
    a._migration_sentinel = object()
    assert b._migration_sentinel is a._migration_sentinel
service = importlib.import_module('label_studio_backend.service')
network = importlib.import_module('modeling.network')
inference = importlib.import_module('modeling.inference')
trainer = importlib.import_module('training.trainer')
assert all(module.config is cfg for module in (service,network,inference,trainer))
assert trainer.ModernBERTForNERRE is service.ModernBERTForNERRE is network.ModernBERTForNERRE
assert service.predict_text is inference.predict_text
assert service.NERREPredictor is inference.NERREPredictor
""")

    def test_trainer_import_does_not_import_service_or_sdk(self):
        self.assert_runs(STUBS + """
import training.trainer
assert 'backend.service' not in sys.modules and 'label_studio_backend.service' not in sys.modules and 'model' not in sys.modules
assert not any(n.startswith(('label_studio_ml','label_studio_sdk')) for n in sys.modules)
""")

    def test_service_fit_imports_trainer_only_when_training_is_requested(self):
        self.assert_runs(STUBS + SERVICE_STUBS + """
from backend.service import ModernBERTModel
assert 'training.trainer' not in sys.modules
service = ModernBERTModel()
events = []
service._load_training_data = lambda: [{'text':'synthetic'}]
service._load_model = lambda: events.append('reload')
service.fit('UNSUPPORTED', {})
assert 'training.trainer' not in sys.modules
fake('training.trainer', train_from_data=lambda **kwargs: (events.append(kwargs) or object(), .5))
service.fit('START_TRAINING', {})
assert events == [{'training_data':[{'text':'synthetic'}], 'train_ratio':.8}, 'reload']
""")

    def test_service_root_remains_original_backend_directory(self):
        self.assert_runs(STUBS + SERVICE_STUBS + """
import os
from backend import service
setup = next(n for n in __import__('ast').walk(__import__('ast').parse(open(service.__file__, encoding='utf-8').read()))
             if isinstance(n, __import__('ast').FunctionDef) and n.name == 'setup')
assignment = next(n for n in setup.body if isinstance(n, __import__('ast').Assign)
                  and isinstance(n.targets[0], __import__('ast').Attribute) and n.targets[0].attr == 'root_dir')
instance = types.SimpleNamespace()
exec(compile(__import__('ast').Module(body=[assignment], type_ignores=[]), '<root-contract>', 'exec'),
     {'self': instance, 'os': os, '__file__': service.__file__})
assert instance.root_dir == sys.path[0]
""")

    def test_legacy_training_cli_preserves_arguments_and_execution(self):
        result = self.assert_runs(STUBS + """
import runpy
sys.argv = ['train_ner_re.py','--help']
runpy.run_path(sys.path[0] + '/train_ner_re.py', run_name='__main__')
""")
        for option in ("--train-file", "--base-model", "--epochs", "--batch-size"):
            self.assertIn(option, result.stdout)
        self.assert_runs(STUBS + """
import json, pathlib, runpy, tempfile
from training import trainer
events = []
cfg.update_model_paths = lambda *args: events.append(args)
sys.modules['torch'].device = lambda value: value
trainer.train_from_data = lambda rows, train_ratio: events.append((rows,train_ratio))
with tempfile.TemporaryDirectory() as folder:
    path = pathlib.Path(folder) / 'synthetic.json'
    path.write_text(json.dumps([{'text':'synthetic'}]))
    sys.argv = ['train_ner_re.py','--train-file',str(path),'--base-model','tiny','--epochs','2','--batch-size','3']
    runpy.run_path(sys.path[0] + '/train_ner_re.py', run_name='__main__')
assert cfg.epochs == 2 and cfg.batch_size == 3
assert events == [('tiny',None), ([{'text':'synthetic'}],.9)]
""")

    def test_original_wsgi_resolves_local_runtime_and_keeps_script_import(self):
        result = self.assert_runs("""
import importlib.util
from pathlib import Path
origin = Path(importlib.util.find_spec('label_studio_ml').origin).resolve()
assert origin == Path(sys.path[0]) / 'label_studio_ml/__init__.py'
""")
        tree = ast.parse((BACKEND / "_wsgi.py").read_text("utf-8"))
        self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == "model" and
                            any(a.name == "ModernBERTModel" for a in n.names) for n in tree.body))

if __name__ == "__main__":
    unittest.main()
