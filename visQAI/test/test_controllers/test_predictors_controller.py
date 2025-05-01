# visQAI/test/controllers/test_predictors_controller.py
import unittest
import tempfile
import os
import json
import pickle
import joblib
from pathlib import Path
from src.controllers.predictors_controller import PredictorsController
from src.model.predictor import BasePredictor


class DummyPredictor:
    def __init__(self):
        self.model = 'dummy_model'
        self.preprocessor = 'dummy_preprocessor'

    def predict(self, data, *args, **kwargs):
        return f'predicted:{data}'


class TestPredictorController(unittest.TestCase):

    def setUp(self):
        self.tmp_storage = tempfile.TemporaryDirectory()
        self.tmp_src = tempfile.TemporaryDirectory()

        self.storage_dir = os.path.join(self.tmp_storage.name, 'storage')
        self.ctrl = PredictorsController(self.storage_dir)

    def tearDown(self):
        self.tmp_storage.cleanup()
        self.tmp_src.cleanup()

    def make_py(self, content: str, basename: str) -> str:
        if not basename.endswith('.py'):
            basename = basename + '.py'
        path = os.path.join(self.tmp_src.name, basename)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def make_binary(self, obj: object, name: str, suffix: str = '.pkl', use_joblib: bool = True) -> str:
        filename = name + suffix
        path = os.path.join(self.tmp_src.name, filename)
        if use_joblib:
            joblib.dump(obj, path)
        else:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        return path

    def test_initial_list_empty(self):
        self.assertEqual(self.ctrl.list(), [])

    def test_add_and_get_module_predictor(self):
        src = self.make_py(
            """
model = 'm'
preprocessor = 'p'
def predict(data):
    return 'got:' + str(data)
""", 'mod1'
        )
        self.ctrl.add('mod1', src)
        self.assertIn('mod1', self.ctrl.list())
        idx_path = Path(self.storage_dir) / 'index.json'
        with open(idx_path) as f:
            idx = json.load(f)
        self.assertIn('mod1', idx)
        info = idx['mod1']
        self.assertTrue(info['path'].endswith('mod1.py'))
        self.assertEqual(info['ext'], '.py')
        bp = self.ctrl.get('mod1')
        self.assertIsInstance(bp, BasePredictor)
        self.assertEqual(bp.predict('X'), 'got:X')

    def test_add_and_get_binary_predictor(self):
        dummy = DummyPredictor()
        src = self.make_binary(dummy, 'bin1', suffix='.pkl', use_joblib=True)
        self.ctrl.add('bin1', src)

        self.assertIn('bin1', self.ctrl.list())
        bp = self.ctrl.get('bin1')
        self.assertIsInstance(bp, BasePredictor)
        self.assertEqual(bp.predict('foo'), 'predicted:foo')

    def test_add_duplicate_name_raises(self):
        src = self.make_py("def predict(data): return data", 'dup')
        self.ctrl.add('dup', src)
        with self.assertRaises(ValueError) as cm:
            self.ctrl.add('dup', src)
        self.assertIn("already exists", str(cm.exception))

    def test_add_nonexistent_source_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.ctrl.add('nope', '/does/not/exist.py')

    def test_add_unsupported_extension_raises(self):
        txt = os.path.join(self.tmp_src.name, 'bad.txt')
        with open(txt, 'w') as f:
            f.write('hello')
        with self.assertRaises(ValueError) as cm:
            self.ctrl.add('bad', txt)
        self.assertIn("Unsupported extension", str(cm.exception))

    def test_get_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.ctrl.get('missing')

    def test_update_module_predictor(self):
        src1 = self.make_py("def predict(x): return 'v1:'+str(x)", 'u1')
        self.ctrl.add('u1', src1)
        src2 = self.make_py("def predict(x): return 'v2:'+str(x)", 'u1_v2')
        self.ctrl.update('u1', src2)
        idx_path = Path(self.storage_dir) / 'index.json'
        with open(idx_path) as f:
            idx = json.load(f)
        info = idx['u1']
        self.assertTrue(Path(info['path']).name.startswith('u1'))
        bp = self.ctrl.get('u1')
        self.assertEqual(bp.predict(5), 'v2:5')

    def test_update_nonexistent_raises(self):
        dummy = DummyPredictor()
        src = self.make_binary(dummy, 'nx', '.pkl', use_joblib=True)
        with self.assertRaises(KeyError):
            self.ctrl.update('nx', src)

    def test_update_unsupported_extension_raises(self):
        src1 = self.make_py("def predict(x): return x", 'good')
        self.ctrl.add('good', src1)
        bad = os.path.join(self.tmp_src.name, 'bad.doc')
        with open(bad, 'w') as f:
            f.write('yo')
        with self.assertRaises(ValueError):
            self.ctrl.update('good', bad)

    def test_remove_predictor(self):
        src = self.make_py("def predict(x): return x", 'toremove')
        self.ctrl.add('toremove', src)
        self.assertIn('toremove', self.ctrl.list())
        idx_path = Path(self.storage_dir) / 'index.json'
        with open(idx_path) as f:
            path_before = json.load(f)['toremove']['path']
        self.assertTrue(Path(path_before).exists())
        self.ctrl.remove('toremove')
        self.assertNotIn('toremove', self.ctrl.list())
        self.assertFalse(Path(path_before).exists())
        with self.assertRaises(KeyError):
            self.ctrl.remove('toremove')


if __name__ == '__main__':
    unittest.main()
