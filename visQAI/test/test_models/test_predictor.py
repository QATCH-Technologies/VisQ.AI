# visQAI/test/model/test_predictor.py
import unittest
from unittest import mock
import tempfile
import os
import pickle
import joblib

from src.model.predictor import (
    load_module_from_file,
    load_binary_predictor,
    BasePredictor
)


class DummyPredictor:
    """A serializable predictor with a .predict method."""

    def __init__(self):
        self.model = 'dummy_model'
        self.preprocessor = 'dummy_preprocessor'

    def predict(self, data, *args, **kwargs):
        return f'predicted:{data}'


class NoPred:
    """A serializable class *without* a .predict method."""

    def __init__(self):
        self.model = 'nop'
        self.preprocessor = 'nop'


class TestPredictor(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def make_py_module(self, content: str, name: str) -> str:
        path = os.path.join(self.tmpdir.name, f'{name}.py')
        with open(path, 'w') as f:
            f.write(content)
        return path

    def make_binary(self, obj: object, suffix: str = '.pkl', use_joblib: bool = True) -> str:
        path = os.path.join(self.tmpdir.name, f'dummy{suffix}')
        if use_joblib:
            joblib.dump(obj, path)
        else:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        return path

    # ─── load_module_from_file ─────────────────────────────────────────────

    def test_load_module_success(self):
        src = """
model = 'mod'
preprocessor = 'prep'
def predict(data, *args, **kwargs):
    return f'ok:{data}'
"""
        mod_path = self.make_py_module(src, 'mod1')
        mod = load_module_from_file(mod_path)
        self.assertTrue(callable(mod.predict))
        self.assertEqual(mod.predict('x'), 'ok:x')
        self.assertEqual(mod.model, 'mod')
        self.assertEqual(mod.preprocessor, 'prep')

    def test_load_module_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_module_from_file('/no/such/file.py')

    def test_load_module_spec_none(self):
        src = "x = 1"
        mod_path = self.make_py_module(src, 'mod2')
        with mock.patch('importlib.util.spec_from_file_location', return_value=None):
            with self.assertRaises(ImportError):
                load_module_from_file(mod_path)

    def test_load_binary_joblib(self):
        dummy = DummyPredictor()
        p = self.make_binary(dummy, '.pkl', use_joblib=True)
        loaded = load_binary_predictor(p)
        self.assertIsInstance(loaded, DummyPredictor)
        self.assertEqual(loaded.predict('foo'), 'predicted:foo')

    def test_load_binary_pickle_ext(self):
        dummy = DummyPredictor()
        p = self.make_binary(dummy, '.pickle', use_joblib=False)
        loaded = load_binary_predictor(p)
        self.assertIsInstance(loaded, DummyPredictor)
        self.assertEqual(loaded.predict(123), 'predicted:123')

    def test_load_binary_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_binary_predictor('/no/where.pkl')

    def test_load_binary_unsupported_ext(self):
        txt = os.path.join(self.tmpdir.name, 'file.txt')
        with open(txt, 'w') as f:
            f.write('oops')
        with self.assertRaises(ValueError):
            load_binary_predictor(txt)

    def test_base_predictor_module(self):
        src = """
model = 'm1'
preprocessor = 'p1'
def predict(data, *args, **kwargs):
    return f'modpred:{data}'
"""
        mod_path = self.make_py_module(src, 'predmod')
        bp = BasePredictor(mod_path)
        self.assertEqual(bp.predict('in'), 'modpred:in')
        self.assertEqual(bp.model, 'm1')
        self.assertEqual(bp.preprocessor, 'p1')
        self.assertEqual(bp.predictor_type, 'module')
        self.assertIn('type=module', repr(bp))

        # reload works
        bp.reload()
        self.assertEqual(bp.predict('again'), 'modpred:again')

        # load new module
        src2 = """
model = 'm2'
preprocessor = 'p2'
def predict(data):
    return 'new:' + str(data)
"""
        mod_path2 = self.make_py_module(src2, 'predmod2')
        bp.load(mod_path2)
        self.assertEqual(bp.predict('X'), 'new:X')
        self.assertEqual(bp.model, 'm2')
        self.assertEqual(bp.preprocessor, 'p2')

        # path setter
        bp.path = mod_path
        self.assertEqual(bp.path, mod_path)
        self.assertEqual(bp.predict('in'), 'modpred:in')

    def test_base_predictor_module_missing_predict(self):
        bad_src = "x = 42"
        bad_path = self.make_py_module(bad_src, 'nopredict')
        with self.assertRaises(AttributeError):
            BasePredictor(bad_path)

    # ─── BasePredictor with binary predictors ───────────────────────────────

    def test_base_predictor_binary(self):
        dummy = DummyPredictor()
        bin_path = self.make_binary(dummy, '.pkl', use_joblib=True)
        bp = BasePredictor(bin_path)
        self.assertEqual(bp.predict('d'), 'predicted:d')
        self.assertEqual(bp.predictor_type, 'binary')
        self.assertIn('type=binary', repr(bp))

    def test_base_predictor_binary_missing_predict(self):
        # Uses module‐level NoPred so pickling works
        bin_path = self.make_binary(NoPred(), '.pkl', use_joblib=True)
        with self.assertRaises(AttributeError):
            BasePredictor(bin_path)

    def test_predict_no_loaded(self):
        src = """
model = 'm'
preprocessor = 'p'
def predict(data):
    return data
"""
        mod_path = self.make_py_module(src, 'mod3')
        bp = BasePredictor(mod_path)
        bp._module = None
        bp._instance = None
        with self.assertRaises(RuntimeError):
            bp.predict('x')


if __name__ == '__main__':
    unittest.main()
