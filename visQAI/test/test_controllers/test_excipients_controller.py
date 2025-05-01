import unittest
import uuid

from src.controllers.excipients_controller import ExcipientsController
from src.db.sqlite_db import sqlite3, _enable_foreign_keys
from src.model.excipient import (
    BaseExcipient,
    VisQExcipient,
    ExcipientProfile,
    ConcentrationUnit
)


class TestExcipientsController(unittest.TestCase):
    def setUp(self):
        # Initialize controller with in-memory SQLite for isolation
        self.ctrl = ExcipientsController()
        # Reconfigure DB to use in-memory
        self.ctrl.db.conn.close()
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        _enable_foreign_keys(conn)
        self.ctrl.db.conn = conn
        # Recreate tables
        self.ctrl.db._create_tables()

    # Base excipient CRUD
    def test_add_and_get_base_excipient(self):
        base = BaseExcipient(name='Base1', etype='TestType')
        created = self.ctrl.add_base_excipient(base)
        self.assertIsInstance(created, BaseExcipient)
        self.assertEqual(created.name, 'Base1')
        # Check list and get
        bases = self.ctrl.list_base_excipients()
        self.assertIn(created, bases)
        fetched = self.ctrl.get_base_excipient(str(created.id))
        self.assertEqual(fetched.id, created.id)
        self.assertEqual(fetched.name, 'Base1')

    def test_add_base_invalid_type(self):
        with self.assertRaises(TypeError):
            self.ctrl.add_base_excipient('not a base')

    def test_get_base_invalid_id(self):
        with self.assertRaises(TypeError):
            self.ctrl.get_base_excipient(123)
        with self.assertRaises(ValueError):
            self.ctrl.get_base_excipient('invalid-uuid')
        # Non-existent UUID returns None
        new_id = str(uuid.uuid4())
        self.assertIsNone(self.ctrl.get_base_excipient(new_id))

    def test_delete_base_excipient(self):
        base = self.ctrl.add_base_excipient(
            BaseExcipient(name='ToDelete', etype='TestType'))
        bid = str(base.id)
        # Delete valid
        self.ctrl.delete_base_excipient(bid)
        self.assertIsNone(self.ctrl.get_base_excipient(bid))
        # Invalid type
        with self.assertRaises(TypeError):
            self.ctrl.delete_base_excipient(123)
        # Invalid uuid
        with self.assertRaises(ValueError):
            self.ctrl.delete_base_excipient('bad-uuid')

    # Variation CRUD
    def test_add_and_list_variations(self):
        base = self.ctrl.add_base_excipient(
            BaseExcipient(name='B', etype='TestType'))
        var = VisQExcipient(name='B', etype='Type',
                            concentration=1.0, unit=ConcentrationUnit.MOLAR)
        created = self.ctrl.add_variation(str(base.id), var)
        self.assertIsInstance(created, VisQExcipient)
        vars_list = self.ctrl.list_variations()
        self.assertTrue(any(v.id == created.id for v in vars_list))

    def test_add_variation_invalid(self):
        # base_id type
        with self.assertRaises(TypeError):
            self.ctrl.add_variation(123, None)
        with self.assertRaises(ValueError):
            self.ctrl.add_variation('bad-uuid', None)
        # exc type
        base = self.ctrl.add_base_excipient(
            BaseExcipient(name='B', etype='TestType'))
        with self.assertRaises(TypeError):
            self.ctrl.add_variation(str(base.id), 'not an excipient')

    def test_get_update_delete_variation(self):
        base = self.ctrl.add_base_excipient(
            BaseExcipient(name='Base', etype='TestType'))
        var = VisQExcipient(name='Base', etype='E',
                            concentration=2.0, unit=ConcentrationUnit.MILLIMOLAR)
        created = self.ctrl.add_variation(str(base.id), var)
        vid = str(created.id)
        # get
        fetched = self.ctrl.get_variation(vid)
        self.assertEqual(fetched.id, created.id)
        # update valid
        created.concentration = 3.0
        self.ctrl.update_variation(created)
        updated = self.ctrl.get_variation(vid)
        self.assertEqual(updated.concentration, 3.0)
        # update invalid
        with self.assertRaises(TypeError):
            self.ctrl.update_variation(None)
        bad = VisQExcipient(name='X', etype='Y',
                            concentration=1.0, unit=ConcentrationUnit.MOLAR)
        with self.assertRaises(ValueError):
            self.ctrl.update_variation(bad)
        # delete
        self.ctrl.delete_variation(vid)
        self.assertIsNone(self.ctrl.get_variation(vid))
        with self.assertRaises(TypeError):
            self.ctrl.delete_variation(123)
        with self.assertRaises(ValueError):
            self.ctrl.delete_variation('bad-uuid')

    # Profile retrieval
    def test_get_profile(self):
        base = self.ctrl.add_base_excipient(
            BaseExcipient(name='Prof', etype='TestType'))
        base_id = str(base.id)
        # no variations => empty profile
        prof = self.ctrl.get_profile(base_id)
        self.assertIsInstance(prof, ExcipientProfile)
        self.assertEqual(prof.get_variations(), [])
        # add variations
        v1 = self.ctrl.add_variation(base_id, VisQExcipient(
            name='Prof', etype='T1', concentration=1.0, unit=ConcentrationUnit.MOLAR
        ))
        v2 = self.ctrl.add_variation(base_id, VisQExcipient(
            name='Prof', etype='T2', concentration=2.0, unit=ConcentrationUnit.MILLIGRAM_PER_ML
        ))
        prof2 = self.ctrl.get_profile(base_id)
        vars_in_profile = prof2.get_variations()
        self.assertEqual({v1.concentration, v2.concentration}, {
                         p.concentration for p in vars_in_profile})

    def test_get_profile_invalid(self):
        with self.assertRaises(TypeError):
            self.ctrl.get_profile(123)
        with self.assertRaises(ValueError):
            self.ctrl.get_profile('bad-uuid')
        # non-existent base returns None
        self.assertIsNone(self.ctrl.get_profile(str(uuid.uuid4())))


if __name__ == '__main__':
    unittest.main()
