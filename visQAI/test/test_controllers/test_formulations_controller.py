import sqlite3
import unittest
import uuid

from src.controllers.excipients_controller import ExcipientsController
from src.controllers.formulations_controller import FormulationsController
from src.db.sqlite_db import SQLiteDB
from src.model.excipient import BaseExcipient, VisQExcipient, ConcentrationUnit
from src.model.formulation import Formulation
from src.model.viscosity import ViscosityProfile, ViscosityPoint


class TestFormulationsController(unittest.TestCase):
    def setUp(self):
        # in-memory SQLite for full isolation
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        # wrap it in our SQLiteDB and recreate tables
        self.shared_db = SQLiteDB()
        self.shared_db.conn.close()
        self.shared_db.conn = conn
        self.shared_db._create_tables()

        # wire controllers to that same DB
        self.exc_ctrl = ExcipientsController()
        self.exc_ctrl.db = self.shared_db

        self.ctrl = FormulationsController()
        self.ctrl.db = self.shared_db
        self.ctrl.exc_ctrl = self.exc_ctrl

    def _make_excipient(self, name='TestBase', etype='T',
                        conc=1.0, unit=ConcentrationUnit.MOLAR):
        """Helper to add one VisQExcipient under a fresh BaseExcipient."""
        base = self.exc_ctrl.add_base_excipient(
            BaseExcipient(name=name, etype=etype))
        var = VisQExcipient(
            name=base.name,
            etype=base.etype,
            concentration=conc,
            unit=unit
        )
        return self.exc_ctrl.add_variation(str(base.id), var)

    def test_add_and_get_formulation(self):
        # create two excipient variations
        v1 = self._make_excipient(name='A', etype='X',
                                  conc=2.5, unit=ConcentrationUnit.MILLIMOLAR)
        v2 = self._make_excipient(name='B', etype='Y',
                                  conc=7.5, unit=ConcentrationUnit.MICROGRAM_PER_ML)

        # build a Formulation
        form = Formulation('MyForm')
        form.add_excipient(v1)
        form.add_excipient(v2)
        form.notes = 'hello'

        # build a viscosity profile with two points
        vp = ViscosityProfile()
        vp.add_point(10.0, 100.0)
        vp.add_point(50.0, 250.0)
        form.viscosity_profile = vp

        # add to DB
        added = self.ctrl.add(form)
        self.assertIsNotNone(added.id, "should assign an ID")

        # retrieve it
        fetched = self.ctrl.get(added.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, 'MyForm')
        self.assertEqual(fetched.notes, 'hello')

        # check viscosity_profile round-trip
        self.assertIsInstance(fetched.viscosity_profile, ViscosityProfile)
        points = fetched.viscosity_profile.points
        # must be sorted by shear_rate
        self.assertEqual(points, [
            ViscosityPoint(10.0, 100.0),
            ViscosityPoint(50.0, 250.0),
        ])

        # excipients match
        fetched_ids = {e.id for e in fetched.excipients}
        self.assertSetEqual(fetched_ids, {v1.id, v2.id})

    def test_all_returns_everything(self):
        # empty initially
        self.assertEqual(self.ctrl.all(), [])

        # add two formulations
        v1 = self._make_excipient()
        f1 = Formulation('F1')
        f1.add_excipient(v1)
        self.ctrl.add(f1)

        v2 = self._make_excipient(name='C2', etype='Z', conc=9.9)
        f2 = Formulation('F2')
        f2.add_excipient(v2)
        # leave notes + viscosity_profile at defaults
        self.ctrl.add(f2)

        names = {f.name for f in self.ctrl.all()}
        self.assertSetEqual(names, {'F1', 'F2'})

    def test_edit_updates_correctly(self):
        # initial
        v0 = self._make_excipient(name='Old', etype='O', conc=1.1)
        form = Formulation('Orig')
        form.add_excipient(v0)
        form.notes = 'N0'
        vp0 = ViscosityProfile()
        vp0.add_point(5.0, 50.0)
        form.viscosity_profile = vp0
        added = self.ctrl.add(form)

        # create another excipient
        v1 = self._make_excipient(name='New', etype='N', conc=4.4)

        # mutate object
        added.name = 'Updated'
        added.notes = 'N1'
        vp1 = ViscosityProfile()
        vp1.add_point(20.0, 200.0)
        added.viscosity_profile = vp1
        added.remove_excipient_by_name(v0.name)
        added.add_excipient(v1)

        self.ctrl.edit(added)

        fetched = self.ctrl.get(added.id)
        self.assertEqual(fetched.name, 'Updated')
        self.assertEqual(fetched.notes, 'N1')
        self.assertEqual(
            fetched.viscosity_profile.points,
            [ViscosityPoint(20.0, 200.0)]
        )
        self.assertEqual([e.id for e in fetched.excipients], [v1.id])

    def test_delete_removes_formulation(self):
        v = self._make_excipient()
        f = Formulation('ToDel')
        f.add_excipient(v)
        added = self.ctrl.add(f)

        self.assertIsNotNone(self.ctrl.get(added.id))
        self.ctrl.delete(added.id)
        self.assertIsNone(self.ctrl.get(added.id))

    def test_get_nonexistent_returns_none(self):
        # generate a random UUID not in DB
        self.assertIsNone(self.ctrl.get(uuid.uuid4()))


if __name__ == '__main__':
    unittest.main()
