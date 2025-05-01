#!/usr/bin/env python3
"""
Unit tests for FormulationsController.
"""
import unittest
from uuid import UUID
from pathlib import Path

from src.db.sqlite_db import SQLiteDB
from src.controllers.formulations_controller import FormulationsController
from src.controllers.excipients_controller import ExcipientsController
from src.model.formulation import Formulation
from src.model.excipient import BaseExcipient, VisQExcipient, ConcentrationUnit
from src.model.viscosity import ViscosityProfile


class TestFormulationsController(unittest.TestCase):
    def setUp(self):
        # Initialize an in-memory SQLiteDB for testing
        self.db = SQLiteDB(Path(':memory:'))
        # Use the same DB in both controllers
        self.ex_ctrl = ExcipientsController()
        self.ex_ctrl.db = self.db
        self.ctrl = FormulationsController(self.db)

        # Create a base excipient and capture its ID
        base = self.ex_ctrl.add_base_excipient(
            BaseExcipient(name='TestBase', etype='TestType')
        )
        self.base_id = str(base.id)

        # Create a variation under that base
        variation = self.ex_ctrl.add_variation(
            self.base_id,
            VisQExcipient(
                name=base.name,
                etype=base.etype,
                concentration=1.0,
                unit=ConcentrationUnit.MOLAR
            )
        )

        self.exc_id = str(variation.id)

    def test_add_and_get_formulation(self):
        # Build a formulation linking the variation and a viscosity profile
        formulation = Formulation(name='FormA')
        var = self.ex_ctrl.get_variation(self.exc_id)
        formulation.add_excipient(var)
        profile_data = {10: 100.0, 20: 200.0}
        formulation.viscosity_profile = ViscosityProfile()
        formulation.viscosity_profile.from_dict(profile_data)
        added = self.ctrl.add_formulation(formulation)
        self.assertEqual(added.id, formulation.id)

        fetched = self.ctrl.get_formulation(str(formulation.id))
        self.assertEqual(fetched.id, formulation.id)
        self.assertEqual(fetched.name, 'FormA')
        self.assertEqual(len(fetched.excipients), 1)
        self.assertEqual(fetched.excipients[0].id, var.id)
        self.assertIsNotNone(fetched.viscosity_profile)
        self.assertEqual(fetched.viscosity_profile.to_dict(), profile_data)

    def test_list_formulations(self):
        f1 = Formulation(name='F1')
        f2 = Formulation(name='F2')
        self.ctrl.add_formulation(f1)
        self.ctrl.add_formulation(f2)
        all_forms = self.ctrl.list_formulations()
        ids = {f.id for f in all_forms}
        self.assertIn(f1.id, ids)
        self.assertIn(f2.id, ids)

    def test_update_formulation(self):
        form = Formulation(name='Initial')
        self.ctrl.add_formulation(form)
        # Modify fields
        form.name = 'Updated'
        form.notes = 'Some notes'
        form.viscosity_profile = ViscosityProfile()
        form.viscosity_profile.add_point(shear_rate=100, viscosity=10)
        # Clear and re-link excipients
        form._excipients.clear()
        var = self.ex_ctrl.get_variation(self.exc_id)
        form.add_excipient(var)

        self.ctrl.update_formulation(form)
        updated = self.ctrl.get_formulation(str(form.id))
        self.assertEqual(updated.name, 'Updated')
        self.assertEqual(updated.notes, 'Some notes')
        self.assertEqual(updated.viscosity_profile.get_point(100), 10)
        self.assertEqual(len(updated.excipients), 1)

    def test_delete_formulation(self):
        form = Formulation(name='ToRemove')
        self.ctrl.add_formulation(form)
        fid = str(form.id)
        # Delete and then verify not found
        self.ctrl.delete_formulation(fid)
        with self.assertRaises(ValueError):
            self.ctrl.get_formulation(fid)

    def test_invalid_inputs(self):
        # get with malformed ID
        with self.assertRaises(TypeError):
            self.ctrl.get_formulation('invalid-uuid')
        # delete with malformed ID
        with self.assertRaises(TypeError):
            self.ctrl.delete_formulation('1234')
        # delete non-existent
        fake_id = str(UUID(int=0))
        with self.assertRaises(ValueError):
            self.ctrl.delete_formulation(fake_id)
        # add invalid type
        with self.assertRaises(TypeError):
            self.ctrl.add_formulation('not-a-formulation')  # type: ignore
        # update invalid type
        with self.assertRaises(TypeError):
            self.ctrl.update_formulation(123)  # type: ignore
        # update non-existent formulation
        new_form = Formulation(name='Ghost')
        with self.assertRaises(ValueError):
            self.ctrl.update_formulation(new_form)

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()
