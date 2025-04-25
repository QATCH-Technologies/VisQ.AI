
import unittest

from src.model.formulation import Formulation
from src.model.excipient import VisQExcipient, ConcentrationUnit
from src.model.viscosity import ViscosityProfile, ViscosityPoint


class TestFormulation(unittest.TestCase):
    def test_init_name_validation(self):
        # Valid name
        f = Formulation("TestForm")
        self.assertEqual(f.name, "TestForm")
        # Non-string name
        with self.assertRaises(TypeError):
            Formulation(123)
        # Empty name
        with self.assertRaises(ValueError):
            Formulation("")
        with self.assertRaises(ValueError):
            Formulation("   ")

    def test_name_setter_valid_and_invalid(self):
        f = Formulation("Name1")
        f.name = " NewName "
        self.assertEqual(f.name, "NewName")
        with self.assertRaises(TypeError):
            f.name = None
        with self.assertRaises(ValueError):
            f.name = ""

    def test_add_excipient_and_duplicates(self):
        f = Formulation("F")
        # Invalid type
        with self.assertRaises(TypeError):
            f.add_excipient("not an excipient")
        # Add valid
        exc = VisQExcipient(name="Sucrose", etype="Sugar",
                            concentration=1.0, unit=ConcentrationUnit.MOLAR)
        f.add_excipient(exc)
        self.assertIn(exc, f.excipients)
        # Duplicate: use equal instance
        exc2 = VisQExcipient(name="Sucrose", etype="Sugar",
                             concentration=1.0, unit=ConcentrationUnit.MOLAR)
        f.add_excipient(exc2)
        self.assertEqual(len(f.excipients), 1)

    def test_excipients_returns_copy(self):
        f = Formulation("F")
        exc = VisQExcipient(name="Buffer", etype="Buffer",
                            concentration=0.5, unit=ConcentrationUnit.MILLIMOLAR)
        f.add_excipient(exc)
        lst = f.excipients
        lst.append(None)
        self.assertNotIn(None, f.excipients)

    def test_remove_excipient_by_name(self):
        f = Formulation("F")
        exc1 = VisQExcipient(name="A", etype="TypeA",
                             concentration=1, unit=ConcentrationUnit.MOLAR)
        exc2 = VisQExcipient(name="B", etype="TypeB",
                             concentration=2, unit=ConcentrationUnit.MILLIMOLAR)
        f.add_excipient(exc1)
        f.add_excipient(exc2)
        # Invalid name type
        with self.assertRaises(TypeError):
            f.remove_excipient_by_name(123)
        # Remove A
        f.remove_excipient_by_name("A")
        self.assertNotIn(exc1, f.excipients)
        self.assertIn(exc2, f.excipients)

    def test_get_excipients_by_type(self):
        f = Formulation("F")
        exc1 = VisQExcipient(name="X", etype="TypeX",
                             concentration=1, unit=ConcentrationUnit.PERCENT_W_V)
        exc2 = VisQExcipient(name="Y", etype="TypeY",
                             concentration=2, unit=ConcentrationUnit.PERCENT_V_V)
        f.add_excipient(exc1)
        f.add_excipient(exc2)
        # Invalid type param
        with self.assertRaises(TypeError):
            f.get_excipients_by_type(456)
        # Filter
        result = f.get_excipients_by_type("typex")
        self.assertEqual(result, [exc1])

    def test_notes_property(self):
        f = Formulation("F")
        # Default
        self.assertEqual(f.notes, "")
        # Valid set
        f.notes = "Some notes"
        self.assertEqual(f.notes, "Some notes")
        # Invalid set
        with self.assertRaises(TypeError):
            f.notes = None

    def test_summary_default_and_content(self):
        f = Formulation("Test")
        summary = f.summary()
        self.assertIn("Formulation: Test", summary)
        self.assertIn("No excipients added.", summary)
        self.assertIn("No viscosity profile set.", summary)
        # Add excipient and notes
        exc = VisQExcipient(name="Sucrose", etype="Sugar",
                            concentration=1.0, unit=ConcentrationUnit.MOLAR)
        f.add_excipient(exc)
        f.notes = "Note1"
        summary2 = f.summary()
        self.assertIn("Sucrose (Sugar) @ 1.0M", summary2)
        self.assertIn("Notes: Note1", summary2)


class TestViscosityPoint(unittest.TestCase):
    def test_valid_point(self):
        p = ViscosityPoint(shear_rate=10.0, viscosity=5.5)
        self.assertEqual(p.shear_rate, 10.0)
        self.assertEqual(p.viscosity, 5.5)

    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            ViscosityPoint(shear_rate="fast", viscosity=1.0)
        with self.assertRaises(TypeError):
            ViscosityPoint(shear_rate=1.0, viscosity="thick")

    def test_negative_values(self):
        with self.assertRaises(ValueError):
            ViscosityPoint(shear_rate=-1.0, viscosity=1.0)
        with self.assertRaises(ValueError):
            ViscosityPoint(shear_rate=1.0, viscosity=-0.5)


class TestViscosityProfile(unittest.TestCase):
    def setUp(self):
        self.profile = ViscosityProfile()

    def test_add_and_list_points(self):
        self.profile.add_point(1.0, 10.0)
        self.profile.add_point(3.0, 30.0)
        self.profile.add_point(2.0, 20.0)
        pts = self.profile.points
        # Should be sorted by shear_rate
        self.assertEqual([p.shear_rate for p in pts], [1.0, 2.0, 3.0])
        self.assertEqual([p.viscosity for p in pts], [10.0, 20.0, 30.0])

    def test_add_duplicate(self):
        self.profile.add_point(5.0, 50.0)
        with self.assertRaises(ValueError):
            self.profile.add_point(5.0, 55.0)

    def test_get_point(self):
        self.profile.add_point(2.5, 25.0)
        p = self.profile.get_point(2.5)
        self.assertIsNotNone(p)
        self.assertEqual(p.viscosity, 25.0)
        # Missing returns None
        self.assertIsNone(self.profile.get_point(9.9))
        # Invalid type
        with self.assertRaises(TypeError):
            self.profile.get_point("fast")

    def test_update_point(self):
        self.profile.add_point(4.0, 40.0)
        self.profile.update_point(4.0, 44.0)
        p = self.profile.get_point(4.0)
        self.assertEqual(p.viscosity, 44.0)
        # Update missing
        with self.assertRaises(KeyError):
            self.profile.update_point(8.0, 80.0)
        # Invalid types
        with self.assertRaises(TypeError):
            self.profile.update_point("rate", 10.0)
        with self.assertRaises(TypeError):
            self.profile.update_point(4.0, "visc")

    def test_remove_point(self):
        self.profile.add_point(7.0, 70.0)
        self.profile.remove_point(7.0)
        self.assertIsNone(self.profile.get_point(7.0))
        # Remove missing
        with self.assertRaises(KeyError):
            self.profile.remove_point(7.0)
        # Invalid type
        with self.assertRaises(TypeError):
            self.profile.remove_point(None)

    def test_clear(self):
        self.profile.add_point(1.0, 10.0)
        self.profile.add_point(2.0, 20.0)
        self.profile.clear()
        self.assertEqual(self.profile.points, [])

    def test_integration_with_real_viscosity_profile(self):
        """
        Test that a real ViscosityProfile can be built, assigned,
        and queried via the Formulation object.
        """
        f = Formulation("F")
        vp = ViscosityProfile()
        # Build profile
        vp.add_point(1.0, 10.0)
        vp.add_point(2.0, 20.0)
        # Assign to formulation
        f.viscosity_profile = vp
        # Access underlying profile
        points = f.viscosity_profile.points
        self.assertEqual(len(points), 2)
        rates = [p.shear_rate for p in points]
        self.assertEqual(rates, [1.0, 2.0])
        viscosities = [p.viscosity for p in points]
        self.assertEqual(viscosities, [10.0, 20.0])


if __name__ == '__main__':
    unittest.main()
