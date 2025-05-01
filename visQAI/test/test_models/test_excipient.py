import unittest
import uuid
from src.model.excipient import (
    BaseExcipient,
    VisQExcipient,
    ExcipientProfile,
    ConcentrationUnit
)


class TestConcentrationUnit(unittest.TestCase):
    def test_str(self):
        self.assertEqual(str(ConcentrationUnit.MOLAR), "M")
        self.assertEqual(str(ConcentrationUnit.MILLIMOLAR), "mM")
        self.assertEqual(str(ConcentrationUnit.MICROGRAM_PER_ML), "μg/mL")

    def test_all_units_unique(self):
        values = [str(unit) for unit in ConcentrationUnit]
        self.assertEqual(len(values), len(set(values)))


class TestBaseExcipient(unittest.TestCase):
    def test_equality_and_hash(self):
        uid = uuid.uuid4()
        e1 = BaseExcipient(name="Test", etype="Generic", id=uid)
        e2 = BaseExcipient(name="Other", etype="Generic", id=uid)
        self.assertEqual(e1, e2)
        self.assertEqual(hash(e1), hash(e2))
        e3 = BaseExcipient(name="Test", etype="Generic", id=uuid.uuid4())
        self.assertNotEqual(e1, e3)

    def test_not_equal_to_other_types(self):
        e = BaseExcipient(name="X", etype="Generic")
        self.assertFalse(e == "X")
        self.assertNotEqual(e, 123)

    def test_name_setter_valid(self):
        e = BaseExcipient(name="Orig", etype="Generic")
        e.name = " NewName "
        self.assertEqual(e.name, "NewName")

    def test_name_setter_invalid(self):
        e = BaseExcipient(name="Test", etype="Generic")
        with self.assertRaises(ValueError):
            e.name = ""
        with self.assertRaises(ValueError):
            e.name = " "
        with self.assertRaises(TypeError):
            e.name = None

    def test_etype_setter_valid(self):
        e = BaseExcipient(name="Name", etype="Type")
        e.etype = " NewType "
        self.assertEqual(e.etype, "NewType")

    def test_etype_setter_invalid(self):
        e = BaseExcipient(name="Name", etype="Type")
        with self.assertRaises(ValueError):
            e.etype = ""
        with self.assertRaises(ValueError):
            e.etype = "  "
        with self.assertRaises(TypeError):
            e.etype = None


class TestVisQExcipient(unittest.TestCase):
    def test_equality_and_hash(self):
        v1 = VisQExcipient(name="Sucrose", etype="Sugar",
                           concentration=1.0, unit=ConcentrationUnit.MOLAR)
        v2 = VisQExcipient(name="Sucrose", etype="Sugar",
                           concentration=1.0, unit=ConcentrationUnit.MOLAR)
        self.assertEqual(v1, v2)
        self.assertEqual(hash(v1), hash(v2))
        v3 = VisQExcipient(name="Sucrose", etype="Sugar",
                           concentration=2.0, unit=ConcentrationUnit.MOLAR)
        self.assertNotEqual(v1, v3)
        v4 = VisQExcipient(name="Sucrose", etype="Sugar",
                           concentration=1.0, unit=ConcentrationUnit.MILLIMOLAR)
        self.assertNotEqual(v1, v4)

    def test_not_equal_to_base(self):
        v = VisQExcipient(name="Buffer", etype="Buffer",
                          concentration=0.1, unit=ConcentrationUnit.PERCENT_W_V)
        b = BaseExcipient(name="Buffer", etype="Buffer")
        self.assertNotEqual(v, b)

    def test_setters_valid(self):
        v = VisQExcipient(name="Name", etype="Type",
                          concentration=1.0, unit=ConcentrationUnit.MOLAR)
        v.etype = " NewType "
        self.assertEqual(v.etype, "NewType")
        v.concentration = 2
        self.assertEqual(v.concentration, 2.0)
        v.unit = ConcentrationUnit.MILLIMOLAR
        self.assertEqual(v.unit, ConcentrationUnit.MILLIMOLAR)

    def test_setters_invalid(self):
        v = VisQExcipient(name="Name", etype="Type",
                          concentration=1.0, unit=ConcentrationUnit.MOLAR)
        with self.assertRaises(ValueError):
            v.etype = ""
        with self.assertRaises(ValueError):
            v.etype = "  "
        with self.assertRaises(TypeError):
            v.etype = None
        with self.assertRaises(TypeError):
            v.concentration = "a"
        with self.assertRaises(ValueError):
            v.concentration = -1
        with self.assertRaises(TypeError):
            v.unit = "mM"


class TestExcipientProfile(unittest.TestCase):
    def setUp(self):
        self.base = BaseExcipient(name="Buffer", etype="Buffer")
        self.profile = ExcipientProfile(self.base)

    def test_add_and_get_variations(self):
        self.profile.add_variation(
            concentration=0.5, unit=ConcentrationUnit.PERCENT_W_V
        )
        vars = self.profile.get_variations()
        self.assertEqual(len(vars), 1)
        var = vars[0]
        self.assertEqual(var.name, "Buffer")
        self.assertEqual(var.etype, "Buffer")
        self.assertEqual(var.concentration, 0.5)
        self.assertEqual(var.unit, ConcentrationUnit.PERCENT_W_V)

    def test_duplicate_variation_raises(self):
        self.profile.add_variation(
            concentration=1.0, unit=ConcentrationUnit.MILLIGRAM_PER_ML
        )
        with self.assertRaises(ValueError):
            self.profile.add_variation(
                concentration=1.0, unit=ConcentrationUnit.MILLIGRAM_PER_ML
            )

    def test_distinct_units_allowed(self):
        self.profile.add_variation(
            concentration=1.0, unit=ConcentrationUnit.PERCENT_W_V
        )
        self.profile.add_variation(
            concentration=1.0, unit=ConcentrationUnit.PERCENT_V_V
        )
        self.assertEqual(len(self.profile.get_variations()), 2)

    def test_variations_return_copy(self):
        self.profile.add_variation(
            concentration=0.2, unit=ConcentrationUnit.MILLIMOLAR
        )
        vars1 = self.profile.get_variations()
        vars1.append(None)
        self.assertNotIn(None, self.profile.get_variations())

    def test_base_excipient_setter_valid(self):
        new_base = BaseExcipient(name="NewBase", etype="NewBase")
        self.profile.base_excipient = new_base
        self.assertIs(self.profile.base_excipient, new_base)

    def test_base_excipient_setter_invalid(self):
        with self.assertRaises(TypeError):
            self.profile.base_excipient = "NotAnExcipient"


if __name__ == "__main__":
    unittest.main()
