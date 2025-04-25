import unittest

from src.model.viscosity import ViscosityPoint, ViscosityProfile


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


if __name__ == '__main__':
    unittest.main()
