import unittest
import plant_analysis as pa


class TestPlantAnalysis(unittest.TestCase):

    # Define tests as methods here
    def test_select_plant_plant_id_not_in_database(self):
        """Expects error to be raised when plant not in database."""
        plant_id = 1000
        with self.assertRaises(IndexError):
            result = pa.select_plant(
                plant_id, pa.plant_table, pa.readings_table)


if __name__ == '__main__':
    unittest.main()
