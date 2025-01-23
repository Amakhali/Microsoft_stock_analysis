import unittest
from scripts.preprocess_data import preprocess_data

class TestModel(unittest.TestCase):
    def test_preprocess_data(self):
        data_path = "../data/msft_stock_data.csv"
        scaled_data, scaler = preprocess_data(data_path)
        self.assertEqual(scaled_data.shape[1], 5)  # Check if all 5 features are present

if __name__ == '__main__':
    unittest.main()