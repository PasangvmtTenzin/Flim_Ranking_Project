# tests/test_analysis.py
import unittest
from src import analysis

class TestAnalysis(unittest.TestCase):
    def test_load_and_clean_imdb_data(self):
        data = analysis.load_and_clean_imdb_data()
        self.assertFalse(data.empty)
        self.assertIn('averageRating', data.columns)
    
    def test_calculate_weak_impact(self):
        data = analysis.load_and_clean_imdb_data()
        weak_impact = analysis.calculate_weak_impact(data)
        self.assertIn('weakImpact', weak_impact.columns)
    
    def test_calculate_strong_impact(self):
        data = analysis.load_and_clean_imdb_data()
        strong_impact = analysis.calculate_strong_impact(data)
        self.assertIn('strongImpact', strong_impact.columns)

if __name__ == '__main__':
    unittest.main()
