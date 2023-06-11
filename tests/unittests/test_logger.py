import unittest
from flight_maneuvers.models.log import StandardOutLogger

class TestLogger(unittest.TestCase):
    
    def test_progressbar(self):
        StandardOutLogger()
        self.assertTrue(True)