import unittest
from unittest.mock import patch
import importlib
from . import deployment

class DeploymentTests(unittest.TestCase):
    def tearDown(self):
        with patch.dict('os.environ', {}, clear=True):
            importlib.reload(deployment)

    def test_default_is_loopback(self):
        with patch.dict('os.environ', {}, clear=True):
            settings = importlib.reload(deployment)
            self.assertEqual(settings.COORDINATOR_URL, 'https://127.0.0.1:8843')

    def test_rejects_public_unspecified_and_multicast(self):
        for host in ('8.8.8.8', '0.0.0.0', '224.0.0.1'):
            with patch.dict('os.environ', {'FLY_COORDINATOR_HOST': host}):
                with self.assertRaises(ValueError): importlib.reload(deployment)

    def test_explicit_private_address_and_origin(self):
        with patch.dict('os.environ', {'FLY_COORDINATOR_HOST':'192.168.30.1',
                                     'FLY_DASHBOARD_ORIGINS':'https://dashboard.example.test'}):
            settings=importlib.reload(deployment)
            self.assertEqual(settings.COORDINATOR_URL, 'https://192.168.30.1:8843')
            self.assertIn('https://dashboard.example.test', settings.ORIGINS)
