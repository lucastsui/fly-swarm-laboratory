"""Deployment-local settings; no personal host addresses belong in source."""
import ipaddress
import os

COORDINATOR_HOST = os.environ.get('FLY_COORDINATOR_HOST', '127.0.0.1')
_address = ipaddress.ip_address(COORDINATOR_HOST)
if not _address.is_private or _address.is_unspecified or _address.is_multicast:
    raise ValueError('FLY_COORDINATOR_HOST must name a private or loopback address')
COORDINATOR_URL = f'https://{COORDINATOR_HOST}:8843'
ORIGINS = {'http://localhost:5173', 'http://127.0.0.1:5173'}
ORIGINS.update(value.strip() for value in os.environ.get('FLY_DASHBOARD_ORIGINS', '').split(',') if value.strip())
