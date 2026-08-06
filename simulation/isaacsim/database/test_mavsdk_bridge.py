#!/usr/bin/env python3
"""Regression tests for MAVSDK process lifecycle handling."""

import importlib.util
import sys
import types
import unittest

_SCIPY_STUBBED = importlib.util.find_spec("scipy") is None
if _SCIPY_STUBBED:
    scipy_module = types.ModuleType("scipy")
    spatial_module = types.ModuleType("scipy.spatial")
    transform_module = types.ModuleType("scipy.spatial.transform")
    transform_module.Rotation = object
    scipy_module.spatial = spatial_module
    spatial_module.transform = transform_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.spatial"] = spatial_module
    sys.modules["scipy.spatial.transform"] = transform_module

from mavsdk_bridge import MavsdkOffboardBridge

if _SCIPY_STUBBED:
    sys.modules.pop("scipy.spatial.transform", None)
    sys.modules.pop("scipy.spatial", None)
    sys.modules.pop("scipy", None)


class _FakeProcess:
    def __init__(self):
        self.wait_timeout = None

    def wait(self, timeout=None):
        self.wait_timeout = timeout


class _FakeSystem:
    def __init__(self):
        self._server_process = _FakeProcess()
        self.stop_called = False

    def _stop_mavsdk_server(self):
        self.stop_called = True


class MavsdkBridgeLifecycleTest(unittest.TestCase):
    def test_owned_server_is_stopped_and_reaped(self):
        drone = _FakeSystem()

        MavsdkOffboardBridge._stop_owned_mavsdk_server(drone)

        self.assertTrue(drone.stop_called)
        self.assertEqual(drone._server_process.wait_timeout, 2.0)

    def test_external_server_is_not_stopped(self):
        class ExternalSystem:
            _server_process = None

        MavsdkOffboardBridge._stop_owned_mavsdk_server(ExternalSystem())


if __name__ == "__main__":
    unittest.main()
