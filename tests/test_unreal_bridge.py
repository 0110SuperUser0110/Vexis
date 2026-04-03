from __future__ import annotations

import json
import unittest
import urllib.request
from time import sleep

from interface.unreal_bridge import UnrealPresenceBridgeServer


class TestUnrealBridge(unittest.TestCase):
    def test_bridge_serves_latest_snapshot(self) -> None:
        bridge = UnrealPresenceBridgeServer(port=8766)
        bridge.start()
        try:
            bridge.publish_snapshot(
                {
                    "visual_state": "thinking",
                    "status_text": "reviewing",
                    "is_thinking": True,
                    "is_speaking": False,
                    "thought_lines": ["parsing input", "routing front layer"],
                }
            )
            sleep(0.05)
            with urllib.request.urlopen('http://127.0.0.1:8766/v1/presence', timeout=2.0) as response:
                payload = json.loads(response.read().decode('utf-8'))
            self.assertEqual(payload['visual_state'], 'thinking')
            self.assertEqual(payload['status_text'], 'reviewing')
            self.assertTrue(payload['is_thinking'])
            self.assertEqual(payload['thought_lines'][0], 'parsing input')
            self.assertTrue(payload['bridge_online'])
        finally:
            bridge.stop()


if __name__ == '__main__':
    unittest.main()
