#!/usr/bin/env python3
import unittest

from episode_sync import EpisodeBundleSynchronizer


class EpisodeBundleSynchronizerTest(unittest.TestCase):
    def test_explicit_episode_generation_resets_monotonic_clock(self):
        sync = EpisodeBundleSynchronizer({"image", "pose"})
        sync.add(10_000_000_000, "image", "old-image")
        sync.add(
            10_000_000_000, "pose", "old-pose", external_generation=4)
        sync.add(10_100_000_000, "image", "partial-old-image")

        generation, complete, reset = sync.add(
            10_200_000_000, "pose", "new-pose", external_generation=5)
        self.assertTrue(reset)
        self.assertEqual(generation, 1)
        self.assertIsNone(complete)
        self.assertNotIn(10_100_000_000, sync.buckets)
        generation, complete, reset = sync.add(
            10_200_000_000, "image", "new-image")
        self.assertFalse(reset)
        self.assertEqual(complete, {
            "pose": "new-pose", "image": "new-image"})
        self.assertEqual(sync.external_reset_count, 1)

    def test_stale_explicit_generation_is_ignored(self):
        sync = EpisodeBundleSynchronizer({"image", "pose"})
        sync.add(1_000_000_000, "pose", "current", external_generation=7)
        generation, complete, reset = sync.add(
            1_100_000_000, "pose", "stale", external_generation=6)
        self.assertFalse(reset)
        self.assertIsNone(complete)
        self.assertEqual(generation, 0)
        self.assertEqual(sync.external_generation, 7)
        self.assertEqual(sync.stale_external_count, 1)

    def test_clock_rewind_clears_old_buckets_and_completes_new_episode(self):
        sync = EpisodeBundleSynchronizer(
            {"anchors", "stereo", "depth0", "depth1", "depth2", "depth3"},
            retained_stamps=2)
        sync.add(52_300_000_000, "anchors", "old-anchor")
        sync.add(52_400_000_000, "stereo", "old-stereo")

        generation, complete, rewound = sync.add(
            100_000_000, "anchors", "new-anchor")
        self.assertTrue(rewound)
        self.assertEqual(generation, 1)
        self.assertIsNone(complete)
        self.assertNotIn(52_300_000_000, sync.buckets)
        self.assertNotIn(52_400_000_000, sync.buckets)

        sync.add(100_000_000, "stereo", "new-stereo")
        sync.add(100_000_000, "depth0", "new-depth0")
        sync.add(100_000_000, "depth1", "new-depth1")
        sync.add(100_000_000, "depth2", "new-depth2")
        generation, complete, rewound = sync.add(
            100_000_000, "depth3", "new-depth3")
        self.assertFalse(rewound)
        self.assertEqual(generation, 1)
        self.assertEqual(complete, {
            "anchors": "new-anchor",
            "stereo": "new-stereo",
            "depth0": "new-depth0",
            "depth1": "new-depth1",
            "depth2": "new-depth2",
            "depth3": "new-depth3",
        })

    def test_late_old_packet_cannot_poison_following_new_stamps(self):
        sync = EpisodeBundleSynchronizer({"a", "b"})
        sync.add(40_000_000_000, "a", "episode-one")
        sync.add(100_000_000, "a", "episode-two")
        # A finite old-session straggler can move the watermark forward, but
        # the next new-session frame triggers another clean generation.
        sync.add(40_100_000_000, "b", "old-straggler")
        generation, _, rewound = sync.add(200_000_000, "a", "new-a")
        self.assertTrue(rewound)
        sync.add(200_000_000, "b", "new-b")
        self.assertEqual(generation, 2)
        self.assertNotIn(40_100_000_000, sync.buckets)

    def test_small_out_of_order_delay_is_not_an_episode_reset(self):
        sync = EpisodeBundleSynchronizer(
            {"a", "b"}, rewind_tolerance_ns=500_000_000)
        sync.add(10_000_000_000, "a", "newer")
        generation, _, rewound = sync.add(
            9_700_000_000, "b", "slightly-older")
        self.assertFalse(rewound)
        self.assertEqual(generation, 0)


if __name__ == "__main__":
    unittest.main()
