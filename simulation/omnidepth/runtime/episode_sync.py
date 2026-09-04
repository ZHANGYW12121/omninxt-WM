#!/usr/bin/env python3
"""Episode-aware timestamp synchronization helpers.

Isaac starts simulation time near zero for every fresh process.  Long-lived
perception sidecars therefore see a large timestamp rewind between benchmark
trials.  This helper treats that rewind as an explicit generation boundary so
old partial bundles cannot prevent the new episode from synchronizing.
"""

from collections import defaultdict


class EpisodeBundleSynchronizer:
    """Collect exact-stamp bundles and reset them after a clock rewind."""

    def __init__(self, required, retained_stamps=12,
                 rewind_tolerance_ns=500_000_000):
        self.required = frozenset(required)
        self.retained_stamps = max(1, int(retained_stamps))
        self.rewind_tolerance_ns = max(0, int(rewind_tolerance_ns))
        self.buckets = defaultdict(dict)
        self.latest_stamp_ns = None
        self.generation = 0
        self.rewind_count = 0
        self.external_generation = None
        self.external_reset_count = 0
        self.stale_external_count = 0

    def _observe_external_generation(self, external_generation):
        if external_generation is None:
            return False, False
        value = int(external_generation)
        if value < 0:
            raise ValueError("external generation must be non-negative")
        if self.external_generation is None:
            self.external_generation = value
            return False, False
        if value < self.external_generation:
            self.stale_external_count += 1
            return False, True
        if value == self.external_generation:
            return False, False
        self.buckets.clear()
        self.latest_stamp_ns = None
        self.external_generation = value
        self.generation += 1
        self.external_reset_count += 1
        return True, False

    def _observe_stamp(self, stamp_ns):
        stamp_ns = int(stamp_ns)
        rewound = (
            self.latest_stamp_ns is not None and
            stamp_ns + self.rewind_tolerance_ns < self.latest_stamp_ns
        )
        if rewound:
            self.buckets.clear()
            self.generation += 1
            self.rewind_count += 1
            self.latest_stamp_ns = stamp_ns
        elif self.latest_stamp_ns is None or stamp_ns > self.latest_stamp_ns:
            self.latest_stamp_ns = stamp_ns
        return rewound

    def observe(self, stamp_ns, external_generation=None):
        """Observe a timestamp that is not part of an exact bundle."""
        changed, stale = self._observe_external_generation(external_generation)
        if stale:
            return False
        return self._observe_stamp(stamp_ns) or changed

    def add(self, stamp_ns, kind, payload, external_generation=None):
        """Add one bundle member.

        Returns ``(generation, complete_bundle, rewound)``.  The bundle is
        ``None`` until all required kinds with the exact same stamp arrive.
        """
        stamp_ns = int(stamp_ns)
        changed, stale = self._observe_external_generation(external_generation)
        if stale:
            return self.generation, None, False
        rewound = self._observe_stamp(stamp_ns) or changed
        self.buckets[stamp_ns][kind] = payload
        complete = None
        if self.required.issubset(self.buckets[stamp_ns]):
            complete = self.buckets.pop(stamp_ns)
        for old_stamp in sorted(self.buckets)[:-self.retained_stamps]:
            self.buckets.pop(old_stamp, None)
        return self.generation, complete, rewound
