import collections
import os
import tempfile
import unittest
from unittest import mock

import map_viewer_server
from map_viewer_server import AgentFeed, accumulate_resetting_counter


class ViewerCounterTests(unittest.TestCase):
    def test_counter_increments_without_double_counting_snapshots(self):
        total, previous = accumulate_resetting_counter(0, None, 4)
        self.assertEqual((total, previous), (4, 4))
        total, previous = accumulate_resetting_counter(total, previous, 4)
        self.assertEqual((total, previous), (4, 4))
        total, previous = accumulate_resetting_counter(total, previous, 7)
        self.assertEqual((total, previous), (7, 7))

    def test_counter_remains_monotonic_after_worker_reset(self):
        total, previous = accumulate_resetting_counter(28, 28, 0)
        self.assertEqual((total, previous), (28, 0))
        total, previous = accumulate_resetting_counter(total, previous, 3)
        self.assertEqual((total, previous), (31, 3))

    def test_history_recovers_when_sb3_rewrites_header_in_place(self):
        feed = object.__new__(AgentFeed)
        feed.history = collections.deque(maxlen=map_viewer_server.HISTORY_MAX_ROWS)
        feed.history_inode = None
        feed.history_offset = 0
        feed.history_header = None
        feed.history_fragment = ""
        feed.history_signature = None

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "progress.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("time/total_timesteps,time/fps\n172032,3994\n")
            with mock.patch.object(map_viewer_server, "PROGRESS_CSV_PATH", csv_path):
                self.assertTrue(feed._read_history())
                self.assertEqual(len(feed.history), 1)

                # SB3 truncates/rebuilds the same inode when new train/*
                # fields appear, and the rewritten file can be larger than
                # the old offset. The old append-only reader missed this.
                with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                    handle.write(
                        "time/total_timesteps,time/fps,train/explained_variance,"
                        "rollout/ep_rew_mean\n"
                        "172032,3994,,\n"
                        "344064,3261,0.72,4100\n"
                    )
                self.assertTrue(feed._read_history())

            self.assertEqual(len(feed.history), 2)
            self.assertEqual(feed.history[-1]["step"], 344064)
            self.assertEqual(feed.history[-1]["train/explained_variance"], 0.72)
            self.assertEqual(feed.history[-1]["rollout/ep_rew_mean"], 4100.0)


if __name__ == "__main__":
    unittest.main()
