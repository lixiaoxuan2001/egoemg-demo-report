import unittest
import numpy as np
from infer_left_emg import merge_windows, window_starts, video_indices


class TimelineTests(unittest.TestCase):
    def test_full_recording_with_final_overlap(self):
        starts = window_starts(40344, 8000, 510, 0)
        np.testing.assert_array_equal(starts, [0, 7490, 14980, 22470, 29960, 32344])
        # Every window predicts the absolute source index; merging must preserve it.
        a = np.stack([np.tile(np.arange(s+510,s+8000),(22,1)) for s in starts])
        out, count = merge_windows(a, starts, 40344, 510, 0, 8000)
        self.assertTrue(np.isnan(out[:510]).all())
        np.testing.assert_array_equal(out[510:, 0], np.arange(510,40344))
        self.assertTrue((count[510:]>0).all())
        self.assertEqual(int(count.max()), 2)

    def test_overlap_average_and_nonfinite_rejection(self):
        a = np.stack([np.ones((22,6)), np.full((22,6),3.)])
        out, count = merge_windows(a,[0,3],11,2,0,8)
        np.testing.assert_array_equal(out[5:8,0],[2,2,2])
        np.testing.assert_array_equal(count[5:8],[2,2,2])
        a[0,0,0]=np.inf
        with self.assertRaises(ValueError): merge_windows(a,[0,3],11,2,0,8)

    def test_exact_window_short_recording_and_right_context(self):
        np.testing.assert_array_equal(window_starts(8000,8000,510,0),[0])
        with self.assertRaises(ValueError): window_starts(7999,8000,510,0)
        starts=window_starts(20,10,2,1)
        a=np.zeros((len(starts),22,7))
        out,count=merge_windows(a,starts,20,2,1,10)
        self.assertTrue((count[2:19]>0).all())
        self.assertTrue(np.isnan(out[[0,1,19]]).all())

    def test_video_rounding_and_missing_initial_context(self):
        t,i=video_indices(40344,2000,30)
        self.assertEqual(len(t),606)
        self.assertLess(len(t)/30-40344/2000,1/30)
        self.assertTrue((i<40344).all())
        self.assertEqual(int((i<510).sum()),8)
        self.assertEqual(i[-1],40333)


if __name__=='__main__': unittest.main()
