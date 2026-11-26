"""Small numerical and shape-contract checks; actual-file roundtrips run in CLI."""
import unittest
import numpy as np
from preprocess_emg import design_sos, filter_emg, make_windows, WINDOW


class PreprocessTests(unittest.TestCase):
    def test_channel_isolation_and_constant_rejection(self):
        t = np.arange(12000) / 2000
        values = np.full((12000, 8), 12.0)
        values[:, 3] += np.sin(2*np.pi*80*t)
        filtered = filter_emg(values, design_sos())
        self.assertLess(np.max(abs(filtered[2000:-2000, [0, 1, 2, 4, 5, 6, 7]])), 1e-10)
        self.assertGreater(np.std(filtered[2000:-2000, 3]), .5)

    def test_invalid_data(self):
        sos = design_sos()
        for values in [np.zeros((10000, 7)), np.zeros((2000, 8)), np.full((10000, 8), np.nan)]:
            with self.assertRaises(ValueError):
                filter_emg(values, sos)

    def test_window_layout_and_boundary_flags(self):
        values = np.arange(40344*8).reshape(40344, 8).astype(np.float32)
        caution = np.zeros(40344, bool)
        for boundary in [0, 20168, 40344]:
            caution[max(0,boundary-1000):min(len(caution),boundary+1000)] = True
        windows, starts, flags = make_windows(values, caution)
        self.assertEqual(windows.shape, (5, 8, WINDOW))
        np.testing.assert_array_equal(starts, [0, 8000, 16000, 24000, 32000])
        np.testing.assert_array_equal(flags, [True, False, True, False, True])
        self.assertEqual(windows[2, 4, 10], values[16010, 4])
        self.assertEqual(len(values) - windows.shape[0]*WINDOW, 344)
        with self.assertRaisesRegex(ValueError, "8000"):
            make_windows(values[:7999], caution[:7999])


if __name__ == "__main__":
    unittest.main()
