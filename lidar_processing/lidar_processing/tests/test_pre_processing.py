import unittest2 as unittest

from lidar_processing.pre_processing import *


class TestRangeCorrection(unittest.TestCase):

    def setUp(self):
        self.original_signal = np.arange(10)
        self.z = np.linspace(3, 100, 10)
        self.measured_signal = self.original_signal / self.z ** 2

    def test_correction_on_artificial_signal(self):
        corrected_signal = apply_range_correction(self.measured_signal, self.z)
        np.testing.assert_almost_equal(corrected_signal, self.original_signal)


class TestNonParalyzableDeadTime(unittest.TestCase):

    def setUp(self):
        self.measurement_interval = 50  # in ns
        self.signal = np.linspace(0, 10)

    def test_no_change_for_zero_deadtime(self):
        corrected_signal = correct_dead_time_nonparalyzable(self.signal, self.measurement_interval, dead_time=0)
        np.testing.assert_almost_equal(corrected_signal, self.signal)

    def test_dead_time_half_of_measurement(self):
        """
        Dead time should be double the input signal.
        """
        dead_time = self.measurement_interval / 2.
        signal = np.array([1])
        corrected_signal = correct_dead_time_nonparalyzable(signal, self.measurement_interval, dead_time=dead_time)
        self.assertEqual(corrected_signal[0], 2)


class TestParalyzableDeadTime(unittest.TestCase):

    def setUp(self):
        self.measurement_interval = 50  # in ns
        self.signal = np.linspace(0, 9, 10)

    def test_no_change_for_zero_deadtime(self):
        corrected_signal = correct_dead_time_paralyzable(self.signal, self.measurement_interval, dead_time=0)
        np.testing.assert_almost_equal(corrected_signal, self.signal)
    
    def test_dead_time_linear_signal(self):
        dead_time = 5  # in ns
        signal_with_dead_time = self.signal * np.exp(-self.signal * dead_time / self.measurement_interval)
        corrected_signal = correct_dead_time_paralyzable(signal_with_dead_time, self.measurement_interval, dead_time)
        np.testing.assert_almost_equal(corrected_signal, self.signal, decimal=3)


class TestSubtractBackground(unittest.TestCase):

    def setUp(self):
        self.ones_signal = np.ones(100)

    def test_constant_signal_gives_zero(self):
        corrected_signal, _, _ = subtract_background(self.ones_signal, 80, 100)
        self.assertTrue(np.all(corrected_signal==0))

    def test_ones_signal_has_one_background(self):
        _, background_mean, _ = subtract_background(self.ones_signal, 80, 100)
        self.assertEqual(background_mean, 1)

    def test_constant_signal_has_zero_deviation(self):
        _, _, background_std = subtract_background(self.ones_signal, 80, 100)
        self.assertEqual(background_std, 0)


class TestSubtractElectronicBackground(unittest.TestCase):

    def test_equal_signal_and_background_gives_zero(self):
        signal = np.random.random(100)
        corrected_signal = subtract_electronic_background(signal, signal)
        self.assertTrue(np.all(corrected_signal==0))


class TestCorrectOverlap(unittest.TestCase):

    def test_no_change_for_overlap_ones(self):
        signal = np.random.random(100)
        overlap = np.ones(100)
        corrected_signal = correct_overlap(signal, overlap, full_overlap_idx=None)
        self.assertTrue(np.all(corrected_signal==signal))

    def test_signal_doubles_for_overlap_zero(self):
        signal = np.ones(1)
        overlap = np.ones(1) * 0.5
        corrected_signal = correct_overlap(signal, overlap, full_overlap_idx=None)
        self.assertEqual(corrected_signal[0], 2)

    def test_overlap_ignored_above_full_overlap(self):
        signal = np.ones(3)
        overlap = np.ones(3) * 0.5
        corrected_signal = correct_overlap(signal, overlap, full_overlap_idx=1)
        self.assertEqual(signal[2], corrected_signal[2])


class TestCorrectTriggerDelay(unittest.TestCase):

    def test_no_change_for_zero_delay(self):
        signal = np.random.random(100)
        altitude = np.arange(100)
        corrected_signal = correct_trigger_delay_ns(signal, altitude, trigger_delay_ns = 0)
        self.assertTrue(np.all(corrected_signal==signal))
    
    def test_bin_shift(self):
        altitude = np.arange(100)
        trigger_delay_bins = 3
        corrected_altitude = correct_trigger_delay_bins(altitude, trigger_delay_bins)
        self.assertTrue(np.all(np.diff(altitude[trigger_delay_bins:])==np.diff(corrected_altitude[trigger_delay_bins:])))

    def test_photons_conserved(self):
        signal = np.random.random(100)
        signal[:25] = np.zeros(25)
        signal[75:] = np.zeros(25)
        altitude = np.arange(100)
        corrected_signal = correct_trigger_delay_ns(signal, altitude, trigger_delay_ns = 2)
        np.testing.assert_almost_equal(np.sum(corrected_signal), np.sum(signal), decimal = 5)

if __name__ == "__main__":
    unittest.main()

