import unittest2 as unittest

from lidar_processing.fit_checks import *


class TestCheckCorrelation(unittest.TestCase):

    def setUp(self):
        angle_values = np.arange(100) / 100. * 2 * np.pi
        self.first_signal = np.sin(angle_values)
        self.second_signal = - self.first_signal

    def test_correlation_equals_one_for_equal_signals(self):
        correlation = check_correlation(self.first_signal, self.first_signal, threshold=None)
        np.testing.assert_almost_equal(correlation, 1)

    def test_correlation_equals_minus_one_for_oposite_signals(self):
        correlation = check_correlation(self.first_signal, self.second_signal, threshold=None)
        np.testing.assert_almost_equal(correlation, -1)

    def test_threshold_true_for_equal_signals(self):
        correlation = check_correlation(self.first_signal, self.first_signal, threshold=0.9)
        np.testing.assert_equal(correlation, True)

    def test_threshold_false_for_oposite_signals(self):
        correlation = check_correlation(self.first_signal, self.second_signal, threshold=-0.9)
        np.testing.assert_equal(correlation, False)


class TestSlidingCheckCorrelation(unittest.TestCase):

    def setUp(self):
        angle_values = np.arange(100) / 100. * 2 * np.pi
        self.first_signal = np.sin(angle_values)
        self.second_signal = - self.first_signal

    def test_correlation_equals_one_for_equal_signals(self):
        correlation = sliding_check_correlation(self.first_signal, self.first_signal, window_length=11, threshold=None)
        np.testing.assert_allclose(correlation, 1)

    def test_correlation_equals_minus_one_for_oposite_signals(self):
        correlation = sliding_check_correlation(self.first_signal, self.second_signal, window_length=11, threshold=None)
        np.testing.assert_allclose(correlation, -1)

    def test_threshold_true_for_equal_signals(self):
        correlation = sliding_check_correlation(self.first_signal, self.first_signal, window_length=11, threshold=0.9)
        np.testing.assert_allclose(correlation, True)

    def test_threshold_false_for_oposite_signals(self):
        correlation = sliding_check_correlation(self.first_signal, self.second_signal, window_length=11, threshold=-0.9)
        np.testing.assert_allclose(correlation, False)


if __name__ == "__main__":
    unittest.main()

