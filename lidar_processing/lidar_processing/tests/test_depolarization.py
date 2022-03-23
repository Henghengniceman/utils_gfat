"""
Testing the code of the depolarization functions.
"""

import unittest2 as unittest
from lidar_processing.depolarization import *

class TestCalibrationConstantCrossTotalProfile(unittest.TestCase):
    
    def test_specific_values(self):
        c_profile = calibration_constant_cross_total_profile(signal_cross_plus45 = 0.36,
                                                             signal_cross_minus45 = 0.9,
                                                             signal_total_plus45 = 2,
                                                             signal_total_minus45 = 1.8,
                                                             r_cross = 189,
                                                             r_total = 0.9)
        self.assertEqual(c_profile, 0.003)
        
    def test_boundary_conditions_zero1(self):
        c_profile = calibration_constant_cross_total_profile(signal_cross_plus45 = 0,
                                                             signal_cross_minus45 = 0.9,
                                                             signal_total_plus45 = 2,
                                                             signal_total_minus45 = 1.8,
                                                             r_cross = 189,
                                                             r_total = 0.9)
        self.assertEqual(c_profile, 0)
        
    def test_boundary_conditions_zero2(self):
        c_profile = calibration_constant_cross_total_profile(signal_cross_plus45 = 0.36,
                                                             signal_cross_minus45 = 0,
                                                             signal_total_plus45 = 2,
                                                             signal_total_minus45 = 1.8,
                                                             r_cross = 189,
                                                             r_total = 0.9)
        self.assertEqual(c_profile, 0)
        
    def test_boundary_conditions_unit(self):
        c_profile = calibration_constant_cross_total_profile(signal_cross_plus45 = 1,
                                                             signal_cross_minus45 = 1,
                                                             signal_total_plus45 = 1,
                                                             signal_total_minus45 = 1,
                                                             r_cross = 1,
                                                             r_total = 1)
        self.assertEqual(c_profile, 1)
    
                                                      
class TestCalibrationConstantCrossParallelProfile(unittest.TestCase):
    
    def test_specific_values(self):
        v_star_profile = calibration_constant_cross_parallel_profile(signal_cross_plus45 = 0.4,
                                                                     signal_cross_minus45 = 0.96,
                                                                     signal_parallel_plus45 = 2,
                                                                     signal_parallel_minus45 = 1.2,
                                                                     t_cross = 0.02,
                                                                     t_parallel = 0.7,
                                                                     r_cross = 0.8,
                                                                     r_parallel = 0.16)
        self.assertEqual(v_star_profile, 0.3)
            
    def test_boundary_conditions_zero1(self):
        v_star_profile = calibration_constant_cross_parallel_profile(signal_cross_plus45 = 0,
                                                                     signal_cross_minus45 = 0.96,
                                                                     signal_parallel_plus45 = 2,
                                                                     signal_parallel_minus45 = 1.2,
                                                                     t_cross = 0.02,
                                                                     t_parallel = 0.7,
                                                                     r_cross = 0.8,
                                                                     r_parallel = 0.16)
        self.assertEqual(v_star_profile, 0)
        
    def test_boundary_conditions_zero2(self):
        v_star_profile = calibration_constant_cross_parallel_profile(signal_cross_plus45 = 0.4,
                                                                     signal_cross_minus45 = 0,
                                                                     signal_parallel_plus45 = 2,
                                                                     signal_parallel_minus45 = 1.2,
                                                                     t_cross = 0.02,
                                                                     t_parallel = 0.7,
                                                                     r_cross = 0.8,
                                                                     r_parallel = 0.16)
        self.assertEqual(v_star_profile, 0)        


class TestCalibrationConstantValue(unittest.TestCase):
    def test_specific_values(self):
        calibration_constant_profile = np.linspace(1,2,11)
        c_mean, c_sem = calibration_constant_value(calibration_constant_profile,
                                                   first_bin = 2,
                                                   bin_length = 7.5,
                                                   lower_limit = 20,
                                                   upper_limit = 30)
        self.assertAlmostEqual(c_mean, 1.5)
        self.assertAlmostEqual(c_sem, 0.057735026)
        
    def test_boundary_conditions_unit(self):
        calibration_constant_profile = np.linspace(1,1,11)
        c_mean, c_sem = calibration_constant_value(calibration_constant_profile,
                                                   first_bin = 2,
                                                   bin_length = 7.5,
                                                   lower_limit = 20,
                                                   upper_limit = 30)
        self.assertEqual(c_mean, 1)
        self.assertEqual(c_sem, 0.)
     
     
class TestVolumeDepolarizationCrossTotal(unittest.TestCase):
    def test_specific_values(self):
        delta_v_uncommon = volume_depolarization_cross_total(signal_cross = 0.44,
                                                       signal_total = 2,
                                                       r_cross = 109.9,
                                                       r_total = 0.9,
                                                       c = 0.02)
        self.assertEqual(delta_v_uncommon, 0.1)

    def test_boundary_conditions_zero(self):
        delta_v_uncommon = volume_depolarization_cross_total(signal_cross = 0,
                                                       signal_total = 2,
                                                       r_cross = 109.9,
                                                       r_total = 0.9,
                                                       c = 0.02)
        self.assertAlmostEqual(delta_v_uncommon, -0.009099181)
        
    def test_boundary_conditions_two(self):
        delta_v_uncommon = volume_depolarization_cross_total(signal_cross = 2,
                                                       signal_total = 2,
                                                       r_cross = 2,
                                                       r_total = 2,
                                                       c = 2)
        self.assertEqual(delta_v_uncommon, -0.5)
        
        
class TestVolumeDepolarizationCrossParallel(unittest.TestCase):
    def test_specific_values(self):
        delta_v_common = volume_depolarization_cross_parallel(signal_cross = 0.44,
                                                          signal_parallel = 2,
                                                          t_cross = 0.03,
                                                          t_parallel = 0.91,
                                                          r_cross = 0.83,
                                                          r_parallel = 0.01,
                                                          v_star = 0.02)
        self.assertAlmostEqual(delta_v_common, 20)

    def test_boundary_conditions_zero(self):
        delta_v_common = volume_depolarization_cross_parallel(signal_cross = 0,
                                                          signal_parallel = 2,
                                                          t_cross = 0.03,
                                                          t_parallel = 0.91,
                                                          r_cross = 0.83,
                                                          r_parallel = 0.01,
                                                          v_star = 0.02)
        
        self.assertAlmostEqual(delta_v_common, -0.01204819)


class TestParticleDepolarization(unittest.TestCase):
    def test_boundary_conditions_zero(self):
        delta_p = particle_depolarization(delta_m = 0,
                                          delta_v = 0,
                                          molecular_backscatter = 1E-6,
                                          particle_backscatter = 9E-6)
        self.assertEqual(delta_p, 0)
        
    def test_boundary_conditions_unit(self):        
        delta_p = particle_depolarization(delta_m = 1,
                                          delta_v = 1,
                                          molecular_backscatter = 1E-6,
                                          particle_backscatter = 9E-6)                                  
        self.assertEqual(delta_p, 1)

        
if __name__ == "__main__":
    unittest.main()