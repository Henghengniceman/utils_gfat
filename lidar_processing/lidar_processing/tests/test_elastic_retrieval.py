"""
Testing the Klett retrieval functions.
"""
import os
import unittest2 as unittest
import numpy as np

from lidar_processing.elastic_retrievals import klett_backscatter_aerosol

# Get the data path
current_path = os.path.dirname(__file__)
data_base_path = os.path.join(current_path, '../../data/klett_retrieval_evaluation/')


class TestKlettBackscatterAerosol(unittest.TestCase):
    def test_specific_profile_355(self):
        #   Specify the data files.
        filename_input = os.path.join(data_base_path, "rc355_20160406_0029_0229.txt")
        filename_reference = os.path.join(data_base_path, "b355_20160406_0029_0229.txt")

        #   Load the data from the files.
        data_input = np.loadtxt(filename_input, delimiter='\t', skiprows=1)
        data_reference = np.loadtxt(filename_reference, delimiter='\t', skiprows=1)

        #   Assign variables to data.
        rc_signal = data_input[:, 1]
        backscatter_reference = data_reference[:, 1]
        molecular_reference = data_reference[:, 2]

        #   Run the test.
        backscatter_output = klett_backscatter_aerosol(range_corrected_signal=rc_signal,
                                                       lidar_ratio_aerosol=30,
                                                       beta_molecular=molecular_reference,
                                                       index_reference=1338,
                                                       reference_range=134,
                                                       beta_aerosol_reference=5E-8,
                                                       bin_length=7.47146)

        np.testing.assert_allclose(backscatter_output, backscatter_reference, rtol=5e-2)

    def test_specific_profile_532(self):
        #   Specify the data files.
        filename_input = os.path.join(data_base_path, "rc532_20160406_0029_0229.txt")
        filename_reference = os.path.join(data_base_path, "b532_20160406_0029_0229.txt")

        #   Load the data from the files.
        data_input = np.loadtxt(filename_input, delimiter='\t', skiprows=1)
        data_reference = np.loadtxt(filename_reference, delimiter='\t', skiprows=1)

        #   Assign variables to data.
        rc_signal = data_input[:, 1]
        backscatter_reference = data_reference[:, 1]
        molecular_reference = data_reference[:, 2]

        #   Run the test.
        backscatter_output = klett_backscatter_aerosol(range_corrected_signal=rc_signal,
                                                       lidar_ratio_aerosol=50,
                                                       beta_molecular=molecular_reference,
                                                       index_reference=1338,
                                                       reference_range=134,
                                                       beta_aerosol_reference=3E-8,
                                                       bin_length=7.47146)

#        from matplotlib import pyplot as plt
#        plt.figure()
#        plt.plot(backscatter_reference, lw=2)
#        plt.plot(backscatter_output, 'r')
#        plt.show()

        np.testing.assert_allclose(backscatter_output, backscatter_reference, rtol=5e-2)
        
    def test_specific_profile_1064(self):
        #   Specify the data files.
        filename_input = os.path.join(data_base_path, "rc1064_20160406_0029_0229.txt")
        filename_reference = os.path.join(data_base_path, "b1064_20160406_0029_0229.txt")

        #   Load the data from the files.
        data_input = np.loadtxt(filename_input, delimiter='\t', skiprows=1)
        data_reference = np.loadtxt(filename_reference, delimiter='\t', skiprows=1)

        #   Assign variables to data.
        rc_signal = data_input[:, 1]
        backscatter_reference = data_reference[:, 1]
        molecular_reference = data_reference[:, 2]

        #   Run the test.
        backscatter_output = klett_backscatter_aerosol(range_corrected_signal=rc_signal,
                                                       lidar_ratio_aerosol=70,
                                                       beta_molecular=molecular_reference,
                                                       index_reference=1338,
                                                       reference_range=134,
                                                       beta_aerosol_reference=1E-8,
                                                       bin_length=7.47146)

#        from matplotlib import pyplot as plt
#        plt.figure()
#        plt.plot(backscatter_reference, lw=2)
#        plt.plot(backscatter_output, 'r')
#        plt.show()

        np.testing.assert_allclose(backscatter_output, backscatter_reference, rtol=5e-2)

if __name__ == "__main__":
    unittest.main()