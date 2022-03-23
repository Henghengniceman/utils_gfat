""" Command line tool to convert Licel binary files to EARLINET telecover files.
"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
import yaml

from lidar_processing import pre_processing

from ..licel import LicelFile
from ..__init__ import __version__

logger = logging.getLogger(__name__)


class TelecoverDataset(object):
    def __init__(self, filenames, settings_path, licel_timezone='UTC'):
        """ Create a telecover dataset object.

        Parameters
        ----------
        filenames : list of str
           List of candidate files to include in the telecover.
        settings_path : str
           Path to the YAML settings file.
        licel_timezone : str
           The timezone of dates found in the Licel files. Should match the available
           timezones in the TZ database.
        """
        self.settings_path = settings_path
        self.filenames = filenames
        self.licel_timezone = licel_timezone

        self.settings = self.read_settings(settings_path)
        self.file_sites = {}
        self.read_files()

        self.check_files_ok()

    @staticmethod
    def read_settings(settings_path):
        """ Read the configuration file.

        The file should be in YAML syntax."""

        if not os.path.isfile(settings_path):
            raise IOError("Wrong path for settings file (%s)" % settings_path)

        with open(settings_path) as yaml_file:
            try:
                settings = yaml.safe_load(yaml_file)
                logger.debug("Read settings file(%s)" % settings_path)
            except:
                raise IOError("Could not parse YAML file (%s)" % settings_path)

        return settings

    def read_files(self):
        for filename in self.filenames:
            # Load only header
            current_file = LicelFile(filename, use_id_as_name=True, licel_timezone=self.licel_timezone, import_now=False)
            current_file.import_header_only()

            # Populate the dictionary linking sites with files.
            site = current_file.site

            if site in self.settings['sectors'].values():
                current_file.import_file()  # Import full data
                if current_file.site in self.file_sites.keys():
                    self.file_sites[site].append(current_file)
                else:
                    self.file_sites[site] = [current_file, ]

    def check_files_ok(self):
        """ Check if the available files are enough to create the ASCII files. """
        # Check that all sectors are available.
        sectors = self.settings['sectors']
        for sector_name, site_name in sectors.items():
            if site_name not in self.file_sites.keys():
                raise ValueError('Site name {0} corresponding to sector {1} not found in any file.'.format(site_name,
                                                                                                           sector_name))

            len_sector = len(self.file_sites[site_name])
            if len_sector > 1:
                logger.info('More than one files ({0}) found in sector {1} (site name {2})'.format(len_sector,
                                                                                                   sector_name,
                                                                                                   site_name))

        # Check that all files have the correct channels
        channels = self.settings['channels'].keys()
        for site_name in sectors.values():
            for current_file in self.file_sites[site_name]:
                for channel in channels:
                    if channel not in current_file.channels.keys():
                        raise ValueError(
                            'File {0} does not contain the required channel {1}.'.format(current_file.file_name,
                                                                                         channel))

    def save_ascii(self, output_dir='.'):
        """ Save the files in the appropriate format.

        Parameters
        ---------
        output_dir : str
           The output directory.

        Returns
        -------
        file_paths : list of str
           A list of output paths for the saved files.
        """

        date, z = self._get_file_metadata()

        file_paths = []

        logger.info('Creating {0} files.'.format(len(self.settings['channels'])))

        for channel_id, channel_settings in list(self.settings['channels'].items()):

            output_filename = '{call_sign}_tc_{date}_{name}.dat'.format(call_sign=self.settings['call_sign'],
                                                                        date=date.strftime('%Y%m%dT%H'),
                                                                        name=channel_settings['short_name'])

            output_path = os.path.join(output_dir, output_filename)

            logger.info("Output file path: {0}".format(output_path))

            header_txt = ("{s[call_sign]} ({s[site]})\r\n"
                          "{s[system_name]}\r\n"
                          "{name}\r\n"
                          "{date}\r\n"
                          "range, {sectors}").format(s=self.settings,
                                                     name=channel_settings['name'],
                                                     date=date.strftime('%d.%m.%Y, %H'),
                                                     sectors=', '.join(self.settings['sector_order']))

            # Get altitude data
            sectors = self.settings['sector_order']

            bin_max = self.settings['bins_to_keep']

            data = [z, ]
            for sector in sectors:
                site_name = self.settings['sectors'][sector]
                files = self.file_sites[site_name]
                channel_data = [self.pre_process_channel(f.channels[channel_id]) for f in files]
                data.append(np.mean(channel_data, axis=0))

            data = np.array(data).T[:bin_max, :]

            np.savetxt(output_path, data, fmt='%.5e', delimiter=',', newline='\r\n', header=header_txt, comments='')

            logger.info("File saved.")

            file_paths.append(output_path)

        return file_paths

    def _get_file_metadata(self):
        """ Get date string and altitude array that describes the measurement.

        Currently, use the date of the first file in the North direction.
        """
        first_sector = self.settings['sector_order'][0]  # Could be N, or NO, or ...
        first_site = self.settings['sectors'][first_sector]
        first_file = self.file_sites[first_site][0]

        channel_id = list(self.settings['channels'].keys())[0]
        channel = first_file.channels[channel_id]

        return first_file.start_time, channel.z

    def pre_process_channel(self, channel):
        """ Pre-process channel according to the settings.

        The role of this method is to choose if the channel is photon or analog and call the approparite
        method.
        """
        settings = self.settings['channels'][channel.channel_name]

        trigger_delay = settings.get('trigger_delay', 0)
        background_min = settings['background_min']
        background_max = settings['background_max']

        if channel.is_analog:
            data_rc = self.pre_process_analog(channel, trigger_delay, background_min, background_max)
        else:
            dead_time = settings['dead_time']
            data_rc = self.pre_process_photon(channel, dead_time, trigger_delay, background_min, background_max)

        return data_rc

    @staticmethod
    def pre_process_photon(channel, dead_time, trigger_delay, background_min, background_max):
        """ Pre-process photon counting signals"""
        interval_ns = channel.bin_width * 1e9 * channel.number_of_shots  # Interval in ns, assuming constant laser shots

        data_dt = pre_processing.correct_dead_time_nonparalyzable(channel.data, interval_ns, dead_time)
        data_bs, background_mean, background_std = pre_processing.subtract_background(data_dt, background_min,
                                                                                      background_max)
        data_tc = pre_processing.correct_trigger_delay_bins(data_bs, trigger_delay)
        data_rc = pre_processing.apply_range_correction(data_tc, channel.z)

        return data_rc

    @staticmethod
    def pre_process_analog(channel, trigger_delay, background_min, background_max):

        data_bs, background_mean, background_std = pre_processing.subtract_background(channel.data,
                                                                                      background_min,
                                                                                      background_max)
        data_tc = pre_processing.correct_trigger_delay_bins(data_bs, trigger_delay)
        data_rc = pre_processing.apply_range_correction(data_tc, channel.z)

        return data_rc


def convert_to_telecover(file_paths, settings_path, licel_timezone, output_dir):
    """ Convert files to SCC. """

    try:
        dataset = TelecoverDataset(file_paths, settings_path, licel_timezone)
    except Exception as err:
        logger.error(err)
        sys.exit(1)

    try:
        dataset.save_ascii(output_dir)
    except Exception as err:
        logger.error(err)
        sys.exit(2)


def main():
    # Define the command line argument
    parser = argparse.ArgumentParser(
        description="A program to convert Licel binary files to EARLIENT telecover ASCII format")
    parser.add_argument("settings_file", help="The path to a parameter YAML.")
    parser.add_argument("files",
                        help="Location of licel files. Use relative path and filename wildcards. (default './*.*')",
                        default="./*.*")
    parser.add_argument('-o', '--output', help='Output directory.', default='.')

    parser.add_argument('--licel_timezone', help="String describing the timezone according to the tz database.",
                        default="UTC", dest="licel_timezone",
                        )
    # Verbosity settings from http://stackoverflow.com/a/20663028
    parser.add_argument('-d', '--debug', help="Print dubuging information.", action="store_const",
                        dest="loglevel", const=logging.DEBUG, default=logging.INFO,
                        )
    parser.add_argument('-s', '--silent', help="Show only warning and error messages.", action="store_const",
                        dest="loglevel", const=logging.WARNING
                        )
    parser.add_argument('--version', help="Show current version.", action='store_true')

    args = parser.parse_args()

    # Get the logger with the appropriate level
    logging.basicConfig(format='%(levelname)s: %(message)s', level=args.loglevel)
    logger = logging.getLogger(__name__)

    # Check for version
    if args.version:
        print("Version: %s" % __version__)
        sys.exit(0)

    # Get a list of files to process
    logger.debug("Search path: {0}".format(os.path.expanduser(args.files)))
    files = glob.glob(os.path.expanduser(args.files))

    # If not files found, exit
    if len(files) == 0:
        logger.error("No files found when searching for %s." % args.files)
        sys.exit(1)

    # If everything OK, proceed
    logger.info("Found {0} files matching {1}".format(len(files), args.files))

    convert_to_telecover(files, args.settings_file, args.licel_timezone, args.output)


if __name__ == "__main__":
    main()
