import os
import sys
import platform
import shutil
from distutils.dir_util import mkpath
import numpy as np
import pandas as pd
import datetime as dt
from scipy import stats
import json
from multiprocessing import Pool
import pdb

MODULE_DIR = os.path.dirname(sys.modules[__name__].__file__)
sys.path.insert(0, MODULE_DIR)
import utils
from lidar_layer_detection import layer_detection
from lidar_preprocessing import apply_bin_zero_correction
from lidar_linear_response_region import estimate_linear_response_region
from utils_gfat import config

""" DEFAULT AUXILIAR INFO
"""
# Root Directory (in NASGFAT)  according to operative system
DATA_DN = config.DATA_DN


""" BIN ZERO 
"""
def build_bin_zero_filename(lidar_name, data_dn=None):
    """
    Build Full path for default bin zero filename

    Parameters
    ----------
    lidar_name: str
        lidar name identifier
    data_dn: str
        data directory

    Returns
    -------
    bin_zero_fn: str
        full path of bin zero fn

    """
    if data_dn is None:
        data_dn = DATA_DN
    if not os.path.isdir(data_dn):
        print("WARNING. Data directory %s not found" % data_dn)
        print("bin zero info will not be reached")
        return False
    if np.logical_or(lidar_name == 'mhc', lidar_name == "MULHACEN"):   # MULHACEN
        lidar_name = "MULHACEN"
    elif np.logical_or(lidar_name == 'vlt', lidar_name == "VELETA"):   # VELETA
        lidar_name = "VELETA"
    else:  # Unknown
        lidar_name = "LIDAR"

    bin_zero_fn = os.path.join(data_dn, lidar_name, "1a", "bin_zero.csv")
    return bin_zero_fn


def create_bin_zero_file(lidar_name, bin_zero_fn=None, data_dn=None,
                         force=False, empty=False):
    """
    Create Bin Zero File

    Parameters
    ----------
    lidar_name: str
        lidar system: mulhacen, veleta
    bin_zero_fn: str
        full path of bin zero file
    data_dn: str
        full path of directory of data where bin zero file will be created if bin_zero_fn is None
    force: bool
        Force to create file
    empty: bool
        Create File Empty or with default data

    Returns
    -------

    """

    if bin_zero_fn is None:
        bin_zero_fn = build_bin_zero_filename(lidar_name, data_dn=data_dn)

    if np.logical_or(force, not os.path.isfile(bin_zero_fn)):
        f = open(bin_zero_fn, 'w')
        # Header
        f.write("# HISTORICAL VALUES OF ZERO_BIN \n"
                "# polarization: none (0), parallel (1), perpendicular(2) \n"
                "# detection_mode: analog (0), photoncounting(1) \n")
        f.close()
        if not empty:
            if lidar_name == "MULHACEN":
                # Default Values [Bravo-Aranda PhD Thesis. Table 4.3]
                wvs = [355.0, 355.0, 387.0, 408.0, 532.0, 532.0, 532.0, 532.0,
                       607.0, 1064.0]
                pols = [0, 0, 0, 0, 1, 1, 2, 2, 0, 0]
                mods = [0, 1, 1, 1, 0, 1, 0, 1, 1, 1]
                bzs = [7, 8, 8, 8, 7, 9, 7, 9, 8, 5]
                # TODO: 1064 es analog (tesis) o photoc (netcdf)?
            elif lidar_name == "VELETA":
                # TODO
                wvs = [-999]
                pols = [-999]
                mods = [-999]
                bzs = [-999]
            else:
                print("ERROR. lidar_name not known")
                return
            date_df = [dt.datetime(1970, 1, 1).strftime("%Y%m%d")] * len(wvs)
            df = pd.DataFrame(
                {"date": date_df, "wavelength": wvs, "polarization": pols,
                 "detection_mode": mods, "bin_zero": bzs})
            # df.date = df.date.apply(lambda x: dt.datetime.strptime(x, "%Y%m%d"))
            df.to_csv(bin_zero_fn, mode='a', index=False, na_rep=np.nan, date_format='%Y%m%d')
            shutil.copyfile(bin_zero_fn, "%s.bkp" % bin_zero_fn)
            print("File %s has been created with default values for bin zero" % bin_zero_fn)
    else:
        print("File %s already exists" % bin_zero_fn)


def get_bin_zero(lidar_name, wv, pol, mod, ref_time=None, bin_zero_fn=None, data_dn=None):
    """
    Reads bin_zero.csv and gets bin zero for the channel defined by the three
    input values for the latest date

    Parameters
    ----------
    lidar_name: str
        lidar system: mulhacen, veleta
    wv: float
        wavelength
    pol: int
        polarization
    mod: int
        detection mode
    ref_time: str
       reference time for getting proper bin zero, YYYYMMDD
    bin_zero_fn: str
        bin zero file
    data_dn: str
        full path of directory of data where bin zero file should be

    Returns
    -------
    bz: int
        zero bin

    """
    # pdb.set_trace()
    print("INFO. Start Get Bin Zero")
    if bin_zero_fn is None:
        bin_zero_fn = build_bin_zero_filename(lidar_name, data_dn=data_dn)

    default = False
    try:
        if os.path.isfile(bin_zero_fn):
            # read bin zero file
            bz_ls = pd.read_csv(bin_zero_fn, comment='#', parse_dates=["date"])
            bz_ls = bz_ls.rename(columns=lambda x: x.strip())
            # bz_ls['date'] = bz_ls['date'].apply(lambda x: x.strftime('%Y%m%d'))

            # find bz for wv, pol, mod
            bz_ls = bz_ls.loc[np.logical_and.reduce((bz_ls.wavelength == wv,
                                                     bz_ls.polarization == pol,
                                                     bz_ls.detection_mode == mod))]
            if bz_ls.empty:
                default = True
                print("No bin zero info for wv=%s, pol=%s, mode=%s" % (wv, pol, mod))
            else:
                # ref time
                if ref_time is None:
                    ref_time = dt.datetime.strptime(dt.datetime.now().strftime("%Y%m%d"), "%Y%m%d")
                ref_time = pd.Timestamp(ref_time)
                # bz from closest date to ref_time
                dates_ls = bz_ls.date.sort_values().values
                i = 0
                ok = 0
                while ok == 0:
                    if i < len(bz_ls):
                        if (pd.Timestamp(dates_ls[i]) - ref_time) < dt.timedelta(0):
                            i += 1
                        else:
                            ok = 1
                    else:
                        ok = 1
                i = i - 1
                # nearest date with bz value
                bz = bz_ls.loc[bz_ls.date == dates_ls[i]].bin_zero.values

                # check if bz is not nan. otherwise, look for previous dates
                # recursively until default values.
                # Otherwise: bz = 0
                fin = False
                while not fin:
                    if np.isnan(bz):
                        i -= 1
                        if i >= 0:
                            bz = bz_ls.loc[bz_ls.date == dates_ls[i]].bin_zero.values
                        else:
                            default = True
                            fin = True
                            print("No value found for bin zero")
                    else:
                        fin = True
        else:
            default = True
            print("Error. File %s not found" % bin_zero_fn)
    except Exception as e:
        default = True
        print(str(e))
        print("WARNING. bin zero file not read")

    """ Default Bin Zero in case of fail """
    if default:
        bz = 0 # this is need to change
        print("Warning: Bin Zero set to default: %i" % bz)

    print("INFO. End Get Bin Zero")

    return int(bz)


def set_bin_zero(lidar_name, ref_time, wv, pol, mod, bz, bin_zero_fn=None, data_dn=None):
    """
    Write bin zero value in bin zero file

    Parameters
    ----------
    lidar_name: str
        lidar system: mulhacen, veleta
    ref_time: str, yyyymmdd
        date of reference for bin zero
    wv: float, array
        wavelengths
    pol: float, array
        polarization
    mod: float, array
        detection mode
    bz: float, array
        bin zero
    bin_zero_fn: str
        full path for bin zero file
    data_dn: str
        full path of directory of data where bin zero file should be

    Returns
    -------

    """

    try:
        # Prepare data into pandas dataframe
        data_dict = {'date': ref_time, 'wavelength': wv, 'polarization': pol,
                     'detection_mode': mod, 'bin_zero': bz}
        if wv.shape:  # if wv is array
            df = pd.DataFrame(data_dict)
        else:  # if wv is float
            df = pd.DataFrame(data_dict, index=[0])
        #df.date = df.date.apply(lambda x: dt.datetime.strptime(x.strftime("%Y%m%d"), "%Y%m%d"))

        """ write data in bin zero file """
        if bin_zero_fn is None:
            bin_zero_fn = build_bin_zero_filename(lidar_name, data_dn=data_dn)
        if not os.path.isfile(bin_zero_fn):  # create if it does not exist
            create_bin_zero_file(lidar_name, force=True, bin_zero_fn=bin_zero_fn)
            df.to_csv(bin_zero_fn, mode='a', index=False, header=False, na_rep=np.nan, date_format='%Y%m%d')
        else:  # add bz to file
            dateparse = lambda x: dt.datetime.strptime(x, '%Y%m%d')
            try:
                bz_ls = pd.read_csv(bin_zero_fn, comment='#', parse_dates=["date"], date_parser=dateparse)
            except Exception as e:
                bz_ls = pd.read_csv("%s.bkp" % bin_zero_fn, comment='#', parse_dates=["date"], date_parser=dateparse)
                print(str(e))
                print("WARNING: Problems reading %s. Bkp is read instead" % bin_zero_fn)
            bz_ls = bz_ls.rename(columns=lambda x: x.strip())
            bz_ls['date'] = bz_ls['date'].apply(lambda x: x.strftime('%Y%m%d'))
            # add/update the bz value for (date, wv, pol, mod)
            for _, row in df.iterrows():
                # check the bz exists:
                idx = np.logical_and.reduce((bz_ls.date == row.date,
                                             bz_ls.wavelength == row.wavelength,
                                             bz_ls.polarization == row.polarization,
                                             bz_ls.detection_mode == row.detection_mode))
                if idx.any():  # substitute
                    bz_ls.loc[idx, "bin_zero"] = row["bin_zero"]
                else:  # add
                    bz_ls = bz_ls.append(row, ignore_index=True)
            # sort by ascending date and wavelength value and update file
            bz_ls = bz_ls.sort_values(by=['date', 'wavelength'])
            create_bin_zero_file(lidar_name, bin_zero_fn=bin_zero_fn, force=True, empty=True)
            bz_ls.to_csv(bin_zero_fn, mode='a', index=False, na_rep=np.nan, date_format='%Y%m%d')
            shutil.copyfile(bin_zero_fn, "%s.bkp" % bin_zero_fn)
    except Exception as e:
        print(str(e))
        print("ERROR. bin zero not set in file %s" % bin_zero_fn)


def estimate_bin_shift_single_profile(signal_an, signal_pc, ranges, bins_arr=None,
                                      method=None, threshold=None, pc_min=None):
    """
    Estimation of AN-PC bin shift for a single profile for parallelizing purposes

    Parameters
    ----------
    signal_an: array
        analog signal (ranges)
    signal_pc: array
        photoncounting signal (ranges)
    ranges: array
        ranges
    bins_arr: array
        array of possible bin zero candidates
    method: str
        method for bin shift estimation.
        layer: layer detection by an and pc channels
        lrr: estimate linear response region
    threshold: float
        minimum r-value (Pearson) for considering valid linear correlation
    pc_min: float
       lower threshold for considering PC values (MHz)

    Returns
    -------
    bin_shift: float
        bin shift an-pc

    """

    """ Default values """
    if bins_arr is None:
        bins_arr = np.arange(-20, 21)
    if method is None:
        method = "layer"
    if threshold is None:
        threshold = 0.80
    if pc_min is None:
        pc_min = 15

    dim_bin = len(bins_arr)
    bin_range = ranges[1] - ranges[0]

    # Applying Method to find a range region to study the AN-PC delay
    if method == "lrr":  # linear response region
        h_min_lr, h_max_lr, r2_lr, r2_arr, hi_arr, he_arr = estimate_linear_response_region(
            ranges, signal_an, signal_pc, pc_min=pc_min)
        try:
            idx = np.logical_and(ranges >= h_min_lr, ranges <= h_max_lr)
            ref_range = ranges[idx]
        except Exception as e:
            idx = ranges*False
            ref_range = []
            print(str(e))
    else:  # layer
        layer_an, ref_range_an = layer_detection(ranges, signal_an)
        layer_pc, ref_range_pc = layer_detection(ranges, signal_pc)
        if np.logical_or(layer_an, layer_pc):
            if not layer_pc:
                ref_range = ref_range_an[0]
            elif not layer_an:
                ref_range = ref_range_pc[0]
            else:
                try:
                    pc_nearest_idx, _ = utils.find_nearest_1d(ref_range_pc, ref_range_an[0])
                    ref_range = np.nanmean([ref_range_an[0], ref_range_pc[pc_nearest_idx]])
                except Exception as e:
                    ref_range = ref_range_an[0]
                    print(str(e))
            # save signals around the ref_range
            idx = np.logical_and(ranges >= ref_range - bin_range*abs(bins_arr[0]),
                                 ranges <= ref_range + bin_range*abs(bins_arr[-1]))
            ref_range = ranges[idx]
            ref_signal_an = signal_an[idx]
            ref_signal_pc = signal_pc[idx]
        else:
            idx = ranges*False
            ref_range = np.asarray(np.nan)
            ref_signal_an = np.asarray(np.nan)
            ref_signal_pc = np.asarray(np.nan)

    # Find best bin shift
    rvalue_bs = np.zeros(dim_bin)*np.nan
    best_bin = np.nan
    if idx.any():
        # positive linear correlation between an and pc for a range of bin-shifts
        # correlation better than threshold
        for j in range(dim_bin):
            slope, intercept, rvalue_bs[j] = utils.linear_regression(
                ref_signal_an, apply_bin_zero_correction(ref_signal_pc, bins_arr[j]))
            if slope < 0:
                rvalue_bs[j] = np.nan

        # bin with best rvalue
        try:
            idx_notnan = ~np.isnan(rvalue_bs)
            rvalue_bs_aux = rvalue_bs[idx_notnan]
            bins_arr_aux = bins_arr[idx_notnan]
            if np.logical_and(len(rvalue_bs_aux) > 0, np.max(rvalue_bs_aux) >= threshold):  # max R2 above threshold
                if (np.diff(rvalue_bs_aux) < 0).all():  # r2_arr always decreasing
                    best_bin = np.nan
                elif (np.diff(rvalue_bs_aux) > 0).all():  # r2_arr always increasing
                    best_bin = np.nan
                else:
                    idx_max = np.argmax(rvalue_bs_aux)
                    if np.logical_or(idx_max > 0, idx_max < len(rvalue_bs_aux)-1):
                        left = np.diff(rvalue_bs_aux[:idx_max])
                        right = np.diff(rvalue_bs_aux[(idx_max + 1):])
                        if np.logical_or((left >= 0).all(), (right <= 0).all()):
                            best_bin = bins_arr_aux[np.nanargmax(rvalue_bs_aux)]
                        else:
                            if np.logical_and(len(left) > 1, len(right) > 1):
                                if np.logical_and((left[-2:] >= 0).all(),
                                                  (right[:2] <= 0).all()):
                                    best_bin = bins_arr_aux[np.nanargmax(rvalue_bs_aux)]
                                else:
                                    best_bin = np.nan
                            elif np.logical_and(len(left) == 1, len(right) > 1):
                                if (right[:2] <= 0).all():
                                    best_bin = bins_arr_aux[np.nanargmax(rvalue_bs_aux)]
                                else:
                                    best_bin = np.nan
                            elif np.logical_and(len(left) > 1, len(right) == 1):
                                if (left[-2:] <= 0).all():
                                    best_bin = bins_arr_aux[np.nanargmax(rvalue_bs_aux)]
                                else:
                                    best_bin = np.nan
                            else:
                                best_bin = np.nan
                    else:  # maximo en un extremo. no interesa
                        best_bin = np.nan
            else:
                best_bin = np.nan
        except:
            best_bin = np.nan

    return best_bin, rvalue_bs, ref_range, ref_signal_an, ref_signal_pc


def estimate_bin_shift(signal_an, signal_pc, ranges, method=None, threshold=None,
                       pc_min=None, parallel=True):
    """
    Estimation of AN-PC bin shift

    Parameters
    ----------
    signal_an: array
        analog signal (time, range)
    signal_pc: array
        photoncounting signal (time, range)
    ranges: array
        ranges
    method: str
        method for bin shift estimation.
        layer: layer detection by an and pc channels
        lrr: estimate linear response region
    threshold: float
        minimum r-value (Pearson) for considering valid linear correlation
    pc_min: float
       lower threshold for considering PC values (MHz)
    parallel: bool
        apply parallelization for single profiles

    Returns
    -------
    bin_shift: float
        bin shift an-pc
    result: dict
        result data

    """

    # check viability
    if not signal_an.shape == signal_pc.shape:
        print("ERROR: analog and pc must have same shape")
        return

    # Default values
    if method is None:
        method = "layer"
    if threshold is None:
        threshold = 0.85
    if pc_min is None:
        pc_min = 15
    if not isinstance(parallel, bool):
        parallel = True

    # Set of possible bin shifts
    bins_arr = np.arange(-20, 21)
    dim_bin = len(bins_arr)

    # force to 2D
    ndims = signal_an.ndim
    if ndims == 1:  # reshape to 2D (1, len(ranges))
        signal_an = signal_an[np.newaxis, :]
        signal_pc = signal_pc[np.newaxis, :]
        parallel = False
    dim_p, dim_h = signal_an.shape

    # Parallel or Sequential Calculation
    if dim_p < 20:
        parallel = False

    # Application of Method
    if parallel:
        with Pool(os.cpu_count()) as pool:
            an_pc_h = [(x, y, ranges, bins_arr, method, threshold, pc_min)
                       for x, y in zip(signal_an, signal_pc)]
            shift_pc = np.array(pool.starmap(estimate_bin_shift_single_profile, an_pc_h))
        bin_shift_profiles = [x[0] for x in shift_pc]
        r_bin_shift_profiles = [x[1].tolist() for x in shift_pc]
        ref_range_profiles = [x[2].tolist() for x in shift_pc]
        ref_signal_an_profiles = [x[3].tolist() for x in shift_pc]
        ref_signal_pc_profiles = [x[4].tolist() for x in shift_pc]
    else:
        bin_shift_profiles = np.zeros(dim_p)
        r_bin_shift_profiles = np.zeros((dim_p, dim_bin))
        ref_range_profiles = []
        ref_signal_an_profiles = []
        ref_signal_pc_profiles = []
        for t in range(dim_p):
            bin_shift_profiles[t], r_bin_shift_profiles[t, :], x, y, z = \
                estimate_bin_shift_single_profile(signal_an[t, :], signal_pc[t, :],
                                                  ranges, bins_arr, method=method)
            ref_range_profiles.append(x.tolist())
            ref_signal_an_profiles.append(y.tolist())
            ref_signal_pc_profiles.append(z.tolist())
        bin_shift_profiles = bin_shift_profiles.tolist()
        r_bin_shift_profiles = r_bin_shift_profiles.tolist()

    # Final Bin Shift: Mode
    try:
        bin_shift = int(stats.mode(bin_shift_profiles)[0][0])
    except Exception as e:
        bin_shift = np.nan
        print(str(e))

    # rearrange results
    result = {
        'num_profiles': dim_p,
        'bins_arr': bins_arr.tolist(),
        'bin_shift': bin_shift,
        'r_bs': r_bin_shift_profiles,
        'ranges': ref_range_profiles,
        'signal_an': ref_signal_an_profiles,
        'signal_pc': ref_signal_pc_profiles
    }
    return bin_shift, result


def estimate_bin_zero_analog(signal, ranges, height_threshold=None, sigma_threshold=None):
    """
    Estimation of Bin Zero / Trigger Delay for analog channels

    Method:
    Bin Zero Estimation based on Near Target Test [Bravo-Aranda PhD Thesis, section 4.1.1.3]
    Backscatter signal produced by a near target. The signal peak should be found at bin=0.
    Otherwise will be the trigger delay
    The peak is meant to be found by using a number of signals and use the time average

    Parameters
    ----------
    signal: array
        signal produced by near target. 1D array (range) or 2D array (time, range)
    ranges: array
        Ranges. 1D array (range)
    height_threshold: float
        max height to evaluate DC. Float (m)
    sigma_threshold: float
        threshold in sigmas (std) for signal over noise to consider detection

    Returns
    -------
    bin_zero: float
        Bin Zero
    result: dict
        dictionary with results

    """

    # Set default values
    if height_threshold is None:
        height_threshold = 100
    if sigma_threshold is None:
        sigma_threshold = 3

    # signal num of profiles
    ndims = signal.ndim
    if ndims == 1:  # reshape to 2D (1, len(ranges))
        signal = signal.reshape(-1, len(ranges))
    n_p = signal.shape[0]

    # The method
    try:
        # trim signal profiles
        idx_bz = ranges < height_threshold
        ranges_bz = ranges[idx_bz]
        signal_bz = signal[:, idx_bz]

        # time-average signal profiles
        signal_bz_avg = np.nanmean(signal_bz, axis=0)

        xx = signal_bz_avg
        mm = np.nanmedian(signal_bz_avg)
        ss = np.nanstd(signal_bz_avg)
        idx = xx > mm + sigma_threshold*ss
        candidate = xx[idx]
        if len(candidate) > 0:
            bin_zero = int(np.argwhere(xx == np.nanmax(candidate))) + 1
        else:
            bin_zero = np.nan
            print("Peak no found")
    except Exception as e:
        bin_zero = np.nan
        ranges_bz = None
        signal_bz = None
        xx = None
        mm = None
        ss = None
        candidate = None
        print(str(e))
        print("ERROR. Bin Zero set to %i." % bin_zero)

    """ output for further analysis """
    result = {
        "bin_zero": bin_zero,
        "height_threshold": height_threshold,
        "sigma_threshold": sigma_threshold,
        "ranges": ranges_bz.tolist(),
        "signal": signal_bz.tolist(),
        "signal_time_avg": xx.tolist(),
        "signal_height_median": mm,
        "signal_height_std": ss,
        "candidate": candidate.tolist()
    }
    return bin_zero, result


def bin_zero_estimation(rs_fl, dc_fl, bin_zero_fn=None, data_dn=None):
    """
    Function for determining Bin Zero for each channel

    Parameters
    ----------
    rs_fl: str
        Wildcard Filelist of Raw Signal Measurements (/data_dir/*Prs*)
    dc_fl: str
        Wildcard Filelist of Dark Current Measurements (/data_dir/*Pdc*)
    bin_zero_fn: str
        fullpath for bin zero file to store the bin zero estimation
    data_dn: str
        full path of directory of data where bin zero file should be

    Returns
    -------
        bin zero for each channel is written in bin zero file

    """

    """ check if data_dn exists. Essential to store the bin zero estimation """
    if data_dn is None:
        data_dn = DATA_DN
    if not os.path.isdir(data_dn):
        print("ERROR. Data directory %s does not exist. Exit program" % data_dn)
        sys.exit()

    """ the estimation """
    try:
        # Read RS and DC data
        rs_ds = reader_xarray(rs_fl)
        dc_ds = reader_xarray(dc_fl)

        if np.logical_and(rs_ds is not None, dc_ds is not None):
            lidar_name = rs_ds.system
            # time, ranges, channels
            times = rs_ds.time.values
            ranges = rs_ds.range.values
            n_channels = rs_ds.dims['n_chan']
            ref_time = utils.numpy_to_datetime(times[0]).strftime("%Y%m%d")

            # wavelengths, polarizations, detection_modes for all channels
            wvs = rs_ds.wavelength.values
            pols = rs_ds.polarization.values
            mods = rs_ds.detection_mode.values

            # the estimation itself
            bzs = np.zeros(n_channels)
            bz_an_result = dict()
            bs_result = dict()
            for iwv in np.unique(wvs):
                for ipo in np.unique(pols[wvs == iwv]):
                    imods = mods[np.logical_and(wvs == iwv, pols == ipo)]
                    if len(imods) > 0:  # there are channels (wv, pol)
                        if np.logical_and.reduce((len(imods) >= 2, 0 in imods, 1 in imods)):  # analog and pc
                            channel_an = int(np.where(np.logical_and.reduce(
                                (wvs == iwv, pols == ipo, mods == 0)))[0])
                            channel_pc = int(np.where(np.logical_and.reduce(
                                (wvs == iwv, pols == ipo, mods == 1)))[0])
                            dc = dc_ds['signal_%02d' % channel_an].values
                            an = rs_ds['signal_%02d' % channel_an].values
                            pc = rs_ds['signal_%02d' % channel_pc].values
                            bzs[channel_an], bz_an_result_wv = estimate_bin_zero_analog(dc, ranges)
                            bin_shift, bs_result_wv = estimate_bin_shift(an, pc, ranges)
                            bzs[channel_pc] = bzs[channel_an] + bin_shift
                        elif np.logical_and(0 in imods, 1 not in imods):  # only analog
                            print("Only bin zero analog")
                            channel_an = int(np.where(np.logical_and.reduce((wvs == iwv, pols == ipo, mods == 0)))[0])
                            dc = dc_ds['signal_%02d' % channel_an].values
                            bzs[channel_an], bz_an_result_wv = estimate_bin_zero_analog(dc, ranges)
                            bs_result_wv = dict()
                        else:
                            bz_an_result_wv = dict()
                            bs_result_wv = dict()
                            print("No analog signal for wv=%s, pol=%s. Impossible to perform bin zero estimations"
                                  % (str(iwv), str(ipo)))
                        bz_an_result[str(iwv)] = bz_an_result_wv
                        bs_result[str(iwv)] = bs_result_wv
            """ write results in json files """
            # bin zero analog
            bin_zero_analog_dn = os.path.join(data_dn, lidar_name, "1a", "bin_zero_analog")
            if not os.path.isdir(bin_zero_analog_dn):
                mkpath(bin_zero_analog_dn)
            bz_an_json_fn = os.path.join(bin_zero_analog_dn, "bin_zero_analog_%s.json" % ref_time)
            with open(bz_an_json_fn, 'w') as fp:
                json.dump(bz_an_result, fp, sort_keys=True, indent=4)
            # bin shift
            bin_shift_dn = os.path.join(data_dn, lidar_name, "1a", "bin_shift")
            if not os.path.isdir(bin_shift_dn):
                mkpath(bin_shift_dn)
            bs_json_fn = os.path.join(bin_shift_dn, "bin_shift_%s.json" % ref_time)
            with open(bs_json_fn, 'w') as fp:
                json.dump(bs_result, fp, sort_keys=True, indent=4)

            """ write Bin Zero in Bin Zero File """
            for cc in range(n_channels):
                if ~np.isnan(wvs[cc]):
                    set_bin_zero(lidar_name, ref_time, wvs[cc], pols[cc], mods[cc], bzs[cc],
                                 bin_zero_fn=bin_zero_fn, data_dn=data_dn)
    except Exception as e:
        print("ERROR in bin_zero_estimation. %s" % str(e))


def bin_zero_estimation_period(date_ini_str, date_end_str, lidar_name=None, 
        bin_zero_fn=None, data_dn=None):
    """
    Estimation of bin zero for each channel for a period of lidar measurements

    Parameters
    ----------
    date_ini_str: str
        initial date (yyyymmdd)
    date_end_str: str
        end date (yyyymmdd)
    lidar_name: str
        name of lidar: MULHACEN
    bin_zero_fn: str
        fullpath for bin zero file to store the bin zero estimation
    data_dn: str
        full path of directory of data where bin zero file should be

    Returns
    -------
    Writes in bin_zero_fn the estimations for the period

    """

    """ Check Inputs """
    if lidar_name is None:
        lidar_name = "MULHACEN"
    if data_dn is None:
        data_dn = DATA_DN
    if not os.path.isdir(data_dn):
        print("ERROR. Data directory %s does not exist. Exit program" % data_dn)
        sys.exit()

    """ Loop over days """
    dates_arr = [x.strftime("%Y%m%d") for x in pd.date_range(date_ini_str, date_end_str)]
    for date_i in dates_arr:
        date_dn = os.path.join(data_dn, lidar_name, "1a", date_i[:4], date_i[4:6], date_i[6:])
        rs_fl = os.path.join(date_dn, "*%s*%s*" % ("Prs", date_i))
        dc_fl = os.path.join(date_dn, "*%s*%s*" % ("Pdc", date_i))
        bin_zero_estimation(rs_fl, dc_fl, bin_zero_fn=bin_zero_fn, data_dn=data_dn)
