import os
import sys
import platform
import shutil
import numpy as np
import pandas as pd
import datetime as dt

from utils_gfat import config

""" DEFAULT AUXILIAR INFO
"""
# Root Directory (in NASGFAT)  according to operative system
DATA_DN = config.DATA_DN


""" DEAD TIME
"""
def build_dead_time_filename(lidar_name, data_dn=None):
    """
    Build Full path for default dead time filename

    Parameters
    ----------
    lidar_name: str
        lidar name identifier
    data_dn: str
        data directory

    Returns
    -------
    dead_time_fn: str
        full path of dead time fn

    """
    if data_dn is None:
        data_dn = DATA_DN
    if not os.path.isdir(data_dn):
        print("WARNING. Data directory %s not found" % data_dn)
        print("dead time info will not be reached")
        return False
    if np.logical_or(lidar_name == 'mhc', lidar_name == "MULHACEN"):   # MULHACEN
        lidar_name = "MULHACEN"
    elif np.logical_or(lidar_name == 'vlt', lidar_name == "VELETA"):   # VELETA
        lidar_name = "VELETA"
    else:  # Unknown
        lidar_name = "LIDAR"

    dead_time_fn = os.path.join(data_dn, lidar_name, "1a", "dead_time.csv")
    return dead_time_fn


def create_dead_time_file(lidar_name, dead_time_fn=None, data_dn=None,
                          force=False, empty=False):
    """
    Create Dead Time File

    Parameters
    ----------
    lidar_name: str
        lidar system: mulhacen, veleta
    dead_time_fn: str
        full path of bin zero file
    data_dn: str
        full path of directory of data where dead time file will be created if dead_time_fn is None
    force: bool
        Force to create file
    empty: bool
        Create File Empty or with default data

    Returns
    -------

    """

    if dead_time_fn is None:
        dead_time_fn = build_dead_time_filename(lidar_name, data_dn=data_dn)

    if np.logical_or(force, not os.path.isfile(dead_time_fn)):
        f = open(dead_time_fn, 'w')
        f.write("# HISTORICAL VALUES OF DEAD TIME\n"
                "# polarization: none (0), parallel (1), perpendicular(2) \n"
                "# detection_mode: analog (0), photoncounting(1) \n"
                "# dead time (ns) \n")
        f.close()
        if not empty:
            if lidar_name == "MULHACEN":
                # Default Values [Bravo-Aranda PhD Thesis. Table 4.3]
                wvs = [355.0, 355.0, 387.0, 408.0, 532.0, 532.0, 532.0, 532.0,
                       607.0, 1064.0]
                pols = [0, 0, 0, 0, 1, 1, 2, 2, 0, 0]
                mods = [0, 1, 1, 1, 0, 1, 0, 1, 1, 1]
                taus = [(1.e3 / 270.) * x for x in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
                # TODO: 1064 es analog (tesis) o photoc (netcdf)?
            elif lidar_name == "VELETA":
                wvs = [-999]
                pols = [-999]
                mods = [-999]
                taus = [-999]
            date_df = [dt.datetime(1970, 1, 1)] * len(wvs)
            df = pd.DataFrame(
                {"date": date_df, "wavelength": wvs, "polarization": pols,
                 "detection_mode": mods, "dead_time": taus})
            # df.date = df.date.apply(lambda x: dt.datetime.strptime(x, "%Y%m%d"))
            df.to_csv(dead_time_fn, mode='a', index=False, na_rep=np.nan, date_format='%Y%m%d')
            shutil.copyfile(dead_time_fn, "%s.bkp" % dead_time_fn)
            print("File %s has been created with default values for dead time" % dead_time_fn)
    else:
        print("File %s already exists" % dead_time_fn)


def get_dead_time(lidar_name, wv, pol, mod, ref_time=None, dead_time_fn=None, data_dn=None):
    """
    Reads dead_time.csv and gets dead time for the channel defined by the three
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
    dead_time_fn: str
        dead time file
    data_dn: str
        full path of directory of data where bin zero file should be

    Returns
    -------
    tau: float
        dead time

    """
    print("INFO. Start Get Dead Time")

    if dead_time_fn is None:
        dead_time_fn = build_dead_time_filename(lidar_name, data_dn=data_dn)

    default = False
    try:
        if os.path.isfile(dead_time_fn):
            # read dead time file
            tau_ls = pd.read_csv(dead_time_fn, comment='#', parse_dates=["date"])
            tau_ls = tau_ls.rename(columns=lambda x: x.strip())

            # find tau for wv, pol, mod
            tau_ls = tau_ls.loc[np.logical_and.reduce((tau_ls.wavelength == wv,
                                                       tau_ls.polarization == pol,
                                                       tau_ls.detection_mode == mod))]
            if tau_ls.empty:
                default = True
                print("No dead time info for wv=%s, pol=%s, mode=%s" % (wv, pol, mod))
            else:
                if ref_time is None:
                    ref_time = dt.datetime.now()
                ref_time = pd.Timestamp(ref_time)
                # tau from closest date to ref_time
                dates_ls = tau_ls.date.sort_values().values
                i = 0
                ok = 0
                while ok == 0:
                    if i < len(tau_ls):
                        if (pd.Timestamp(dates_ls[i]) - ref_time) < dt.timedelta(0):
                            i += 1
                        else:
                            ok = 1
                    else:
                        ok = 1
                i = i - 1
                # nearest date with bz value
                tau = tau_ls.loc[tau_ls.date == dates_ls[i]].dead_time.values

                # check if tau is not nan. otherwise, look for previous dates
                # recursively until default values.
                # Otherwise: tau = default
                fin = False
                while not fin:
                    if np.isnan(tau):
                        i -= 1
                        if i >= 0:
                            tau = tau_ls.loc[tau_ls.date == dates_ls[i]].dead_time.values
                        else:
                            default = True
                            fin = True
                    else:
                        fin = True
        else:
            default = True
            print("File not found %s" % dead_time_fn)

    except Exception as e:
        default = True
        print(str(e))
        print("WARNING. dead time file not read")

    """ Default dead time """
    if default:
        tau = (1. / 270.) * 1e3  # ns
        print("dead time set to default: %6.3f (ns)" % tau)

    print("INFO. End Get Dead Time")

    return tau


def set_dead_time(lidar_name, ref_time, wv, pol, mod, tau, dead_time_fn=None, data_dn=None):
    """

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
    tau: float, array
        dead time
    dead_time_fn: str
        full path for dead time file
    data_dn: str
        full path of directory of data where dead time file should be

    Returns
    -------

    """

    try:
        # Prepare data into pandas dataframe
        data_dict = {'date': ref_time, 'wavelength': wv, 'polarization': pol,
                     'detection_mode': mod, 'bin_zero': tau}
        if wv.shape:  # if wv is array
            df = pd.DataFrame(data_dict)
        else:  # if wv is float
            df = pd.DataFrame(data_dict, index=[0])
        df.date = df.date.apply(lambda x: dt.datetime.strptime(x.strftime("%Y%m%d"), "%Y%m%d"))

        """ write data in dead time file """
        if dead_time_fn is None:
            dead_time_fn = build_dead_time_filename(lidar_name, data_dn=data_dn)
        if not os.path.isfile(dead_time_fn):  # create if it does not exist
            create_dead_time_file(force=True)
            df.to_csv(dead_time_fn, mode='a', index=False, header=False, na_rep=np.nan, date_format='%Y%m%d')
        else:  # add tau to file
            dateparse = lambda x: dt.datetime.strptime(x, '%Y%m%d')
            try:
                tau_ls = pd.read_csv(dead_time_fn, comment='#', parse_dates=["date"], date_parser=dateparse)
            except Exception as e:
                tau_ls = pd.read_csv("%s.bkp" % dead_time_fn, comment='#', parse_dates=["date"], date_parser=dateparse)
                print(str(e))
                print("WARNING: Problems reading %s. Bkp is read instead" % dead_time_fn)
            tau_ls = tau_ls.rename(columns=lambda x: x.strip())
            tau_ls['date'] = tau_ls['date'].apply(lambda x: x.strftime('%Y%m%d'))
            # add/update the tau value for (date, wv, pol, mod)
            for _, row in df.iterrows():
                # check the bz exists:
                idx = np.logical_and.reduce((tau_ls.date == row.date,
                                             tau_ls.wavelength == row.wavelength,
                                             tau_ls.polarization == row.polarization,
                                             tau_ls.detection_mode == row.detection_mode))
                if idx.any():  # substitute
                    tau_ls.loc[idx, "dead_time"] = row["dead_time"]
                else:  # add
                    tau_ls = tau_ls.append(row, ignore_index=True)
            # sort by ascending date and wavelength value and update file
            tau_ls = tau_ls.sort_values(by=['date', 'wavelength'])
            create_dead_time_file(lidar_name, dead_time_fn=dead_time_fn, force=True, empty=True)
            tau_ls.to_csv(dead_time_fn, mode='a', index=False, na_rep=np.nan, date_format='%Y%m%d')
            shutil.copyfile(dead_time_fn, "%s.bkp" % dead_time_fn)
    except Exception as e:
        print(str(e))
        print("ERROR. bin zero not set in file %s" % dead_time_fn)


def estimate_dead_time(signal_an, signal_pc, signal_gl, times, ranges):
    """
    Input:

    Output:
    """
    # TODO: Implementar
    print("To be implemented")
    return None
