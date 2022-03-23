import platform
from pathlib import Path

from utils_gfat import logs

logger = logs.create_logger()

# Root Directory (in NASGFAT)  according to operative system
if platform.system() == 'Windows':
    #SYSTEM_DN = "Y:"
    SYSTEM_DN = "../" #Here I change the path into my PC
else:
    SYSTEM_DN = "/mnt/NASGFAT"
SYSTEM_DN = Path(SYSTEM_DN)
# DATA_DN = Path.joinpath(SYSTEM_DN, "datos")
DATA_DN = SYSTEM_DN


# Set DATA_DN
def set_data_dn(dn):
    """[summary]

    Args:
        dn ([type]): [description]
    """
    dn = Path(dn)
    if dn.is_dir():
        DATA_DN = Path(dn)
        logger.info("%s is the new DATA_DN" % dn)
    else:
        logger.error("%s does not exist" % dn)