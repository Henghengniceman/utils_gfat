"""
function needed to create plot with GFAT formatting
"""

import os
from cycler import cycler

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.dates as mdates
import datetime as dt 
import numpy as np
import pdb

BASE_DIR = os.path.dirname(__file__)

COEFF = 2.


COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c',
    '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22',
    '#17becf'
]

plt.rcParams['axes.prop_cycle'] = cycler('color', COLORS)

plt.rcParams['xtick.major.pad'] = 1.5 * COEFF
plt.rcParams['xtick.minor.pad'] = 1.5 * COEFF
plt.rcParams['ytick.major.pad'] = 1.5 * COEFF
plt.rcParams['ytick.minor.pad'] = 1.5 * COEFF

plt.rcParams['xtick.major.size'] = 1. * COEFF
plt.rcParams['xtick.minor.size'] = 1. * COEFF
plt.rcParams['ytick.major.size'] = 1. * COEFF
plt.rcParams['ytick.minor.size'] = 1. * COEFF

plt.rcParams['xtick.labelsize'] = 5 * COEFF
plt.rcParams['ytick.labelsize'] = 5 * COEFF

plt.rcParams['axes.linewidth'] = 0.5 * COEFF
plt.rcParams['axes.labelsize'] = 5 * COEFF
plt.rcParams['axes.facecolor'] = '#c7c7c7'

plt.rcParams['legend.numpoints'] = 3
plt.rcParams['legend.fontsize'] = 3.5 * COEFF
plt.rcParams['legend.facecolor'] = '#ffffff'

plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5 *COEFF
plt.rcParams['grid.alpha'] = 0.5


plt.rcParams['figure.subplot.hspace'] = 0.2
plt.rcParams['figure.subplot.wspace'] = 0.2
plt.rcParams['figure.subplot.bottom'] = .11
plt.rcParams['figure.subplot.left'] = .14
plt.rcParams['figure.subplot.right'] = .95
plt.rcParams['figure.subplot.top'] = .82

plt.rcParams['figure.figsize'] = 2.913 * COEFF, 2.047 * COEFF
plt.rcParams['figure.facecolor'] = '#ffffff'

plt.rcParams['lines.markersize'] = 2.6 * COEFF
plt.rcParams['lines.markeredgewidth'] = 0.5 * COEFF
plt.rcParams['lines.linewidth'] = 0.5 * COEFF


def title1(mytitle, coef):
    """
    inclus le titre au document.
        @param mytitle: titre du document.
        @param coef : coefficient GFAT (renvoye par la fonction formatGFAT).
    """

    plt.figtext(0.5, 0.95, mytitle, fontsize=6.5*coef, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')
    return


def title2(mytitle, coef):
    """
    inclus le sous titre au document.
        @param mytitle: titre du document.
        @param coef : coefficient GFAT (renvoye par la fonction formatGFAT).
    """

    plt.figtext(0.5, 0.89, mytitle, fontsize=5.5*coef,
                horizontalalignment='center', verticalalignment='center')
    return


def title3(mytitle, coef):
    """
    inclus le sous sous titre au document.
        @param mytitle: titre du document.
        @param coef : coefficient GFAT (renvoye par la fonction formatGFAT).
    """
    plt.figtext(0.5, 0.85, mytitle, fontsize=4.5*coef,
                horizontalalignment='center', verticalalignment='center')
    return

def watermark(fig, ax, scale=15, alpha=0.25, xpos=65, ypos=315, logofile='GFAT'):    
    """ Place watermark in bottom right of figure. 
    fig: figure handle
    ax: axes handle
    alpha: alpha channel, ie transparency
    xpos: horizontal location of the figure in pixel
    ypos: vertical location of the figure in pixel
    logofile: file path of the image to use as logo. Default: 'GFAT' redirects to the GFAT logo.
    """

    # Get the pixel dimensions of the figure
    width, height = fig.get_size_inches()*fig.dpi

    # Import logo and scale accordingly
    if logofile == 'GFAT':
        logofile = os.path.join(BASE_DIR, 'logos', 'LOGO_GFAT_150pp.png')
    img = Image.open(logofile)
    wm_width = int(width/scale) # make the watermark 1/10 of the figure size
    scaling = (wm_width / float(img.size[0]))
    wm_height = int(float(img.size[1])*float(scaling))
    img = img.resize((wm_width, wm_height), Image.ANTIALIAS)

    # Place the watermark in the lower right of the figure
    plt.figimage(img, xpos, ypos, alpha=alpha, zorder=1)

def tmp_f(date_dt):
    """
    convert datetime to numerical date
    """
    return mdates.num2date(date_dt).strftime('%H')

def color_list(n):
    import matplotlib.pylab as pl
    colors = pl.cm.jet(np.linspace(0,1,n))
    return colors 

def font_axes(ax,fontsize=14):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

def gapsizer(ax, time, range, gapsize, colour='#c7c7c7'):
    """
    This function creates a rectangle of color 'colour' when time gap 
    are found in the array 'time'. 
    """
        # search for holes in data
    # --------------------------------------------------------------------
    dif_time = time[1:] - time[0:-1]
    print(type(dif_time))
    for index, delta in enumerate(dif_time):
        if delta > dt.timedelta(minutes=gapsize):
            # missing hide bad data
            start = mdates.date2num(time[index])
            end = mdates.date2num(time[index + 1])
            width = end - start

            # Plot rectangle
            end = mdates.date2num(time[index + 1])
            rect = mpl.patches.Rectangle(
                (start, 0), width, np.nanmax(range),
                color=colour)
            ax.add_patch(rect)

class OOMFormatter(mpl.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        mpl.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % mpl.ticker._mathdefault(self.format)
