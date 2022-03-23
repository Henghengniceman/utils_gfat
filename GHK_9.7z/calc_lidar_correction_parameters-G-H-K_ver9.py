# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:33:10 2016
@author: Volker.Freudenthaler@lmu.de
calc_lidar_correction_parameters-G-H-K_ver2b.py: with output to text.file
               ver2c.py: with output of input values
                  ver3a.py: option to remove the rotational error epsilon for normal measurements  08.07.16, vf
                  ver3b.py: several bugs fixed 08.07.16, vf
                  ver3c.py: some code lines moved in the if structures and combined at end => now the option to remove the rotational error epsilon for normal measurements works 09.07.16, vf
                  ver4a.py: error loops 09.07.16, vf
                  ver4b.py: with function 09.07.16, vf
                  ver4c.py: function and for loop split to speed up the code 09.07.16, vf
                              is faster (9 instead of 16 sec) but less clear code
                  ver4c2.py:  S2g Bug in B fixed
                  ver4c3.py:    incl. loop over LDRtrue with plot of errors
                  ver4c4.py: incl. PollyXT with ER error
                  ver4c5.py: colored hist overlays for certain parameters
                  ver4c6.py: colored hist overlays for certain parameters in function
                  ver4c5c  :
                  ver5a    : with Type = 6 : retarding diattenuator at +-22.5° (with 180° retardance = HWP), first vector equations
                  ver6     : with rotated Pol-Filters behind the PBS + some vector equations, only for Loc = 3
                          todo: correct unpol transmittance; compare with ver 5a.
                  ver6b
                  ver6c    rotated Pol-Filters and TypeC = 6 for all LocC; QWP calibrator; resorting of code; correct rotator calib without rot-error;
                  ver6d    plots also with iPython and python command prompt (under Anaconda at least)
                  ver6e    varying LDRCal
                        ver6f    angles from degree to rad before loop ( only 2% less processor time)
                  ver6g    varying LDRCal and K calculated for assumed setup (input ver6e)
                  ver6h    Trying to correct the absolute values of GH
                  ver6i    Several bugs fixed: => most GH equations newly formulated.  use ver6e inputs
                  ver7     "  just a new main version for ver6i - now saving LDRMIN - LDRMAX to file
                  ver7a    Bugfix:  when NOT explicitly varying LDRCal, not LDRCal0 is used, but the last LDRCal = 0.45 from the previous loop over LDRCal to determine K(LDRCal)
                  ver7b    cosmetic changes: YesNo function, plot title, warning if N is too large, elapsed time; equation source: rotated_diattenuator_X22x5deg.odt
                  ver8a    output Itotal (F11) with error
                  ver8b    more code comments
                  ver8c    Tp, Ts, Rp, Rs individually with individual errors
                  ver8c-Ralph-6  For POLLY_XT   RP = 1 - TP  RS = 1 - TS
                  ver9     Output Infos to Console and File
                  ver9-Ralph-6  For POLLY_XT   RP = 1 - TP  RS = 1 - TS
                  ver9-PollXT  same as before


Equation reference: http://www.atmos-meas-tech-discuss.net/amt-2015-338/amt-2015-338.pdf
With equations code from Appendix C
Python 3.4.2
"""
#!/usr/bin/env python3
from __future__ import print_function
#import math
import numpy as np
import sys
import os
#import seaborn as sns
import matplotlib.pyplot as plt
from time import clock

#from matplotlib.backends.backend_pdf import PdfPages
#pdffile = '{}.pdf'.format('path')
#pp = PdfPages(pdffile)
## pp.savefig can be called multiple times to save to multiple pages
#pp.savefig()
#pp.close()

from contextlib import contextmanager
@contextmanager
def redirect_stdout(new_target):
    old_target, sys.stdout = sys.stdout, new_target # replace sys.stdout
    try:
        yield new_target # run some code with the replaced stdout
    finally:
        sys.stdout.flush()
        sys.stdout = old_target # restore to the previous value
'''
real_raw_input = vars(__builtins__).get('raw_input',input)
'''
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass

from distutils.util import strtobool
def user_yes_no_query(question):
    sys.stdout.write('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            sys.stdout.write('Please respond with \'y\' or \'n\'.\n')

#if user_yes_no_query('want to exit?') == 1: sys.exit()

'''
## {{{ http://code.activestate.com/recipes/577058/ (r2)
def query_yes_no(question, default="yes"):
    valid = {"yes":"yes",   "y":"yes",  "ye":"yes",
             "no":"no",     "n":"no"}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while 1:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")
## end of http://code.activestate.com/recipes/577058/ }}}
'''
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
fname = os.path.basename(abspath)
os.chdir(dname)

#PrintToOutputFile = True

sqr05 = 0.5**0.5

# ---- Initial definition of variables; the actual values will be read in with exec(open('./optic_input.py').read()) below
LID = "internal"
EID = "internal"
# --- IL Laser IL and +-Uncertainty
bL = 1.    #degree of linear polarization; default 1
RotL, dRotL, nRotL     = 0.0, 0.0,     1    #alpha; rotation of laser polarization in degrees; default 0
# --- ME Emitter and +-Uncertainty
DiE, dDiE, nDiE     = 0., 0.00,     1    # Diattenuation
TiE         = 1.        # Unpolarized transmittance
RetE, dRetE, nRetE     = 0., 180.0,     0    # Retardance in degrees
RotE, dRotE, nRotE     = 0., 0.0,     0    # beta: Rotation of optical element in degrees
# --- MO Receiver Optics including telescope
DiO,  dDiO, nDiO     = -0.055, 0.003,     1
TiO                 = 0.9
RetO, dRetO, nRetO     = 0., 180.0,     2
RotO, dRotO, nRotO     = 0., 0.1,     1    #gamma
# --- PBS MT transmitting path defined with (TS,TP);  and +-Uncertainty
TP,   dTP, nTP     = 0.98,     0.02,    1
TS,   dTS, nTS     = 0.001, 0.001,    1
TiT = 0.5 * (TP + TS)
DiT = (TP-TS)/(TP+TS)
# PolFilter
RetT, dRetT, nRetT  = 0.,     180.,    0
ERaT, dERaT, nERaT  = 0.001, 0.001,    1
RotaT, dRotaT, nRotaT = 0.,     3., 1
DaT = (1-ERaT)/(1+ERaT)
TaT = 0.5*(1+ERaT)
# --- PBS MR reflecting path defined with (RS,RP);  and +-Uncertainty
RS, dRS, nRS = 1 - TS,  0., 0
RP, dRP, nRP = 1 - TP,  0., 0
TiR = 0.5 * (RP + RS)
DiR = (RP-RS)/(RP+RS)
# PolFilter
RetR, dRetR, nRetR  = 0.,     180.,     0
ERaR, dERaR, nERaR  = 0.001, 0.001, 1
RotaR,dRotaR,nRotaR = 90.,     3.,        1
DaR = (1-ERaR)/(1+ERaR)
TaR = 0.5*(1+ERaR)

# Parellel signal detected in the transmitted channel => Y = 1, or in the reflected channel => Y = -1
Y = -1.

# Calibrator =  type defined by matrix values
LocC = 4     # location of calibrator: behind laser = 1; behind emitter = 2; before receiver = 3; before PBS = 4

TypeC = 3   # linear polarizer calibrator
# example with extinction ratio 0.001
DiC, dDiC, nDiC     = 1.0,     0.,     0    # ideal 1.0
TiC = 0.5    # ideal 0.5
RetC, dRetC, nRetC     = 0.,     0.,     0
RotC, dRotC, nRotC     = 0.0,     0.1,     0    #constant calibrator offset epsilon
RotationErrorEpsilonForNormalMeasurements = False    #     is in general False for TypeC == 3 calibrator

# Rotation error without calibrator: if False, then epsilon = 0 for normal measurements
RotationErrorEpsilonForNormalMeasurements = True

# LDRCal assumed atmospheric linear depolarization ratio during the calibration measurements (first guess)
LDRCal0,dLDRCal,nLDRCal= 0.25, 0.04, 1
LDRCal = LDRCal0
# measured LDRm will be corrected with calculated parameters
LDRmeas = 0.015
# LDRtrue for simulation of measurement => LDRsim
LDRtrue = 0.5
LDRtrue2 = 0.004

# Initialize other values to 0
ER, nER, dER = 0.001, 0, 0.001
K = 0.
Km = 0.
Kp = 0.
LDRcorr = 0.
Eta = 0.
Ir = 0.
It = 0.
h = 1.

Loc = ['', 'behind laser', 'behind emitter', 'before receiver', 'before PBS']
Type = ['', 'mechanical rotator', 'hwp rotator', 'linear polarizer', 'qwp rotator', 'circular polarizer', 'real HWP +-22.5°']
dY = ['reflected channel', '', 'transmitted channel']

#  end of initial definition of variables
# *******************************************************************************************************************************

# --- Read actual lidar system parameters from ./optic_input.py  (must be in the same directory)

#InputFile = 'optic_input_ver6e_POLIS_355.py'
#InputFile = 'optic_input_ver6e_POLIS_355_JA.py'
#InputFile = 'optic_input_ver6c_POLIS_532.py'
#InputFile = 'optic_input_ver6e_POLIS_532.py'
#InputFile = 'optic_input_ver8c_POLIS_532.py'
#InputFile = 'optic_input_ver6e_MUSA.py'
#InputFile = 'optic_input_ver6e_MUSA_JA.py'
#InputFile = 'optic_input_ver6e_PollyXTSea.py'
#InputFile = 'optic_input_ver6e_PollyXTSea_JA.py'
#InputFile = 'optic_input_ver6e_PollyXT_RALPH.py'
#InputFile = 'optic_input_ver8c_PollyXT_RALPH.py'
#InputFile = 'optic_input_ver8c_PollyXT_RALPH_2.py'
#InputFile = 'optic_input_ver8c_PollyXT_RALPH_3.py'
#InputFile = 'optic_input_ver8c_PollyXT_RALPH_4.py'
#InputFile = 'optic_input_ver8c_PollyXT_RALPH_5.py'
#InputFile = 'optic_input_ver8c_PollyXT_RALPH_6.py'
#InputFile = 'optic_input_ver8c_PollyXT_RALPH_7.py'
#InputFile = 'optic_input_ver8a_MOHP_DPL_355.py'
InputFile = 'optic_input_ver9_MOHP_DPL_355.py'
#InputFile = 'optic_input_ver6e_RALI.py'
#InputFile = 'optic_input_ver6e_RALI_JA.py'
#InputFile = 'optic_input_ver6e_RALI_new.py'
#InputFile = 'optic_input_ver6e_RALI_act.py'
#InputFile = 'optic_input_ver6e_MULHACEN.py'
#InputFile = 'optic_input_ver6e_MULHACEN_JA.py'
#InputFile = 'optic_input_ver6e-IPRAL.py'
#InputFile = 'optic_input_ver6e-IPRAL_JA.py'
#InputFile = 'optic_input_ver6e-LB21.py'
#InputFile = 'optic_input_ver6e-LB21_JA.py'
#InputFile = 'optic_input_ver6e_Bertha_b_355.py'
#InputFile = 'optic_input_ver6e_Bertha_b_532.py'
#InputFile = 'optic_input_ver6e_Bertha_b_1064.py'

'''
print("From ", dname)
print("Running ", fname)
print("Reading input file ", InputFile, " for")
'''
# this works with Python 2 - and 3?
exec(open('./'+InputFile).read(), globals())
#  end of read actual system parameters

# --- Manual Parameter Change ---
#  (use for quick parameter changes without changing the input file )
#DiO = 0.
#LDRtrue = 0.45
#LDRtrue2 = 0.004
#Y = -1
#LocC = 4 #location of calibrator: 1 = behind laser; 2 = behind emitter; 3 = before receiver; 4 = before PBS
##TypeC = 6  Don't change the TypeC here
#RotationErrorEpsilonForNormalMeasurements = True
#LDRCal = 0.25
#bL = 0.8
## --- Errors
RotL0, dRotL, nRotL = RotL, dRotL, nRotL

DiE0,  dDiE,  nDiE  = DiE,  dDiE,  nDiE
RetE0, dRetE, nRetE = RetE, dRetE, nRetE
RotE0, dRotE, nRotE = RotE, dRotE, nRotE

DiO0,  dDiO,  nDiO  = DiO,  dDiO,  nDiO
RetO0, dRetO, nRetO = RetO, dRetO, nRetO
RotO0, dRotO, nRotO = RotO, dRotO, nRotO

DiC0,  dDiC,  nDiC  = DiC,  dDiC,  nDiC
RetC0, dRetC, nRetC = RetC, dRetC, nRetC
RotC0, dRotC, nRotC = RotC, dRotC, nRotC

TP0,   dTP,   nTP   = TP,   dTP,     nTP
TS0,   dTS,   nTS   = TS,   dTS,     nTS
RetT0, dRetT, nRetT = RetT, dRetT, nRetT

ERaT0, dERaT, nERaT = ERaT, dERaT, nERaT
RotaT0,dRotaT,nRotaT= RotaT,dRotaT,nRotaT

RP0,   dRP,   nRP   = RP,   dRP,   nRP
RS0,   dRS,   nRS   = RS,   dRS,   nRS
RetR0, dRetR, nRetR = RetR, dRetR, nRetR

ERaR0, dERaR, nERaR = ERaR, dERaR, nERaR
RotaR0,dRotaR,nRotaR= RotaR,dRotaR,nRotaR

LDRCal0,dLDRCal,nLDRCal=LDRCal,dLDRCal,nLDRCal
#LDRCal0,dLDRCal,nLDRCal=LDRCal,dLDRCal,0
# ---------- End of manual parameter change

RotL, RotE, RetE, DiE, RotO, RetO, DiO, RotC, RetC, DiC = RotL0, RotE0, RetE0, DiE0, RotO0, RetO0, DiO0, RotC0, RetC0, DiC0
TP, TS, RP, RS, ERaT, RotaT, RetT, ERaR, RotaR, RetR = TP0, TS0, RP0, RS0 , ERaT0, RotaT0, RetT0, ERaR0, RotaR0, RetR0
LDRCal = LDRCal0
DTa0, TTa0, DRa0, TRa0, LDRsimx, LDRCorr = 0,0,0,0,0,0

TiT = 0.5 * (TP + TS)
DiT = (TP-TS)/(TP+TS)
ZiT = (1. - DiT**2)**0.5
TiR = 0.5 * (RP + RS)
DiR = (RP-RS)/(RP+RS)
ZiR = (1. - DiR**2)**0.5

# --------------------------------------------------------
def Calc(RotL, RotE, RetE, DiE, RotO, RetO, DiO, RotC, RetC, DiC, TP, TS, RP, RS, ERaT, RotaT, RetT, ERaR, RotaR, RetR, LDRCal):
    # ---- Do the calculations of bra-ket vectors
    h = -1. if TypeC == 2 else 1
    # from input file:  assumed LDRCal for calibration measurements
    aCal = (1.-LDRCal)/(1+LDRCal)
    # from input file: measured LDRm and true LDRtrue, LDRtrue2  =>
    #ameas = (1.-LDRmeas)/(1+LDRmeas)
    atrue = (1.-LDRtrue)/(1+LDRtrue)
    #atrue2 = (1.-LDRtrue2)/(1+LDRtrue2)

    # angles of emitter and laser and calibrator and receiver optics
    # RotL = alpha, RotE = beta, RotO = gamma, RotC = epsilon
    S2a = np.sin(2*np.deg2rad(RotL))
    C2a = np.cos(2*np.deg2rad(RotL))
    S2b = np.sin(2*np.deg2rad(RotE))
    C2b = np.cos(2*np.deg2rad(RotE))
    S2ab = np.sin(np.deg2rad(2*RotL-2*RotE))
    C2ab = np.cos(np.deg2rad(2*RotL-2*RotE))
    S2g = np.sin(np.deg2rad(2*RotO))
    C2g = np.cos(np.deg2rad(2*RotO))

    # Laser with Degree of linear polarization DOLP = bL
    IinL = 1.
    QinL = bL
    UinL = 0.
    VinL = (1. - bL**2)**0.5

    # Stokes Input Vector rotation Eq. E.4
    A = C2a*QinL - S2a*UinL
    B = S2a*QinL + C2a*UinL
    # Stokes Input Vector rotation Eq. E.9
    C = C2ab*QinL - S2ab*UinL
    D = S2ab*QinL + C2ab*UinL

    # emitter optics
    CosE = np.cos(np.deg2rad(RetE))
    SinE = np.sin(np.deg2rad(RetE))
    ZiE = (1. - DiE**2)**0.5
    WiE = (1. - ZiE*CosE)

    # Stokes Input Vector after emitter optics equivalent to Eq. E.9 with already rotated input vector from Eq. E.4
    # b = beta
    IinE = (IinL + DiE*C)
    QinE = (C2b*DiE*IinL + A + S2b*(WiE*D - ZiE*SinE*VinL))
    UinE = (S2b*DiE*IinL + B - C2b*(WiE*D - ZiE*SinE*VinL))
    VinE = (-ZiE*SinE*D + ZiE*CosE*VinL)

    # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
    IinF = IinE
    QinF = aCal*QinE
    UinF = -aCal*UinE
    VinF = (1.-2.*aCal)*VinE

    # receiver optics
    CosO = np.cos(np.deg2rad(RetO))
    SinO = np.sin(np.deg2rad(RetO))
    ZiO = (1. - DiO**2)**0.5
    WiO = (1. - ZiO*CosO)

    # calibrator
    CosC = np.cos(np.deg2rad(RetC))
    SinC = np.sin(np.deg2rad(RetC))
    ZiC = (1. - DiC**2)**0.5
    WiC = (1. - ZiC*CosC)

    # Stokes Input Vector before the polarising beam splitter Eq. E.31
    A = C2g*QinE - S2g*UinE
    B = S2g*QinE + C2g*UinE

    IinP = (IinE + DiO*aCal*A)
    QinP = (C2g*DiO*IinE + aCal*QinE - S2g*(WiO*aCal*B + ZiO*SinO*(1-2*aCal)*VinE))
    UinP = (S2g*DiO*IinE - aCal*UinE + C2g*(WiO*aCal*B + ZiO*SinO*(1-2*aCal)*VinE))
    VinP = (ZiO*SinO*aCal*B + ZiO*CosO*(1-2*aCal)*VinE)

    #-------------------------
    # F11 assuemd to be = 1  => measured: F11m = IinP / IinE with atrue
    #F11sim = TiO*(IinE + DiO*atrue*A)/IinE
    #-------------------------

    # For PollyXT
    # analyser
    #RS = 1 - TS
    #RP = 1 - TP

    TiT = 0.5 * (TP + TS)
    DiT = (TP-TS)/(TP+TS)
    ZiT = (1. - DiT**2)**0.5
    TiR = 0.5 * (RP + RS)
    DiR = (RP-RS)/(RP+RS)
    ZiR = (1. - DiR**2)**0.5
    CosT = np.cos(np.deg2rad(RetT))
    SinT = np.sin(np.deg2rad(RetT))
    CosR = np.cos(np.deg2rad(RetR))
    SinR = np.sin(np.deg2rad(RetR))

    DaT = (1-ERaT)/(1+ERaT)
    DaR = (1-ERaR)/(1+ERaR)
    TaT = 0.5*(1+ERaT)
    TaR = 0.5*(1+ERaR)

    S2aT = np.sin(np.deg2rad(h*2*RotaT))
    C2aT = np.cos(np.deg2rad(2*RotaT))
    S2aR = np.sin(np.deg2rad(h*2*RotaR))
    C2aR = np.cos(np.deg2rad(2*RotaR))

    # Aanalyzer As before the PBS Eq. D.5
    ATP1 = (1+C2aT*DaT*DiT)
    ATP2 = Y*(DiT+C2aT*DaT)
    ATP3 = Y*S2aT*DaT*ZiT*CosT
    ATP4 = S2aT*DaT*ZiT*SinT
    ATP = np.array([ATP1,ATP2,ATP3,ATP4])

    ARP1 = (1+C2aR*DaR*DiR)
    ARP2 = Y*(DiR+C2aR*DaR)
    ARP3 = Y*S2aR*DaR*ZiR*CosR
    ARP4 = S2aR*DaR*ZiR*SinR
    ARP = np.array([ARP1,ARP2,ARP3,ARP4])

    DTa = ATP2*Y/ATP1
    DRa = ARP2*Y/ARP1

    # ---- Calculate signals and correction parameters for diffeent locations and calibrators
    if LocC == 4:  # Calibrator before the PBS
        #print("Calibrator location not implemented yet")

        #S2ge = np.sin(np.deg2rad(2*RotO + h*2*RotC))
        #C2ge = np.cos(np.deg2rad(2*RotO + h*2*RotC))
        S2e = np.sin(np.deg2rad(h*2*RotC))
        C2e = np.cos(np.deg2rad(2*RotC))
        # rotated AinP by epsilon Eq. C.3
        ATP2e = C2e*ATP2 + S2e*ATP3
        ATP3e = C2e*ATP3 - S2e*ATP2
        ARP2e = C2e*ARP2 + S2e*ARP3
        ARP3e = C2e*ARP3 - S2e*ARP2
        ATPe = np.array([ATP1,ATP2e,ATP3e,ATP4])
        ARPe = np.array([ARP1,ARP2e,ARP3e,ARP4])
        # Stokes Input Vector before the polarising beam splitter Eq. E.31
        A = C2g*QinE - S2g*UinE
        B = S2g*QinE + C2g*UinE
        #C = (WiO*aCal*B + ZiO*SinO*(1-2*aCal)*VinE)
        Co = ZiO*SinO*VinE
        Ca = (WiO*B - 2*ZiO*SinO*VinE)
        #C = Co + aCal*Ca
        #IinP = (IinE + DiO*aCal*A)
        #QinP = (C2g*DiO*IinE + aCal*QinE - S2g*C)
        #UinP = (S2g*DiO*IinE - aCal*UinE + C2g*C)
        #VinP = (ZiO*SinO*aCal*B + ZiO*CosO*(1-2*aCal)*VinE)
        IinPo = IinE
        QinPo = (C2g*DiO*IinE - S2g*Co)
        UinPo = (S2g*DiO*IinE + C2g*Co)
        VinPo = ZiO*CosO*VinE

        IinPa = DiO*A
        QinPa = QinE - S2g*Ca
        UinPa = -UinE + C2g*Ca
        VinPa = ZiO*(SinO*B - 2*CosO*VinE)

        IinP = IinPo + aCal*IinPa
        QinP = QinPo + aCal*QinPa
        UinP = UinPo + aCal*UinPa
        VinP = VinPo + aCal*VinPa
        # Stokes Input Vector before the polarising beam splitter rotated by epsilon Eq. C.3
        #QinPe = C2e*QinP + S2e*UinP
        #UinPe = C2e*UinP - S2e*QinP
        QinPoe = C2e*QinPo + S2e*UinPo
        UinPoe = C2e*UinPo - S2e*QinPo
        QinPae = C2e*QinPa + S2e*UinPa
        UinPae = C2e*UinPa - S2e*QinPa
        QinPe = C2e*QinP + S2e*UinP
        UinPe = C2e*UinP - S2e*QinP

        # Calibration signals and Calibration correction K from measurements with LDRCal / aCal
        if (TypeC == 2) or (TypeC == 1):  # rotator calibration Eq. C.4
            # parameters for calibration with aCal
            AT = ATP1*IinP + h*ATP4*VinP
            BT = ATP3e*QinP - h*ATP2e*UinP
            AR = ARP1*IinP + h*ARP4*VinP
            BR = ARP3e*QinP - h*ARP2e*UinP
            # Correction paremeters for normal measurements; they are independent of LDR
            if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                IS1 = np.array([IinPo,QinPo,UinPo,VinPo])
                IS2 = np.array([IinPa,QinPa,UinPa,VinPa])
                GT = np.dot(ATP,IS1)
                GR = np.dot(ARP,IS1)
                HT = np.dot(ATP,IS2)
                HR = np.dot(ARP,IS2)
            else:
                IS1 = np.array([IinPo,QinPo,UinPo,VinPo])
                IS2 = np.array([IinPa,QinPa,UinPa,VinPa])
                GT = np.dot(ATPe,IS1)
                GR = np.dot(ARPe,IS1)
                HT = np.dot(ATPe,IS2)
                HR = np.dot(ARPe,IS2)
        elif (TypeC == 3) or (TypeC == 4):  # linear polariser calibration Eq. C.5
            # parameters for calibration with aCal
            AT = ATP1*IinP + ATP3e*UinPe + ZiC*CosC*(ATP2e*QinPe + ATP4*VinP)
            BT = DiC*(ATP1*UinPe + ATP3e*IinP) - ZiC*SinC*(ATP2e*VinP - ATP4*QinPe)
            AR = ARP1*IinP + ARP3e*UinPe + ZiC*CosC*(ARP2e*QinPe + ARP4*VinP)
            BR = DiC*(ARP1*UinPe + ARP3e*IinP) - ZiC*SinC*(ARP2e*VinP - ARP4*QinPe)
            # Correction paremeters for normal measurements; they are independent of LDR
            if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                IS1 = np.array([IinPo,QinPo,UinPo,VinPo])
                IS2 = np.array([IinPa,QinPa,UinPa,VinPa])
                GT = np.dot(ATP,IS1)
                GR = np.dot(ARP,IS1)
                HT = np.dot(ATP,IS2)
                HR = np.dot(ARP,IS2)
            else:
                IS1e = np.array([IinPo+DiC*QinPoe,DiC*IinPo+QinPoe,ZiC*(CosC*UinPoe+SinC*VinPo),-ZiC*(SinC*UinPoe-CosC*VinPo)])
                IS2e = np.array([IinPa+DiC*QinPae,DiC*IinPa+QinPae,ZiC*(CosC*UinPae+SinC*VinPa),-ZiC*(SinC*UinPae-CosC*VinPa)])
                GT = np.dot(ATPe,IS1e)
                GR = np.dot(ARPe,IS1e)
                HT = np.dot(ATPe,IS2e)
                HR = np.dot(ARPe,IS2e)
        elif (TypeC == 6):  # diattenuator calibration +-22.5° rotated_diattenuator_X22x5deg.odt
            # parameters for calibration with aCal
            AT = ATP1*IinP + sqr05*DiC*(ATP1*QinPe + ATP2e*IinP) + (1-0.5*WiC)*(ATP2e*QinPe + ATP3e*UinPe) + ZiC*(sqr05*SinC*(ATP3e*VinP-ATP4*UinPe) + ATP4*CosC*VinP)
            BT = sqr05*DiC*(ATP1*UinPe + ATP3e*IinP) + 0.5*WiC*(ATP2e*UinPe + ATP3e*QinPe) - sqr05*ZiC*SinC*(ATP2e*VinP - ATP4*QinPe)
            AR = ARP1*IinP + sqr05*DiC*(ARP1*QinPe + ARP2e*IinP) + (1-0.5*WiC)*(ARP2e*QinPe + ARP3e*UinPe) + ZiC*(sqr05*SinC*(ARP3e*VinP-ARP4*UinPe) + ARP4*CosC*VinP)
            BR = sqr05*DiC*(ARP1*UinPe + ARP3e*IinP) + 0.5*WiC*(ARP2e*UinPe + ARP3e*QinPe) - sqr05*ZiC*SinC*(ARP2e*VinP - ARP4*QinPe)
            # Correction paremeters for normal measurements; they are independent of LDR
            if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                IS1 = np.array([IinPo,QinPo,UinPo,VinPo])
                IS2 = np.array([IinPa,QinPa,UinPa,VinPa])
                GT = np.dot(ATP,IS1)
                GR = np.dot(ARP,IS1)
                HT = np.dot(ATP,IS2)
                HR = np.dot(ARP,IS2)
            else:
                IS1e = np.array([IinPo+DiC*QinPoe,DiC*IinPo+QinPoe,ZiC*(CosC*UinPoe+SinC*VinPo),-ZiC*(SinC*UinPoe-CosC*VinPo)])
                IS2e = np.array([IinPa+DiC*QinPae,DiC*IinPa+QinPae,ZiC*(CosC*UinPae+SinC*VinPa),-ZiC*(SinC*UinPae-CosC*VinPa)])
                GT = np.dot(ATPe,IS1e)
                GR = np.dot(ARPe,IS1e)
                HT = np.dot(ATPe,IS2e)
                HR = np.dot(ARPe,IS2e)
        else:
            print("Calibrator not implemented yet")
            sys.exit()

    elif LocC == 3:  # C before receiver optics Eq.57

        #S2ge = np.sin(np.deg2rad(2*RotO - 2*RotC))
        #C2ge = np.cos(np.deg2rad(2*RotO - 2*RotC))
        S2e = np.sin(np.deg2rad(2*RotC))
        C2e = np.cos(np.deg2rad(2*RotC))

        # As with C before the receiver optics (rotated_diattenuator_X22x5deg.odt)
        AF1 = np.array([1,C2g*DiO,S2g*DiO,0])
        AF2 = np.array([C2g*DiO,1-S2g**2*WiO,S2g*C2g*WiO,-S2g*ZiO*SinO])
        AF3 = np.array([S2g*DiO,S2g*C2g*WiO,1-C2g**2*WiO,C2g*ZiO*SinO])
        AF4 = np.array([0,S2g*SinO,-C2g*SinO,CosO])

        ATF = (ATP1*AF1+ATP2*AF2+ATP3*AF3+ATP4*AF4)
        ARF = (ARP1*AF1+ARP2*AF2+ARP3*AF3+ARP4*AF4)
        ATF2 = ATF[1]
        ATF3 = ATF[2]
        ARF2 = ARF[1]
        ARF3 = ARF[2]

        # rotated AinF by epsilon
        ATF1 = ATF[0]
        ATF4 = ATF[3]
        ATF2e = C2e*ATF[1] + S2e*ATF[2]
        ATF3e = C2e*ATF[2] - S2e*ATF[1]
        ARF1 = ARF[0]
        ARF4 = ARF[3]
        ARF2e = C2e*ARF[1] + S2e*ARF[2]
        ARF3e = C2e*ARF[2] - S2e*ARF[1]

        ATFe = np.array([ATF1,ATF2e,ATF3e,ATF4])
        ARFe = np.array([ARF1,ARF2e,ARF3e,ARF4])

        QinEe = C2e*QinE + S2e*UinE
        UinEe = C2e*UinE - S2e*QinE

        # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
        IinF = IinE
        QinF = aCal*QinE
        UinF = -aCal*UinE
        VinF = (1.-2.*aCal)*VinE

        IinFo = IinE
        QinFo = 0.
        UinFo = 0.
        VinFo = VinE

        IinFa = 0.
        QinFa = QinE
        UinFa = -UinE
        VinFa = -2.*VinE

        # Stokes Input Vector before receiver optics rotated by epsilon Eq. C.3
        QinFe = C2e*QinF + S2e*UinF
        UinFe = C2e*UinF - S2e*QinF
        QinFoe = C2e*QinFo + S2e*UinFo
        UinFoe = C2e*UinFo - S2e*QinFo
        QinFae = C2e*QinFa + S2e*UinFa
        UinFae = C2e*UinFa - S2e*QinFa

        # Calibration signals and Calibration correction K from measurements with LDRCal / aCal
        if (TypeC == 2) or (TypeC == 1):   # rotator calibration Eq. C.4
            # parameters for calibration with aCal
            AT = ATF1*IinF + ATF4*h*VinF
            BT = ATF3e*QinF - ATF2e*h*UinF
            AR = ARF1*IinF + ARF4*h*VinF
            BR = ARF3e*QinF - ARF2e*h*UinF
            # Correction paremeters for normal measurements; they are independent of LDR
            if (not RotationErrorEpsilonForNormalMeasurements):
                GT = ATF1*IinE + ATF4*VinE
                GR = ARF1*IinE + ARF4*VinE
                HT = ATF2*QinE - ATF3*UinE - ATF4*2*VinE
                HR = ARF2*QinE - ARF3*UinE - ARF4*2*VinE
            else:
                GT = ATF1*IinE + ATF4*h*VinE
                GR = ARF1*IinE + ARF4*h*VinE
                HT = ATF2e*QinE - ATF3e*h*UinE - ATF4*h*2*VinE
                HR = ARF2e*QinE - ARF3e*h*UinE - ARF4*h*2*VinE
        elif (TypeC == 3) or (TypeC == 4):  # linear polariser calibration Eq. C.5
            # p = +45°, m = -45°
            IF1e = np.array([IinF, ZiC*CosC*QinFe, UinFe, ZiC*CosC*VinF])
            IF2e = np.array([DiC*UinFe, -ZiC*SinC*VinF, DiC*IinF, ZiC*SinC*QinFe])
            AT = np.dot(ATFe,IF1e)
            AR = np.dot(ARFe,IF1e)
            BT = np.dot(ATFe,IF2e)
            BR = np.dot(ARFe,IF2e)

            # Correction paremeters for normal measurements; they are independent of LDR  --- the same as for TypeC = 6
            if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                IS1 = np.array([IinE,0,0,VinE])
                IS2 = np.array([0,QinE,-UinE,-2*VinE])
                GT = np.dot(ATF,IS1)
                GR = np.dot(ARF,IS1)
                HT = np.dot(ATF,IS2)
                HR = np.dot(ARF,IS2)
            else:
                IS1e = np.array([IinFo+DiC*QinFoe,DiC*IinFo+QinFoe,ZiC*(CosC*UinFoe+SinC*VinFo),-ZiC*(SinC*UinFoe-CosC*VinFo)])
                IS2e = np.array([IinFa+DiC*QinFae,DiC*IinFa+QinFae,ZiC*(CosC*UinFae+SinC*VinFa),-ZiC*(SinC*UinFae-CosC*VinFa)])
                GT = np.dot(ATFe,IS1e)
                GR = np.dot(ARFe,IS1e)
                HT = np.dot(ATFe,IS2e)
                HR = np.dot(ARFe,IS2e)

        elif (TypeC == 6):  # diattenuator calibration +-22.5° rotated_diattenuator_X22x5deg.odt
            # parameters for calibration with aCal
            IF1e = np.array([IinF+sqr05*DiC*QinFe, sqr05*DiC*IinF+(1-0.5*WiC)*QinFe, (1-0.5*WiC)*UinFe+sqr05*ZiC*SinC*VinF, -sqr05*ZiC*SinC*UinFe+ZiC*CosC*VinF])
            IF2e = np.array([sqr05*DiC*UinFe, 0.5*WiC*UinFe-sqr05*ZiC*SinC*VinF, sqr05*DiC*IinF+0.5*WiC*QinFe, sqr05*ZiC*SinC*QinFe])
            AT = np.dot(ATFe,IF1e)
            AR = np.dot(ARFe,IF1e)
            BT = np.dot(ATFe,IF2e)
            BR = np.dot(ARFe,IF2e)

            # Correction paremeters for normal measurements; they are independent of LDR
            if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                #IS1 = np.array([IinE,0,0,VinE])
                #IS2 = np.array([0,QinE,-UinE,-2*VinE])
                IS1 = np.array([IinFo,0,0,VinFo])
                IS2 = np.array([0,QinFa,UinFa,VinFa])
                GT = np.dot(ATF,IS1)
                GR = np.dot(ARF,IS1)
                HT = np.dot(ATF,IS2)
                HR = np.dot(ARF,IS2)
            else:
                IS1e = np.array([IinFo+DiC*QinFoe,DiC*IinFo+QinFoe,ZiC*(CosC*UinFoe+SinC*VinFo),-ZiC*(SinC*UinFoe-CosC*VinFo)])
                IS2e = np.array([IinFa+DiC*QinFae,DiC*IinFa+QinFae,ZiC*(CosC*UinFae+SinC*VinFa),-ZiC*(SinC*UinFae-CosC*VinFa)])
                #IS1e = np.array([IinFo,0,0,VinFo])
                #IS2e = np.array([0,QinFae,UinFae,VinFa])
                GT = np.dot(ATFe,IS1e)
                GR = np.dot(ARFe,IS1e)
                HT = np.dot(ATFe,IS2e)
                HR = np.dot(ARFe,IS2e)

        else:
            print('Calibrator not implemented yet')
            sys.exit()

    elif LocC == 2:  # C behind emitter optics Eq.57 -------------------------------------------------------
        #print("Calibrator location not implemented yet")
        S2e = np.sin(np.deg2rad(2*RotC))
        C2e = np.cos(np.deg2rad(2*RotC))

        # AS with C before the receiver optics (see document rotated_diattenuator_X22x5deg.odt)
        AF1 = np.array([1,C2g*DiO,S2g*DiO,0])
        AF2 = np.array([C2g*DiO,1-S2g**2*WiO,S2g*C2g*WiO,-S2g*ZiO*SinO])
        AF3 = np.array([S2g*DiO, S2g*C2g*WiO, 1-C2g**2*WiO, C2g*ZiO*SinO])
        AF4 = np.array([0, S2g*SinO, -C2g*SinO, CosO])

        ATF = (ATP1*AF1+ATP2*AF2+ATP3*AF3+ATP4*AF4)
        ARF = (ARP1*AF1+ARP2*AF2+ARP3*AF3+ARP4*AF4)
        ATF1 = ATF[0]
        ATF2 = ATF[1]
        ATF3 = ATF[2]
        ATF4 = ATF[3]
        ARF1 = ARF[0]
        ARF2 = ARF[1]
        ARF3 = ARF[2]
        ARF4 = ARF[3]

        # AS with C behind the emitter
        # terms without aCal
        ATE1o, ARE1o = ATF1, ARF1
        ATE2o, ARE2o = 0., 0.
        ATE3o, ARE3o = 0., 0.
        ATE4o, ARE4o = ATF4, ARF4
        # terms with aCal
        ATE1a, ARE1a = 0. , 0.
        ATE2a, ARE2a = ATF2, ARF2
        ATE3a, ARE3a = -ATF3, -ARF3
        ATE4a, ARE4a = -2*ATF4, -2*ARF4
        # rotated AinEa by epsilon
        ATE2ae =  C2e*ATF2 + S2e*ATF3
        ATE3ae = -S2e*ATF2 - C2e*ATF3
        ARE2ae =  C2e*ARF2 + S2e*ARF3
        ARE3ae = -S2e*ARF2 - C2e*ARF3

        ATE1 = ATE1o
        ATE2e = aCal*ATE2ae
        ATE3e = aCal*ATE3ae
        ATE4 = (1-2*aCal)*ATF4
        ARE1 = ARE1o
        ARE2e = aCal*ARE2ae
        ARE3e = aCal*ARE3ae
        ARE4 = (1-2*aCal)*ARF4

        # rotated IinE
        QinEe = C2e*QinE + S2e*UinE
        UinEe = C2e*UinE - S2e*QinE

        # Calibration signals and Calibration correction K from measurements with LDRCal / aCal
        if (TypeC == 2) or (TypeC == 1):   #  +++++++++ rotator calibration Eq. C.4
            AT = ATE1o*IinE + (ATE4o+aCal*ATE4a)*h*VinE
            BT = aCal * (ATE3ae*QinEe - ATE2ae*h*UinEe)
            AR = ARE1o*IinE + (ARE4o+aCal*ARE4a)*h*VinE
            BR = aCal * (ARE3ae*QinEe - ARE2ae*h*UinEe)

            # Correction paremeters for normal measurements; they are independent of LDR
            if (not RotationErrorEpsilonForNormalMeasurements):
                # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
                GT = ATE1o*IinE + ATE4o*h*VinE
                GR = ARE1o*IinE + ARE4o*h*VinE
                HT = ATE2a*QinE + ATE3a*h*UinEe + ATE4a*h*VinE
                HR = ARE2a*QinE + ARE3a*h*UinEe + ARE4a*h*VinE
            else:
                GT = ATE1o*IinE + ATE4o*h*VinE
                GR = ARE1o*IinE + ARE4o*h*VinE
                HT = ATE2ae*QinE + ATE3ae*h*UinEe + ATE4a*h*VinE
                HR = ARE2ae*QinE + ARE3ae*h*UinEe + ARE4a*h*VinE

        elif (TypeC == 3) or (TypeC == 4):  # +++++++++ linear polariser calibration Eq. C.5
            # p = +45°, m = -45°
            AT = ATE1*IinE + ZiC*CosC*(ATE2e*QinEe + ATE4*VinE) + ATE3e*UinEe
            BT = DiC*(ATE1*UinEe + ATE3e*IinE) + ZiC*SinC*(ATE4*QinEe - ATE2e*VinE)
            AR = ARE1*IinE + ZiC*CosC*(ARE2e*QinEe + ARE4*VinE) + ARE3e*UinEe
            BR = DiC*(ARE1*UinEe + ARE3e*IinE) + ZiC*SinC*(ARE4*QinEe - ARE2e*VinE)

            # Correction paremeters for normal measurements; they are independent of LDR
            if (not RotationErrorEpsilonForNormalMeasurements):
                # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
                GT = ATE1o*IinE + ATE4o*VinE
                GR = ARE1o*IinE + ARE4o*VinE
                HT = ATE2a*QinE + ATE3a*UinE + ATE4a*VinE
                HR = ARE2a*QinE + ARE3a*UinE + ARE4a*VinE
            else:
                D = IinE + DiC*QinEe
                A = DiC*IinE + QinEe
                B = ZiC*(CosC*UinEe + SinC*VinE)
                C = -ZiC*(SinC*UinEe - CosC*VinE)
                GT = ATE1o*D + ATE4o*C
                GR = ARE1o*D + ARE4o*C
                HT = ATE2a*A + ATE3a*B + ATE4a*C
                HR = ARE2a*A + ARE3a*B + ARE4a*C

        elif (TypeC == 6):  # real HWP calibration +-22.5° rotated_diattenuator_X22x5deg.odt
            # p = +22.5°, m = -22.5°
            IE1e = np.array([IinE+sqr05*DiC*QinEe, sqr05*DiC*IinE+(1-0.5*WiC)*QinEe, (1-0.5*WiC)*UinEe+sqr05*ZiC*SinC*VinE, -sqr05*ZiC*SinC*UinEe+ZiC*CosC*VinE])
            IE2e = np.array([sqr05*DiC*UinEe, 0.5*WiC*UinEe-sqr05*ZiC*SinC*VinE, sqr05*DiC*IinE+0.5*WiC*QinEe, sqr05*ZiC*SinC*QinEe])
            ATEe = np.array([ATE1,ATE2e,ATE3e,ATE4])
            AREe = np.array([ARE1,ARE2e,ARE3e,ARE4])
            AT = np.dot(ATEe,IE1e)
            AR = np.dot(AREe,IE1e)
            BT = np.dot(ATEe,IE2e)
            BR = np.dot(AREe,IE2e)

            # Correction paremeters for normal measurements; they are independent of LDR
            if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                GT = ATE1o*IinE + ATE4o*VinE
                GR = ARE1o*IinE + ARE4o*VinE
                HT = ATE2a*QinE + ATE3a*UinE + ATE4a*VinE
                HR = ARE2a*QinE + ARE3a*UinE + ARE4a*VinE
            else:
                D = IinE + DiC*QinEe
                A = DiC*IinE + QinEe
                B = ZiC*(CosC*UinEe + SinC*VinE)
                C = -ZiC*(SinC*UinEe - CosC*VinE)
                GT = ATE1o*D + ATE4o*C
                GR = ARE1o*D + ARE4o*C
                HT = ATE2a*A + ATE3a*B + ATE4a*C
                HR = ARE2a*A + ARE3a*B + ARE4a*C

        else:
            print('Calibrator not implemented yet')
            sys.exit()

    else:
        print("Calibrator location not implemented yet")
        sys.exit()

    # Determination of the correction K of the calibration factor
    IoutTp = TaT*TiT*TiO*TiE*(AT + BT)
    IoutTm = TaT*TiT*TiO*TiE*(AT - BT)
    IoutRp = TaR*TiR*TiO*TiE*(AR + BR)
    IoutRm = TaR*TiR*TiO*TiE*(AR - BR)

    # --- Results and Corrections; electronic etaR and etaT are assumed to be 1
    Etapx = IoutRp/IoutTp
    Etamx = IoutRm/IoutTm
    Etax = (Etapx*Etamx)**0.5

    Eta = (TaR*TiR)/(TaT*TiT)   # Eta = Eta*/K  Eq. 84
    K = Etax / Eta

    #  For comparison with Volkers Libreoffice Müller Matrix spreadsheet
    #Eta_test_p = (IoutRp/IoutTp)
    #Eta_test_m = (IoutRm/IoutTm)
    #Eta_test = (Eta_test_p*Eta_test_m)**0.5

    # ----- Forward simulated signals and LDRsim with atrue; from input file
    It = TaT*TiT*TiO*TiE*(GT+atrue*HT)
    Ir = TaR*TiR*TiO*TiE*(GR+atrue*HR)
    # LDRsim = 1/Eta*Ir/It  # simulated LDR* with Y from input file
    LDRsim = Ir/It  # simulated uncorrected LDR with Y from input file
    # Corrected LDRsimCorr from forward simulated LDRsim (atrue)
    # LDRsimCorr = (1./Eta*LDRsim*(GT+HT)-(GR+HR))/((GR-HR)-1./Eta*LDRsim*(GT-HT))
    if Y == -1.:
        LDRsimx = 1./LDRsim
    else:
        LDRsimx = LDRsim

    # The following is correct without doubt
    #LDRCorr = (LDRsim*K/Etax*(GT+HT)-(GR+HR))/((GR-HR)-LDRsim*K/Etax*(GT-HT))

    # The following is a test whether the equations for calibration Etax and normal  signal (GHK, LDRsim) are consistent
    LDRCorr = (LDRsim/Eta*(GT+HT)-(GR+HR))/((GR-HR)-LDRsim*K/Etax*(GT-HT))

    TTa = TiT*TaT #*ATP1
    TRa = TiR*TaR #*ARP1

    F11sim = 1/(TiO*TiE)*((HR*Etax/K*It/TTa-HT*Ir/TRa)/(HR*GT-HT*GR))    # IL = 1, Etat = Etar = 1

    return (GT, HT, GR, HR, K, Eta, LDRsimx, LDRCorr, DTa, DRa, TTa, TRa, F11sim)
# *******************************************************************************************************************************

# --- CALC truth
GT0, HT0, GR0, HR0, K0, Eta0, LDRsimx, LDRCorr, DTa0, DRa0, TTa0, TRa0, F11sim0 = Calc(RotL0, RotE0, RetE0, DiE0, RotO0, RetO0, DiO0, RotC0, RetC0, DiC0, TP0, TS0, RP0, RS0, ERaT0, RotaT0, RetT0, ERaR0, RotaR0, RetR0, LDRCal0)

# --------------------------------------------------------
with open('output_' + LID + '.dat', 'w') as f:
    with redirect_stdout(f):
        print("From ", dname)
        print("Running ", fname)
        print("Reading input file ", InputFile) #, "  for Lidar system :", EID, ", ", LID)
        print("for Lidar system: ", EID, ", ", LID)
        # --- Print iput information*********************************
        print(" --- Input parameters: value ±error / ±steps  ----------------------")
        print("{0:8} {1:8} {2:8.5f}; {3:8} {4:7.4f}±{5:7.4f}/{6:2d}".format("Laser: ", "DOLP = ", bL, "        rotation alpha = ", RotL0, dRotL, nRotL))
        print("              Diatt.,             Tunpol,   Retard.,   Rotation (deg)")
        print("{0:12} {1:7.4f}±{2:7.4f}/{8:2d}, {3:7.4f}, {4:3.0f}±{5:3.0f}/{9:2d}, {6:7.4f}±{7:7.4f}/{10:2d}".format("Emitter    ", DiE0, dDiE, TiE, RetE0, dRetE, RotE0, dRotE, nDiE, nRetE, nRotE))
        print("{0:12} {1:7.4f}±{2:7.4f}/{8:2d}, {3:7.4f}, {4:3.0f}±{5:3.0f}/{9:2d}, {6:7.4f}±{7:7.4f}/{10:2d}".format("Receiver   ", DiO0, dDiO, TiO, RetO0, dRetO, RotO0, dRotO, nDiO, nRetO, nRotO))
        print("{0:12} {1:7.4f}±{2:7.4f}/{8:2d}, {3:7.4f}, {4:3.0f}±{5:3.0f}/{9:2d}, {6:7.4f}±{7:7.4f}/{10:2d}".format("Calibrator ", DiC0, dDiC, TiC, RetC0, dRetC, RotC0, dRotC, nDiC, nRetC, nRotC))
        print("{0:12}".format(" --- Pol.-filter ---"))
        print("{0:12}{1:7.4f}±{2:7.4f}/{3:2d}, {4:7.4f}±{5:7.4f}/{6:2d}".format("ERT,     ERR    :", ERaT0, dERaT, nERaT, ERaR0, dERaR, nERaR))
        print("{0:12}{1:7.4f}±{2:7.4f}/{3:2d}, {4:7.4f}±{5:7.4f}/{6:2d}".format("RotaT  , RotaR  :", RotaT0, dRotaT, nRotaT, RotaR0,dRotaR,nRotaR))
        print("{0:12}".format(" --- PBS ---"))
        print("{0:12}{1:7.4f}±{2:7.4f}/{9:2d}, {3:7.4f}±{4:7.4f}/{10:2d}, {5:7.4f}±{6:7.4f}/{11:2d},{7:7.4f}±{8:7.4f}/{12:2d}".format("TP,TS,RP,RS     :", TP0, dTP, TS0, dTS, RP0, dRP, RS0, dRS, nTP, nTS, nRP, nRS))
        print("{0:12}{1:7.4f},{2:7.4f}, {3:7.4f},{4:7.4f}, {5:1.0f}".format("DT,TT,DR,TR,Y   :", DiT, TiT, DiR, TiR, Y))
        print("{0:12}".format(" --- Combined PBS + Pol.-filter ---"))
        print("{0:12}{1:7.4f},{2:7.4f}, {3:7.4f},{4:7.4f}".format("DTa,TTa,DRa,TRa: ", DTa0, TTa0, DRa0, TRa0))
        print()
        print("Rotation Error Epsilon For Normal Measurements = ", RotationErrorEpsilonForNormalMeasurements)
        #print ('LocC = ', LocC, Loc[LocC], '; TypeC = ',TypeC, Type[TypeC])
        print(Type[TypeC], Loc[LocC], "; Parallel signal detected in", dY[int(Y+1)])
        #  end of print actual system parameters
        # ******************************************************************************

        #print()
        #print(" --- LDRCal during calibration | simulated and corrected LDRs -------------")
        #print("{0:8} |{1:8}->{2:8},{3:9}->{4:9} |{5:8}->{6:8}".format(" LDRCal"," LDRtrue", " LDRsim"," LDRtrue2", " LDRsim2", " LDRmeas", " LDRcorr"))
        #print("{0:8.5f} |{1:8.5f}->{2:8.5f},{3:9.5f}->{4:9.5f} |{5:8.5f}->{6:8.5f}".format(LDRCal, LDRtrue, LDRsim, LDRtrue2, LDRsim2, LDRmeas, LDRCorr))
        #print("{0:8}       |{1:8}->{2:8}->{3:8}".format(" LDRCal"," LDRtrue", " LDRsimx", " LDRcorr"))
        #print("{0:6.3f}±{1:5.3f}/{2:2d}|{3:8.5f}->{4:8.5f}->{5:8.5f}".format(LDRCal0, dLDRCal, nLDRCal, LDRtrue, LDRsimx, LDRCorr))
        #print("{0:8}       |{1:8}->{2:8}->{3:8}".format(" LDRCal"," LDRtrue", " LDRsimx", " LDRcorr"))
        #print(" --- LDRCal during calibration")
        print("{0:26}: {1:6.3f}±{2:5.3f}/{3:2d}".format("LDRCal during calibration", LDRCal0, dLDRCal, nLDRCal))

        #print("{0:8}={1:8.5f};{2:8}={3:8.5f}".format(" IinP",IinP," F11sim",F11sim))
        print()

        K0List = np.zeros(3)
        LDRsimxList = np.zeros(3)
        LDRCalList = 0.004, 0.2, 0.45
        for i,LDRCal in enumerate(LDRCalList):
            GT0, HT0, GR0, HR0, K0, Eta0, LDRsimx, LDRCorr, DTa0, DRa0, TTa0, TRa0, F11sim0 = Calc(RotL0, RotE0, RetE0, DiE0, RotO0, RetO0, DiO0, RotC0, RetC0, DiC0, TP0, TS0, RP0, RS0, ERaT0, RotaT0, RetT0, ERaR0, RotaR0, RetR0, LDRCal)
            K0List[i] = K0
            LDRsimxList[i] = LDRsimx

        print("{0:8},{1:8},{2:8},{3:8},{4:9},{5:9},{6:9}".format(" GR", " GT", " HR", " HT", " K(0.004)", " K(0.2)", " K(0.45)"))
        print("{0:8.5f},{1:8.5f},{2:8.5f},{3:8.5f},{4:9.5f},{5:9.5f},{6:9.5f}".format(GR0, GT0, HR0, HT0, K0List[0], K0List[1], K0List[2]))
        print('========================================================================')

        print("{0:9},{1:9},{2:9}".format("  LDRtrue", "  LDRsimx", "  LDRCorr"))
        LDRtrueList = 0.004, 0.02, 0.2, 0.45
        for i,LDRtrue in enumerate(LDRtrueList):
            GT0, HT0, GR0, HR0, K0, Eta0, LDRsimx, LDRCorr, DTa0, DRa0, TTa0, TRa0, F11sim0 = Calc(RotL0, RotE0, RetE0, DiE0, RotO0, RetO0, DiO0, RotC0, RetC0, DiC0, TP0, TS0, RP0, RS0, ERaT0, RotaT0, RetT0, ERaR0, RotaR0, RetR0, LDRCal0)
            print("{0:9.5f},{1:9.5f},{2:9.5f}".format(LDRtrue, LDRsimx, LDRCorr))


file = open('output_' + LID + '.dat', 'r')
print (file.read())
file.close()

'''
if(PrintToOutputFile):
    f = open('output_ver7.dat', 'w')
    old_target = sys.stdout
    sys.stdout = f

    print("something")

if(PrintToOutputFile):
    sys.stdout.flush()
    f.close
    sys.stdout = old_target
'''
# --- CALC again truth with LDRCal0 to reset all 0-values
GT0, HT0, GR0, HR0, K0, Eta0, LDRsimx, LDRCorr, DTa0, DRa0, TTa0, TRa0, F11sim0 = Calc(RotL0, RotE0, RetE0, DiE0, RotO0, RetO0, DiO0, RotC0, RetC0, DiC0, TP0, TS0, RP0, RS0, ERaT0, RotaT0, RetT0, ERaR0, RotaR0, RetR0, LDRCal0)

# --- Start Errors calculation

iN = -1
N = ((nRotL*2+1)*
    (nRotE*2+1)*(nRetE*2+1)*(nDiE*2+1)*
    (nRotO*2+1)*(nRetO*2+1)*(nDiO*2+1)*
    (nRotC*2+1)*(nRetC*2+1)*(nDiC*2+1)*
    (nTP*2+1)*(nTS*2+1)*(nRP*2+1)*(nRS*2+1)*(nERaT*2+1)*(nERaR*2+1)*
    (nRotaT*2+1)*(nRotaR*2+1)*(nRetT*2+1)*(nRetR*2+1)*(nLDRCal*2+1))
print("N = ",N ," ", end="")

if N > 1e6:
    if user_yes_no_query('Warning: processing ' + str(N) + ' samples will take very long. Do you want to proceed?') == 0: sys.exit()
if N > 5e6:
    if user_yes_no_query('Warning: the memory required for ' + str(N) + ' samples might be ' + '{0:5.1f}'.format(N/4e6) + ' GB. Do you anyway want to proceed?') == 0: sys.exit()

#if user_yes_no_query('Warning: processing' + str(N) + ' samples will take very long. Do you want to proceed?') == 0: sys.exit()

# --- Arrays for plotting ------
LDRmin = np.zeros(5)
LDRmax = np.zeros(5)
F11min = np.zeros(5)
F11max = np.zeros(5)

LDRrange = np.zeros(5)
LDRrange = 0.004, 0.02, 0.1, 0.3, 0.45
#aLDRsimx = np.zeros(N)
#aLDRsimx2 = np.zeros(N)
#aLDRcorr = np.zeros(N)
#aLDRcorr2 = np.zeros(N)
aERaT = np.zeros(N)
aERaR = np.zeros(N)
aRotaT = np.zeros(N)
aRotaR = np.zeros(N)
aRetT = np.zeros(N)
aRetR = np.zeros(N)
aTP = np.zeros(N)
aTS = np.zeros(N)
aRP = np.zeros(N)
aRS = np.zeros(N)
aDiE = np.zeros(N)
aDiO = np.zeros(N)
aDiC = np.zeros(N)
aRotC = np.zeros(N)
aRetC = np.zeros(N)
aRotL = np.zeros(N)
aRetE = np.zeros(N)
aRotE = np.zeros(N)
aRetO = np.zeros(N)
aRotO = np.zeros(N)
aLDRCal = np.zeros(N)
aA = np.zeros((5,N))
aX = np.zeros((5,N))
aF11corr = np.zeros((5,N))

atime = clock()
dtime = clock()

# --- Calc Error signals
#GT, HT, GR, HR, K, Eta, LDRsim = Calc(RotL, RotE, RetE, DiE, RotO, RetO, DiO, RotC, RetC, DiC, TP, TS)
# ---- Do the calculations of bra-ket vectors
h = -1. if TypeC == 2 else 1

# from input file: measured LDRm and true LDRtrue, LDRtrue2  =>
ameas = (1.-LDRmeas)/(1+LDRmeas)
atrue = (1.-LDRtrue)/(1+LDRtrue)
atrue2 = (1.-LDRtrue2)/(1+LDRtrue2)

for iLDRCal in range(-nLDRCal,nLDRCal+1):
    # from input file:  assumed LDRCal for calibration measurements
    LDRCal = LDRCal0
    if nLDRCal > 0: LDRCal = LDRCal0 + iLDRCal*dLDRCal/nLDRCal

    GT0, HT0, GR0, HR0, K0, Eta0, LDRsimx, LDRCorr, DTa0, DRa0, TTa0, TRa0, F11sim0 = Calc(RotL0, RotE0, RetE0, DiE0, RotO0, RetO0, DiO0, RotC0, RetC0, DiC0, TP0, TS0, RP0, RS0, ERaT0, RotaT0, RetT0, ERaR0, RotaR0, RetR0, LDRCal)
    aCal = (1.-LDRCal)/(1+LDRCal)
    for iRotL, iRotE, iRetE, iDiE \
        in [(iRotL,iRotE,iRetE,iDiE)
        for iRotL in range(-nRotL,nRotL+1)
        for iRotE in range(-nRotE,nRotE+1)
        for iRetE in range(-nRetE,nRetE+1)
        for iDiE in range(-nDiE,nDiE+1)]:

        if nRotL > 0: RotL = RotL0 + iRotL*dRotL/nRotL
        if nRotE > 0: RotE = RotE0 + iRotE*dRotE/nRotE
        if nRetE > 0: RetE = RetE0 + iRetE*dRetE/nRetE
        if nDiE > 0:  DiE  = DiE0  + iDiE*dDiE/nDiE

        # angles of emitter and laser and calibrator and receiver optics
        # RotL = alpha, RotE = beta, RotO = gamma, RotC = epsilon
        S2a = np.sin(2*np.deg2rad(RotL))
        C2a = np.cos(2*np.deg2rad(RotL))
        S2b = np.sin(2*np.deg2rad(RotE))
        C2b = np.cos(2*np.deg2rad(RotE))
        S2ab = np.sin(np.deg2rad(2*RotL-2*RotE))
        C2ab = np.cos(np.deg2rad(2*RotL-2*RotE))

        # Laser with Degree of linear polarization DOLP = bL
        IinL = 1.
        QinL = bL
        UinL = 0.
        VinL = (1. - bL**2)**0.5

        # Stokes Input Vector rotation Eq. E.4
        A = C2a*QinL - S2a*UinL
        B = S2a*QinL + C2a*UinL
        # Stokes Input Vector rotation Eq. E.9
        C = C2ab*QinL - S2ab*UinL
        D = S2ab*QinL + C2ab*UinL

        # emitter optics
        CosE = np.cos(np.deg2rad(RetE))
        SinE = np.sin(np.deg2rad(RetE))
        ZiE = (1. - DiE**2)**0.5
        WiE = (1. - ZiE*CosE)

        # Stokes Input Vector after emitter optics equivalent to Eq. E.9 with already rotated input vector from Eq. E.4
        # b = beta
        IinE = (IinL + DiE*C)
        QinE = (C2b*DiE*IinL + A + S2b*(WiE*D - ZiE*SinE*VinL))
        UinE = (S2b*DiE*IinL + B - C2b*(WiE*D - ZiE*SinE*VinL))
        VinE = (-ZiE*SinE*D + ZiE*CosE*VinL)

        #-------------------------
        # F11 assuemd to be = 1  => measured: F11m = IinP / IinE with atrue
        #F11sim = (IinE + DiO*atrue*(C2g*QinE - S2g*UinE))/IinE
        #-------------------------

        for iRotO, iRetO, iDiO, iRotC, iRetC, iDiC, iTP, iTS, iRP, iRS, iERaT, iRotaT, iRetT, iERaR, iRotaR, iRetR \
            in [(iRotO,iRetO,iDiO,iRotC,iRetC,iDiC,iTP,iTS,iRP,iRS,iERaT,iRotaT,iRetT,iERaR,iRotaR,iRetR )
            for iRotO in range(-nRotO,nRotO+1)
            for iRetO in range(-nRetO,nRetO+1)
            for iDiO in range(-nDiO,nDiO+1)
            for iRotC in range(-nRotC,nRotC+1)
            for iRetC in range(-nRetC,nRetC+1)
            for iDiC in range(-nDiC,nDiC+1)
            for iTP in range(-nTP,nTP+1)
            for iTS in range(-nTS,nTS+1)
            for iRP in range(-nRP,nRP+1)
            for iRS in range(-nRS,nRS+1)
            for iERaT in range(-nERaT,nERaT+1)
            for iRotaT in range(-nRotaT,nRotaT+1)
            for iRetT in range(-nRetT,nRetT+1)
            for iERaR in range(-nERaR,nERaR+1)
            for iRotaR in range(-nRotaR,nRotaR+1)
            for iRetR in range(-nRetR,nRetR+1)]:

            iN = iN + 1
            if (iN == 10001):
                ctime = clock()
                print(" estimated time ", "{0:4.2f}".format(N/10000 * (ctime-atime)), "sec ") #, end="")
                print("\r elapsed time ", "{0:5.0f}".format((ctime-atime)), "sec ", end="\r")
            ctime = clock()
            if ((ctime - dtime) > 10):
                print("\r elapsed time ", "{0:5.0f}".format((ctime-atime)), "sec ", end="\r")
                dtime = ctime

            if nRotO > 0: RotO = RotO0 + iRotO*dRotO/nRotO
            if nRetO > 0: RetO = RetO0 + iRetO*dRetO/nRetO
            if nDiO > 0:  DiO  = DiO0  + iDiO*dDiO/nDiO
            if nRotC > 0: RotC = RotC0 + iRotC*dRotC/nRotC
            if nRetC > 0: RetC = RetC0 + iRetC*dRetC/nRetC
            if nDiC > 0:  DiC  = DiC0  + iDiC*dDiC/nDiC
            if nTP > 0:   TP   = TP0   + iTP*dTP/nTP
            if nTS > 0:   TS   = TS0   + iTS*dTS/nTS
            if nRP > 0:   RP   = RP0   + iRP*dRP/nRP
            if nRS > 0:   RS   = RS0   + iRS*dRS/nRS
            if nERaT > 0: ERaT = ERaT0 + iERaT*dERaT/nERaT
            if nRotaT > 0:RotaT= RotaT0+ iRotaT*dRotaT/nRotaT
            if nRetT > 0: RetT = RetT0 + iRetT*dRetT/nRetT
            if nERaR > 0: ERaR = ERaR0 + iERaR*dERaR/nERaR
            if nRotaR > 0:RotaR= RotaR0+ iRotaR*dRotaR/nRotaR
            if nRetR > 0: RetR = RetR0 + iRetR*dRetR/nRetR

            #print("{0:5.2f}, {1:5.2f}, {2:5.2f}, {3:10d}".format(RotL, RotE, RotO, iN))

            # receiver optics
            CosO = np.cos(np.deg2rad(RetO))
            SinO = np.sin(np.deg2rad(RetO))
            ZiO = (1. - DiO**2)**0.5
            WiO = (1. - ZiO*CosO)
            S2g = np.sin(np.deg2rad(2*RotO))
            C2g = np.cos(np.deg2rad(2*RotO))
            # calibrator
            CosC = np.cos(np.deg2rad(RetC))
            SinC = np.sin(np.deg2rad(RetC))
            ZiC = (1. - DiC**2)**0.5
            WiC = (1. - ZiC*CosC)

            #For POLLY_XT
            # analyser
            #RS = 1 - TS
            #RP = 1 - TP
            TiT = 0.5 * (TP + TS)
            DiT = (TP-TS)/(TP+TS)
            ZiT = (1. - DiT**2)**0.5
            TiR = 0.5 * (RP + RS)
            DiR = (RP-RS)/(RP+RS)
            ZiR = (1. - DiR**2)**0.5
            CosT = np.cos(np.deg2rad(RetT))
            SinT = np.sin(np.deg2rad(RetT))
            CosR = np.cos(np.deg2rad(RetR))
            SinR = np.sin(np.deg2rad(RetR))

            DaT = (1-ERaT)/(1+ERaT)
            DaR = (1-ERaR)/(1+ERaR)
            TaT = 0.5*(1+ERaT)
            TaR = 0.5*(1+ERaR)

            S2aT = np.sin(np.deg2rad(h*2*RotaT))
            C2aT = np.cos(np.deg2rad(2*RotaT))
            S2aR = np.sin(np.deg2rad(h*2*RotaR))
            C2aR = np.cos(np.deg2rad(2*RotaR))

            # Aanalyzer As before the PBS Eq. D.5
            ATP1 = (1+C2aT*DaT*DiT)
            ATP2 = Y*(DiT+C2aT*DaT)
            ATP3 = Y*S2aT*DaT*ZiT*CosT
            ATP4 = S2aT*DaT*ZiT*SinT
            ATP = np.array([ATP1,ATP2,ATP3,ATP4])

            ARP1 = (1+C2aR*DaR*DiR)
            ARP2 = Y*(DiR+C2aR*DaR)
            ARP3 = Y*S2aR*DaR*ZiR*CosR
            ARP4 = S2aR*DaR*ZiR*SinR
            ARP = np.array([ARP1,ARP2,ARP3,ARP4])

            TTa = TiT*TaT #*ATP1
            TRa = TiR*TaR #*ARP1

            # ---- Calculate signals and correction parameters for diffeent locations and calibrators
            if LocC == 4:  # Calibrator before the PBS
                #print("Calibrator location not implemented yet")

                #S2ge = np.sin(np.deg2rad(2*RotO + h*2*RotC))
                #C2ge = np.cos(np.deg2rad(2*RotO + h*2*RotC))
                S2e = np.sin(np.deg2rad(h*2*RotC))
                C2e = np.cos(np.deg2rad(2*RotC))
                # rotated AinP by epsilon Eq. C.3
                ATP2e = C2e*ATP2 + S2e*ATP3
                ATP3e = C2e*ATP3 - S2e*ATP2
                ARP2e = C2e*ARP2 + S2e*ARP3
                ARP3e = C2e*ARP3 - S2e*ARP2
                ATPe = np.array([ATP1,ATP2e,ATP3e,ATP4])
                ARPe = np.array([ARP1,ARP2e,ARP3e,ARP4])
                # Stokes Input Vector before the polarising beam splitter Eq. E.31
                A = C2g*QinE - S2g*UinE
                B = S2g*QinE + C2g*UinE
                #C = (WiO*aCal*B + ZiO*SinO*(1-2*aCal)*VinE)
                Co = ZiO*SinO*VinE
                Ca = (WiO*B - 2*ZiO*SinO*VinE)
                #C = Co + aCal*Ca
                #IinP = (IinE + DiO*aCal*A)
                #QinP = (C2g*DiO*IinE + aCal*QinE - S2g*C)
                #UinP = (S2g*DiO*IinE - aCal*UinE + C2g*C)
                #VinP = (ZiO*SinO*aCal*B + ZiO*CosO*(1-2*aCal)*VinE)
                IinPo = IinE
                QinPo = (C2g*DiO*IinE - S2g*Co)
                UinPo = (S2g*DiO*IinE + C2g*Co)
                VinPo = ZiO*CosO*VinE

                IinPa = DiO*A
                QinPa = QinE - S2g*Ca
                UinPa = -UinE + C2g*Ca
                VinPa = ZiO*(SinO*B - 2*CosO*VinE)

                IinP = IinPo + aCal*IinPa
                QinP = QinPo + aCal*QinPa
                UinP = UinPo + aCal*UinPa
                VinP = VinPo + aCal*VinPa
                # Stokes Input Vector before the polarising beam splitter rotated by epsilon Eq. C.3
                #QinPe = C2e*QinP + S2e*UinP
                #UinPe = C2e*UinP - S2e*QinP
                QinPoe = C2e*QinPo + S2e*UinPo
                UinPoe = C2e*UinPo - S2e*QinPo
                QinPae = C2e*QinPa + S2e*UinPa
                UinPae = C2e*UinPa - S2e*QinPa
                QinPe = C2e*QinP + S2e*UinP
                UinPe = C2e*UinP - S2e*QinP

                # Calibration signals and Calibration correction K from measurements with LDRCal / aCal
                if (TypeC == 2) or (TypeC == 1):  # rotator calibration Eq. C.4
                    # parameters for calibration with aCal
                    AT = ATP1*IinP + h*ATP4*VinP
                    BT = ATP3e*QinP - h*ATP2e*UinP
                    AR = ARP1*IinP + h*ARP4*VinP
                    BR = ARP3e*QinP - h*ARP2e*UinP
                    # Correction paremeters for normal measurements; they are independent of LDR
                    if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                        IS1 = np.array([IinPo,QinPo,UinPo,VinPo])
                        IS2 = np.array([IinPa,QinPa,UinPa,VinPa])
                        GT = np.dot(ATP,IS1)
                        GR = np.dot(ARP,IS1)
                        HT = np.dot(ATP,IS2)
                        HR = np.dot(ARP,IS2)
                    else:
                        IS1 = np.array([IinPo,QinPo,UinPo,VinPo])
                        IS2 = np.array([IinPa,QinPa,UinPa,VinPa])
                        GT = np.dot(ATPe,IS1)
                        GR = np.dot(ARPe,IS1)
                        HT = np.dot(ATPe,IS2)
                        HR = np.dot(ARPe,IS2)
                elif (TypeC == 3) or (TypeC == 4):  # linear polariser calibration Eq. C.5
                    # parameters for calibration with aCal
                    AT = ATP1*IinP + ATP3e*UinPe + ZiC*CosC*(ATP2e*QinPe + ATP4*VinP)
                    BT = DiC*(ATP1*UinPe + ATP3e*IinP) - ZiC*SinC*(ATP2e*VinP - ATP4*QinPe)
                    AR = ARP1*IinP + ARP3e*UinPe + ZiC*CosC*(ARP2e*QinPe + ARP4*VinP)
                    BR = DiC*(ARP1*UinPe + ARP3e*IinP) - ZiC*SinC*(ARP2e*VinP - ARP4*QinPe)
                    # Correction paremeters for normal measurements; they are independent of LDR
                    if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                        IS1 = np.array([IinPo,QinPo,UinPo,VinPo])
                        IS2 = np.array([IinPa,QinPa,UinPa,VinPa])
                        GT = np.dot(ATP,IS1)
                        GR = np.dot(ARP,IS1)
                        HT = np.dot(ATP,IS2)
                        HR = np.dot(ARP,IS2)
                    else:
                        IS1e = np.array([IinPo+DiC*QinPoe,DiC*IinPo+QinPoe,ZiC*(CosC*UinPoe+SinC*VinPo),-ZiC*(SinC*UinPoe-CosC*VinPo)])
                        IS2e = np.array([IinPa+DiC*QinPae,DiC*IinPa+QinPae,ZiC*(CosC*UinPae+SinC*VinPa),-ZiC*(SinC*UinPae-CosC*VinPa)])
                        GT = np.dot(ATPe,IS1e)
                        GR = np.dot(ARPe,IS1e)
                        HT = np.dot(ATPe,IS2e)
                        HR = np.dot(ARPe,IS2e)
                elif (TypeC == 6):  # diattenuator calibration +-22.5° rotated_diattenuator_X22x5deg.odt
                    # parameters for calibration with aCal
                    AT = ATP1*IinP + sqr05*DiC*(ATP1*QinPe + ATP2e*IinP) + (1-0.5*WiC)*(ATP2e*QinPe + ATP3e*UinPe) + ZiC*(sqr05*SinC*(ATP3e*VinP-ATP4*UinPe) + ATP4*CosC*VinP)
                    BT = sqr05*DiC*(ATP1*UinPe + ATP3e*IinP) + 0.5*WiC*(ATP2e*UinPe + ATP3e*QinPe) - sqr05*ZiC*SinC*(ATP2e*VinP - ATP4*QinPe)
                    AR = ARP1*IinP + sqr05*DiC*(ARP1*QinPe + ARP2e*IinP) + (1-0.5*WiC)*(ARP2e*QinPe + ARP3e*UinPe) + ZiC*(sqr05*SinC*(ARP3e*VinP-ARP4*UinPe) + ARP4*CosC*VinP)
                    BR = sqr05*DiC*(ARP1*UinPe + ARP3e*IinP) + 0.5*WiC*(ARP2e*UinPe + ARP3e*QinPe) - sqr05*ZiC*SinC*(ARP2e*VinP - ARP4*QinPe)
                    # Correction paremeters for normal measurements; they are independent of LDR
                    if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                        IS1 = np.array([IinPo,QinPo,UinPo,VinPo])
                        IS2 = np.array([IinPa,QinPa,UinPa,VinPa])
                        GT = np.dot(ATP,IS1)
                        GR = np.dot(ARP,IS1)
                        HT = np.dot(ATP,IS2)
                        HR = np.dot(ARP,IS2)
                    else:
                        IS1e = np.array([IinPo+DiC*QinPoe,DiC*IinPo+QinPoe,ZiC*(CosC*UinPoe+SinC*VinPo),-ZiC*(SinC*UinPoe-CosC*VinPo)])
                        IS2e = np.array([IinPa+DiC*QinPae,DiC*IinPa+QinPae,ZiC*(CosC*UinPae+SinC*VinPa),-ZiC*(SinC*UinPae-CosC*VinPa)])
                        GT = np.dot(ATPe,IS1e)
                        GR = np.dot(ARPe,IS1e)
                        HT = np.dot(ATPe,IS2e)
                        HR = np.dot(ARPe,IS2e)
                else:
                    print("Calibrator not implemented yet")
                    sys.exit()

            elif LocC == 3:  # C before receiver optics Eq.57

                #S2ge = np.sin(np.deg2rad(2*RotO - 2*RotC))
                #C2ge = np.cos(np.deg2rad(2*RotO - 2*RotC))
                S2e = np.sin(np.deg2rad(2*RotC))
                C2e = np.cos(np.deg2rad(2*RotC))

                # AS with C before the receiver optics (see document rotated_diattenuator_X22x5deg.odt)
                AF1 = np.array([1,C2g*DiO,S2g*DiO,0])
                AF2 = np.array([C2g*DiO,1-S2g**2*WiO,S2g*C2g*WiO,-S2g*ZiO*SinO])
                AF3 = np.array([S2g*DiO, S2g*C2g*WiO, 1-C2g**2*WiO, C2g*ZiO*SinO])
                AF4 = np.array([0, S2g*SinO, -C2g*SinO, CosO])

                ATF = (ATP1*AF1+ATP2*AF2+ATP3*AF3+ATP4*AF4)
                ARF = (ARP1*AF1+ARP2*AF2+ARP3*AF3+ARP4*AF4)
                ATF1 = ATF[0]
                ATF2 = ATF[1]
                ATF3 = ATF[2]
                ATF4 = ATF[3]
                ARF1 = ARF[0]
                ARF2 = ARF[1]
                ARF3 = ARF[2]
                ARF4 = ARF[3]

                # rotated AinF by epsilon
                ATF2e = C2e*ATF[1] + S2e*ATF[2]
                ATF3e = C2e*ATF[2] - S2e*ATF[1]
                ARF2e = C2e*ARF[1] + S2e*ARF[2]
                ARF3e = C2e*ARF[2] - S2e*ARF[1]

                ATFe = np.array([ATF1,ATF2e,ATF3e,ATF4])
                ARFe = np.array([ARF1,ARF2e,ARF3e,ARF4])

                QinEe = C2e*QinE + S2e*UinE
                UinEe = C2e*UinE - S2e*QinE

                # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
                IinF = IinE
                QinF = aCal*QinE
                UinF = -aCal*UinE
                VinF = (1.-2.*aCal)*VinE

                IinFo = IinE
                QinFo = 0.
                UinFo = 0.
                VinFo = VinE

                IinFa = 0.
                QinFa = QinE
                UinFa = -UinE
                VinFa = -2.*VinE

                # Stokes Input Vector before receiver optics rotated by epsilon Eq. C.3
                QinFe = C2e*QinF + S2e*UinF
                UinFe = C2e*UinF - S2e*QinF
                QinFoe = C2e*QinFo + S2e*UinFo
                UinFoe = C2e*UinFo - S2e*QinFo
                QinFae = C2e*QinFa + S2e*UinFa
                UinFae = C2e*UinFa - S2e*QinFa

                # Calibration signals and Calibration correction K from measurements with LDRCal / aCal
                if (TypeC == 2) or (TypeC == 1):   # rotator calibration Eq. C.4
                    AT = ATF1*IinF + ATF4*h*VinF
                    BT = ATF3e*QinF - ATF2e*h*UinF
                    AR = ARF1*IinF + ARF4*h*VinF
                    BR = ARF3e*QinF - ARF2e*h*UinF

                    # Correction paremeters for normal measurements; they are independent of LDR
                    if (not RotationErrorEpsilonForNormalMeasurements):
                        GT = ATF1*IinE + ATF4*VinE
                        GR = ARF1*IinE + ARF4*VinE
                        HT = ATF2*QinE - ATF3*UinE - ATF4*2*VinE
                        HR = ARF2*QinE - ARF3*UinE - ARF4*2*VinE
                    else:
                        GT = ATF1*IinE + ATF4*h*VinE
                        GR = ARF1*IinE + ARF4*h*VinE
                        HT = ATF2e*QinE - ATF3e*h*UinE - ATF4*h*2*VinE
                        HR = ARF2e*QinE - ARF3e*h*UinE - ARF4*h*2*VinE

                elif (TypeC == 3) or (TypeC == 4):  # linear polariser calibration Eq. C.5
                    # p = +45°, m = -45°
                    IF1e = np.array([IinF, ZiC*CosC*QinFe, UinFe, ZiC*CosC*VinF])
                    IF2e = np.array([DiC*UinFe, -ZiC*SinC*VinF, DiC*IinF, ZiC*SinC*QinFe])

                    AT = np.dot(ATFe,IF1e)
                    AR = np.dot(ARFe,IF1e)
                    BT = np.dot(ATFe,IF2e)
                    BR = np.dot(ARFe,IF2e)

                    # Correction paremeters for normal measurements; they are independent of LDR  --- the same as for TypeC = 6
                    if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                        IS1 = np.array([IinE,0,0,VinE])
                        IS2 = np.array([0,QinE,-UinE,-2*VinE])

                        GT = np.dot(ATF,IS1)
                        GR = np.dot(ARF,IS1)
                        HT = np.dot(ATF,IS2)
                        HR = np.dot(ARF,IS2)
                    else:
                        IS1e = np.array([IinFo+DiC*QinFoe,DiC*IinFo+QinFoe,ZiC*(CosC*UinFoe+SinC*VinFo),-ZiC*(SinC*UinFoe-CosC*VinFo)])
                        IS2e = np.array([IinFa+DiC*QinFae,DiC*IinFa+QinFae,ZiC*(CosC*UinFae+SinC*VinFa),-ZiC*(SinC*UinFae-CosC*VinFa)])
                        GT = np.dot(ATFe,IS1e)
                        GR = np.dot(ARFe,IS1e)
                        HT = np.dot(ATFe,IS2e)
                        HR = np.dot(ARFe,IS2e)

                elif (TypeC == 6):  # diattenuator calibration +-22.5° rotated_diattenuator_X22x5deg.odt
                    # p = +22.5°, m = -22.5°
                    IF1e = np.array([IinF+sqr05*DiC*QinFe, sqr05*DiC*IinF+(1-0.5*WiC)*QinFe, (1-0.5*WiC)*UinFe+sqr05*ZiC*SinC*VinF, -sqr05*ZiC*SinC*UinFe+ZiC*CosC*VinF])
                    IF2e = np.array([sqr05*DiC*UinFe, 0.5*WiC*UinFe-sqr05*ZiC*SinC*VinF, sqr05*DiC*IinF+0.5*WiC*QinFe, sqr05*ZiC*SinC*QinFe])

                    AT = np.dot(ATFe,IF1e)
                    AR = np.dot(ARFe,IF1e)
                    BT = np.dot(ATFe,IF2e)
                    BR = np.dot(ARFe,IF2e)

                    # Correction paremeters for normal measurements; they are independent of LDR
                    if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                        #IS1 = np.array([IinE,0,0,VinE])
                        #IS2 = np.array([0,QinE,-UinE,-2*VinE])
                        IS1 = np.array([IinFo,0,0,VinFo])
                        IS2 = np.array([0,QinFa,UinFa,VinFa])
                        GT = np.dot(ATF,IS1)
                        GR = np.dot(ARF,IS1)
                        HT = np.dot(ATF,IS2)
                        HR = np.dot(ARF,IS2)
                    else:
                        #IS1e = np.array([IinE,DiC*IinE,ZiC*SinC*VinE,ZiC*CosC*VinE])
                        #IS2e = np.array([DiC*QinEe,QinEe,-ZiC*(CosC*UinEe+2*SinC*VinE),ZiC*(SinC*UinEe-2*CosC*VinE)])
                        IS1e = np.array([IinFo+DiC*QinFoe,DiC*IinFo+QinFoe,ZiC*(CosC*UinFoe+SinC*VinFo),-ZiC*(SinC*UinFoe-CosC*VinFo)])
                        IS2e = np.array([IinFa+DiC*QinFae,DiC*IinFa+QinFae,ZiC*(CosC*UinFae+SinC*VinFa),-ZiC*(SinC*UinFae-CosC*VinFa)])
                        GT = np.dot(ATFe,IS1e)
                        GR = np.dot(ARFe,IS1e)
                        HT = np.dot(ATFe,IS2e)
                        HR = np.dot(ARFe,IS2e)


                else:
                    print('Calibrator not implemented yet')
                    sys.exit()

            elif LocC == 2:  # C behind emitter optics Eq.57
                #print("Calibrator location not implemented yet")
                S2e = np.sin(np.deg2rad(2*RotC))
                C2e = np.cos(np.deg2rad(2*RotC))

                # AS with C before the receiver optics (see document rotated_diattenuator_X22x5deg.odt)
                AF1 = np.array([1,C2g*DiO,S2g*DiO,0])
                AF2 = np.array([C2g*DiO,1-S2g**2*WiO,S2g*C2g*WiO,-S2g*ZiO*SinO])
                AF3 = np.array([S2g*DiO, S2g*C2g*WiO, 1-C2g**2*WiO, C2g*ZiO*SinO])
                AF4 = np.array([0, S2g*SinO, -C2g*SinO, CosO])

                ATF = (ATP1*AF1+ATP2*AF2+ATP3*AF3+ATP4*AF4)
                ARF = (ARP1*AF1+ARP2*AF2+ARP3*AF3+ARP4*AF4)
                ATF1 = ATF[0]
                ATF2 = ATF[1]
                ATF3 = ATF[2]
                ATF4 = ATF[3]
                ARF1 = ARF[0]
                ARF2 = ARF[1]
                ARF3 = ARF[2]
                ARF4 = ARF[3]

                # AS with C behind the emitter  --------------------------------------------
                # terms without aCal
                ATE1o, ARE1o = ATF1, ARF1
                ATE2o, ARE2o = 0., 0.
                ATE3o, ARE3o = 0., 0.
                ATE4o, ARE4o = ATF4, ARF4
                # terms with aCal
                ATE1a, ARE1a = 0. , 0.
                ATE2a, ARE2a = ATF2, ARF2
                ATE3a, ARE3a = -ATF3, -ARF3
                ATE4a, ARE4a = -2*ATF4, -2*ARF4
                # rotated AinEa by epsilon
                ATE2ae =  C2e*ATF2 + S2e*ATF3
                ATE3ae = -S2e*ATF2 - C2e*ATF3
                ARE2ae =  C2e*ARF2 + S2e*ARF3
                ARE3ae = -S2e*ARF2 - C2e*ARF3

                ATE1 = ATE1o
                ATE2e = aCal*ATE2ae
                ATE3e = aCal*ATE3ae
                ATE4 = (1-2*aCal)*ATF4
                ARE1 = ARE1o
                ARE2e = aCal*ARE2ae
                ARE3e = aCal*ARE3ae
                ARE4 = (1-2*aCal)*ARF4

                # rotated IinE
                QinEe = C2e*QinE + S2e*UinE
                UinEe = C2e*UinE - S2e*QinE

                # --- Calibration signals and Calibration correction K from measurements with LDRCal / aCal
                if (TypeC == 2) or (TypeC == 1):   #  +++++++++ rotator calibration Eq. C.4
                    AT = ATE1o*IinE + (ATE4o+aCal*ATE4a)*h*VinE
                    BT = aCal * (ATE3ae*QinEe - ATE2ae*h*UinEe)
                    AR = ARE1o*IinE + (ARE4o+aCal*ARE4a)*h*VinE
                    BR = aCal * (ARE3ae*QinEe - ARE2ae*h*UinEe)

                    # Correction paremeters for normal measurements; they are independent of LDR
                    if (not RotationErrorEpsilonForNormalMeasurements):
                        # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
                        GT = ATE1o*IinE + ATE4o*h*VinE
                        GR = ARE1o*IinE + ARE4o*h*VinE
                        HT = ATE2a*QinE + ATE3a*h*UinEe + ATE4a*h*VinE
                        HR = ARE2a*QinE + ARE3a*h*UinEe + ARE4a*h*VinE
                    else:
                        GT = ATE1o*IinE + ATE4o*h*VinE
                        GR = ARE1o*IinE + ARE4o*h*VinE
                        HT = ATE2ae*QinE + ATE3ae*h*UinEe + ATE4a*h*VinE
                        HR = ARE2ae*QinE + ARE3ae*h*UinEe + ARE4a*h*VinE

                elif (TypeC == 3) or (TypeC == 4):  # +++++++++ linear polariser calibration Eq. C.5
                    # p = +45°, m = -45°
                    AT = ATE1*IinE + ZiC*CosC*(ATE2e*QinEe + ATE4*VinE) + ATE3e*UinEe
                    BT = DiC*(ATE1*UinEe + ATE3e*IinE) + ZiC*SinC*(ATE4*QinEe - ATE2e*VinE)
                    AR = ARE1*IinE + ZiC*CosC*(ARE2e*QinEe + ARE4*VinE) + ARE3e*UinEe
                    BR = DiC*(ARE1*UinEe + ARE3e*IinE) + ZiC*SinC*(ARE4*QinEe - ARE2e*VinE)

                    # Correction paremeters for normal measurements; they are independent of LDR
                    if (not RotationErrorEpsilonForNormalMeasurements):
                        # Stokes Input Vector before receiver optics Eq. E.19 (after atmosphere F)
                        GT = ATE1o*IinE + ATE4o*VinE
                        GR = ARE1o*IinE + ARE4o*VinE
                        HT = ATE2a*QinE + ATE3a*UinE + ATE4a*VinE
                        HR = ARE2a*QinE + ARE3a*UinE + ARE4a*VinE
                    else:
                        D = IinE + DiC*QinEe
                        A = DiC*IinE + QinEe
                        B = ZiC*(CosC*UinEe + SinC*VinE)
                        C = -ZiC*(SinC*UinEe - CosC*VinE)
                        GT = ATE1o*D + ATE4o*C
                        GR = ARE1o*D + ARE4o*C
                        HT = ATE2a*A + ATE3a*B + ATE4a*C
                        HR = ARE2a*A + ARE3a*B + ARE4a*C

                elif (TypeC == 6):  # real HWP calibration +-22.5° rotated_diattenuator_X22x5deg.odt
                    # p = +22.5°, m = -22.5°
                    IE1e = np.array([IinE+sqr05*DiC*QinEe, sqr05*DiC*IinE+(1-0.5*WiC)*QinEe, (1-0.5*WiC)*UinEe+sqr05*ZiC*SinC*VinE, -sqr05*ZiC*SinC*UinEe+ZiC*CosC*VinE])
                    IE2e = np.array([sqr05*DiC*UinEe, 0.5*WiC*UinEe-sqr05*ZiC*SinC*VinE, sqr05*DiC*IinE+0.5*WiC*QinEe, sqr05*ZiC*SinC*QinEe])
                    ATEe = np.array([ATE1,ATE2e,ATE3e,ATE4])
                    AREe = np.array([ARE1,ARE2e,ARE3e,ARE4])
                    AT = np.dot(ATEe,IE1e)
                    AR = np.dot(AREe,IE1e)
                    BT = np.dot(ATEe,IE2e)
                    BR = np.dot(AREe,IE2e)

                    # Correction paremeters for normal measurements; they are independent of LDR
                    if (not RotationErrorEpsilonForNormalMeasurements):   # calibrator taken out
                        GT = ATE1o*IinE + ATE4o*VinE
                        GR = ARE1o*IinE + ARE4o*VinE
                        HT = ATE2a*QinE + ATE3a*UinE + ATE4a*VinE
                        HR = ARE2a*QinE + ARE3a*UinE + ARE4a*VinE
                    else:
                        D = IinE + DiC*QinEe
                        A = DiC*IinE + QinEe
                        B = ZiC*(CosC*UinEe + SinC*VinE)
                        C = -ZiC*(SinC*UinEe - CosC*VinE)
                        GT = ATE1o*D + ATE4o*C
                        GR = ARE1o*D + ARE4o*C
                        HT = ATE2a*A + ATE3a*B + ATE4a*C
                        HR = ARE2a*A + ARE3a*B + ARE4a*C

                else:
                    print('Calibrator not implemented yet')
                    sys.exit()

            # Calibration signals with aCal => Determination of the correction K of the real calibration factor
            IoutTp = TaT*TiT*TiO*TiE*(AT + BT)
            IoutTm = TaT*TiT*TiO*TiE*(AT - BT)
            IoutRp = TaR*TiR*TiO*TiE*(AR + BR)
            IoutRm = TaR*TiR*TiO*TiE*(AR - BR)
            # --- Results and Corrections; electronic etaR and etaT are assumed to be 1
            #Eta = TiR/TiT   # Eta = Eta*/K  Eq. 84
            Etapx = IoutRp/IoutTp
            Etamx = IoutRm/IoutTm
            Etax = (Etapx*Etamx)**0.5
            K = Etax / Eta0
            #print("{0:6.3f},{1:6.3f},{2:6.3f},{3:6.3f},{4:6.3f},{5:6.3f},{6:6.3f},{7:6.3f},{8:6.3f},{9:6.3f},{10:6.3f}".format(AT, BT, AR, BR, DiC, ZiC, RetO, TP, TS, Kp, Km))
            #print("{0:6.3f},{1:6.3f},{2:6.3f},{3:6.3f}".format(DiC, ZiC, Kp, Km))

            #  For comparison with Volkers Libreoffice Müller Matrix spreadsheet
            #Eta_test_p = (IoutRp/IoutTp)
            #Eta_test_m = (IoutRm/IoutTm)
            #Eta_test = (Eta_test_p*Eta_test_m)**0.5

            # *************************************************************************
            iLDR = -1
            for LDRTrue in LDRrange:
                iLDR = iLDR + 1
                atrue = (1-LDRTrue)/(1+LDRTrue)
                # ----- Forward simulated signals and LDRsim with atrue; from input file
                It = TaT*TiT*TiO*TiE*(GT+atrue*HT) #  TaT*TiT*TiC*TiO*IinL*(GT+atrue*HT)
                Ir = TaR*TiR*TiO*TiE*(GR+atrue*HR) #  TaR*TiR*TiC*TiO*IinL*(GR+atrue*HR)

                # LDRsim = 1/Eta*Ir/It  # simulated LDR* with Y from input file
                LDRsim = Ir/It  # simulated uncorrected LDR with Y from input file
                '''
                if Y == 1.:
                    LDRsimx = LDRsim
                    LDRsimx2 = LDRsim2
                else:
                    LDRsimx = 1./LDRsim
                    LDRsimx2 = 1./LDRsim2
                '''
                # ----- Backward correction
                # Corrected LDRCorr from forward simulated LDRsim (atrue) with assumed true G0,H0,K0
                LDRCorr = (LDRsim*K0/Etax*(GT0+HT0)-(GR0+HR0))/((GR0-HR0)-LDRsim*K0/Etax*(GT0-HT0))

                # -- F11corr from It and Ir and calibration EtaX
                Text1 = "F11corr from It and Ir with calibration EtaX: x-axis: F11corr(LDRtrue) / F11corr(LDRtrue = 0.004) - 1"
                F11corr = 1/(TiO*TiE)*((HR0*Etax/K0*It/TTa-HT0*Ir/TRa)/(HR0*GT0-HT0*GR0))    # IL = 1  Eq.(64)

                #Text1 = "F11corr from It and Ir without corrections but with calibration EtaX: x-axis: F11corr(LDRtrue) devided by F11corr(LDRtrue = 0.004)"
                #F11corr = 0.5/(TiO*TiE)*(Etax*It/TTa+Ir/TRa)    # IL = 1  Eq.(64)

                # -- It from It only with atrue without corrections - for BERTHA (and PollyXTs)
                #Text1 = " x-axis: IT(LDRtrue) / IT(LDRtrue = 0.004) - 1"
                #F11corr = It/(TaT*TiT*TiO*TiE)   #/(TaT*TiT*TiO*TiE*(GT0+atrue*HT0))
                # !!! see below line 1673ff

                aF11corr[iLDR,iN] = F11corr
                aA[iLDR,iN] = LDRCorr

                aX[0,iN] = GR
                aX[1,iN] = GT
                aX[2,iN] = HR
                aX[3,iN] = HT
                aX[4,iN] = K

                aLDRCal[iN] = iLDRCal
                aERaT[iN] = iERaT
                aERaR[iN] = iERaR
                aRotaT[iN] = iRotaT
                aRotaR[iN] = iRotaR
                aRetT[iN] = iRetT
                aRetR[iN] = iRetR

                aRotL[iN] = iRotL
                aRotE[iN] = iRotE
                aRetE[iN] = iRetE
                aRotO[iN] = iRotO
                aRetO[iN] = iRetO
                aRotC[iN] = iRotC
                aRetC[iN] = iRetC
                aDiO[iN] = iDiO
                aDiE[iN] = iDiE
                aDiC[iN] = iDiC
                aTP[iN] = iTP
                aTS[iN] = iTS
                aRP[iN] = iRP
                aRS[iN] = iRS

# --- END loop
btime = clock()
print("\r done in      ", "{0:5.0f}".format(btime-atime), "sec") #, end="\r")

# --- Plot -----------------------------------------------------------------
#sns.set_style("whitegrid")
#sns.set_palette("bright", 6)

'''
fig2 = plt.figure()
plt.plot(aA[2,:],'b.')
plt.plot(aA[3,:],'r.')
plt.plot(aA[4,:],'g.')
#plt.plot(aA[6,:],'c.')
plt.show
'''
# Plot LDR
def PlotSubHist(aVar, aX, X0, daX, iaX, naX):
    fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(25, 2))
    iLDR = -1
    for LDRTrue in LDRrange:
        iLDR = iLDR + 1

        LDRmin[iLDR] = np.min(aA[iLDR,:])
        LDRmax[iLDR] = np.max(aA[iLDR,:])
        Rmin = LDRmin[iLDR] * 0.995 #  np.min(aA[iLDR,:])    * 0.995
        Rmax = LDRmax[iLDR] * 1.005 #  np.max(aA[iLDR,:])    * 1.005

        #plt.subplot(5,2,iLDR+1)
        plt.subplot(1,5,iLDR+1)
        (n, bins, patches) = plt.hist(aA[iLDR,:],
                 bins=100, log=False,
                 range=[Rmin, Rmax],
                 alpha=0.5, normed=False, color = '0.5', histtype='stepfilled')

        for iaX in range(-naX,naX+1):
            plt.hist(aA[iLDR,aX == iaX],
                     range=[Rmin, Rmax],
                     bins=100, log=False, alpha=0.3, normed=False, histtype='stepfilled', label = str(round(X0 + iaX*daX/naX,5)))

            if (iLDR == 2): plt.legend()

        plt.tick_params(axis='both', labelsize=9)
        plt.plot([LDRTrue, LDRTrue], [0, np.max(n)], 'r-', lw=2)

    #plt.title(LID + '  ' + aVar, fontsize=18)
    #plt.ylabel('frequency', fontsize=10)
    #plt.xlabel('LDRcorr', fontsize=10)
    #fig.tight_layout()
    fig.suptitle(LID + '  ' + str(Type[TypeC]) + ' ' + str(Loc[LocC])  + ' - ' + aVar, fontsize=14, y=1.05)
    #plt.show()
    #fig.savefig(LID + '_' + aVar + '.png', dpi=150, bbox_inches='tight', pad_inches=0)
    #plt.close
    return

if (nRotL > 0): PlotSubHist("RotL", aRotL, RotL0, dRotL, iRotL, nRotL)
if (nRetE > 0): PlotSubHist("RetE", aRetE, RetE0, dRetE, iRetE, nRetE)
if (nRotE > 0): PlotSubHist("RotE", aRotE, RotE0, dRotE, iRotE, nRotE)
if (nDiE > 0): PlotSubHist("DiE", aDiE, DiE0, dDiE, iDiE, nDiE)
if (nRetO > 0): PlotSubHist("RetO", aRetO, RetO0, dRetO, iRetO, nRetO)
if (nRotO > 0): PlotSubHist("RotO", aRotO, RotO0, dRotO, iRotO, nRotO)
if (nDiO > 0): PlotSubHist("DiO", aDiO, DiO0, dDiO, iDiO, nDiO)
if (nDiC > 0): PlotSubHist("DiC", aDiC, DiC0, dDiC, iDiC, nDiC)
if (nRotC > 0): PlotSubHist("RotC", aRotC, RotC0, dRotC, iRotC, nRotC)
if (nRetC > 0): PlotSubHist("RetC", aRetC, RetC0, dRetC, iRetC, nRetC)
if (nTP > 0): PlotSubHist("TP", aTP, TP0, dTP, iTP, nTP)
if (nTS > 0): PlotSubHist("TS", aTS, TS0, dTS, iTS, nTS)
if (nRP > 0): PlotSubHist("RP", aRP, RP0, dRP, iRP, nRP)
if (nRS > 0): PlotSubHist("RS", aRS, RS0, dRS, iRS, nRS)
if (nRetT > 0): PlotSubHist("RetT", aRetT, RetT0, dRetT, iRetT, nRetT)
if (nRetR > 0): PlotSubHist("RetR", aRetR, RetR0, dRetR, iRetR, nRetR)
if (nERaT > 0): PlotSubHist("ERaT", aERaT, ERaT0, dERaT, iERaT, nERaT)
if (nERaR > 0): PlotSubHist("ERaR", aERaR, ERaR0, dERaR, iERaR, nERaR)
if (nRotaT > 0): PlotSubHist("RotaT", aRotaT, RotaT0, dRotaT, iRotaT, nRotaT)
if (nRotaR > 0): PlotSubHist("RotaR", aRotaR, RotaR0, dRotaR, iRotaR, nRotaR)
if (nLDRCal > 0): PlotSubHist("LDRCal", aLDRCal, LDRCal0, dLDRCal, iLDRCal, nLDRCal)

plt.show()
plt.close

print()
#print("IT(LDRtrue) devided by IT(LDRtrue = 0.004)")
print(Text1)
print()

iLDR = 5
for LDRTrue in LDRrange:
    iLDR = iLDR - 1
    aF11corr[iLDR,:] = aF11corr[iLDR,:] / aF11corr[0,:] - 1.0

# Plot F11
def PlotSubHistF11(aVar, aX, X0, daX, iaX, naX):
    fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(25, 2))
    iLDR = -1
    for LDRTrue in LDRrange:
        iLDR = iLDR + 1

        '''
        F11min[iLDR] = np.min(aF11corr[iLDR,:])
        F11max[iLDR] = np.max(aF11corr[iLDR,:])
        Rmin = F11min[iLDR] * 0.995 #  np.min(aA[iLDR,:])    * 0.995
        Rmax = F11max[iLDR] * 1.005 #  np.max(aA[iLDR,:])    * 1.005
        '''
        #Rmin = 0.8
        #Rmax = 1.2

        #plt.subplot(5,2,iLDR+1)
        plt.subplot(1,5,iLDR+1)
        (n, bins, patches) = plt.hist(aF11corr[iLDR,:],
                 bins=100, log=False,
                 alpha=0.5, normed=False, color = '0.5', histtype='stepfilled')

        for iaX in range(-naX,naX+1):
            plt.hist(aF11corr[iLDR,aX == iaX],
                     bins=100, log=False, alpha=0.3, normed=False, histtype='stepfilled', label = str(round(X0 + iaX*daX/naX,5)))

            if (iLDR == 2): plt.legend()

        plt.tick_params(axis='both', labelsize=9)
        #plt.plot([LDRTrue, LDRTrue], [0, np.max(n)], 'r-', lw=2)

    #plt.title(LID + '  ' + aVar, fontsize=18)
    #plt.ylabel('frequency', fontsize=10)
    #plt.xlabel('LDRcorr', fontsize=10)
    #fig.tight_layout()
    fig.suptitle(LID + '  ' + str(Type[TypeC]) + ' ' + str(Loc[LocC])  + ' - ' + aVar, fontsize=14, y=1.05)
    #plt.show()
    #fig.savefig(LID + '_' + aVar + '.png', dpi=150, bbox_inches='tight', pad_inches=0)
    #plt.close
    return

if (nRotL > 0): PlotSubHistF11("RotL", aRotL, RotL0, dRotL, iRotL, nRotL)
if (nRetE > 0): PlotSubHistF11("RetE", aRetE, RetE0, dRetE, iRetE, nRetE)
if (nRotE > 0): PlotSubHistF11("RotE", aRotE, RotE0, dRotE, iRotE, nRotE)
if (nDiE > 0): PlotSubHistF11("DiE", aDiE, DiE0, dDiE, iDiE, nDiE)
if (nRetO > 0): PlotSubHistF11("RetO", aRetO, RetO0, dRetO, iRetO, nRetO)
if (nRotO > 0): PlotSubHistF11("RotO", aRotO, RotO0, dRotO, iRotO, nRotO)
if (nDiO > 0): PlotSubHistF11("DiO", aDiO, DiO0, dDiO, iDiO, nDiO)
if (nDiC > 0): PlotSubHistF11("DiC", aDiC, DiC0, dDiC, iDiC, nDiC)
if (nRotC > 0): PlotSubHistF11("RotC", aRotC, RotC0, dRotC, iRotC, nRotC)
if (nRetC > 0): PlotSubHistF11("RetC", aRetC, RetC0, dRetC, iRetC, nRetC)
if (nTP > 0): PlotSubHistF11("TP", aTP, TP0, dTP, iTP, nTP)
if (nTS > 0): PlotSubHistF11("TS", aTS, TS0, dTS, iTS, nTS)
if (nRP > 0): PlotSubHistF11("RP", aRP, RP0, dRP, iRP, nRP)
if (nRS > 0): PlotSubHistF11("RS", aRS, RS0, dRS, iRS, nRS)
if (nRetT > 0): PlotSubHistF11("RetT", aRetT, RetT0, dRetT, iRetT, nRetT)
if (nRetR > 0): PlotSubHistF11("RetR", aRetR, RetR0, dRetR, iRetR, nRetR)
if (nERaT > 0): PlotSubHistF11("ERaT", aERaT, ERaT0, dERaT, iERaT, nERaT)
if (nERaR > 0): PlotSubHistF11("ERaR", aERaR, ERaR0, dERaR, iERaR, nERaR)
if (nRotaT > 0): PlotSubHistF11("RotaT", aRotaT, RotaT0, dRotaT, iRotaT, nRotaT)
if (nRotaR > 0): PlotSubHistF11("RotaR", aRotaR, RotaR0, dRotaR, iRotaR, nRotaR)
if (nLDRCal > 0): PlotSubHistF11("LDRCal", aLDRCal, LDRCal0, dLDRCal, iLDRCal, nLDRCal)

plt.show()
plt.close
'''
# only histogram
#print("******************* " + aVar + " *******************")
fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
iLDR = -1
for LDRTrue in LDRrange:
    iLDR = iLDR + 1
    LDRmin[iLDR] = np.min(aA[iLDR,:])
    LDRmax[iLDR] = np.max(aA[iLDR,:])
    Rmin = np.min(aA[iLDR,:])    * 0.999
    Rmax = np.max(aA[iLDR,:])    * 1.001
    plt.subplot(5,2,iLDR+1)
    (n, bins, patches) = plt.hist(aA[iLDR,:],
             range=[Rmin, Rmax],
             bins=200, log=False, alpha=0.2, normed=False, color = '0.5', histtype='stepfilled')
    plt.tick_params(axis='both', labelsize=9)
    plt.plot([LDRTrue, LDRTrue], [0, np.max(n)], 'r-', lw=2)
plt.show()
plt.close
'''

# --- Plot LDRmin, LDRmax
fig2 = plt.figure()
plt.plot(LDRrange,LDRmax-LDRrange, linewidth=2.0, color='b')
plt.plot(LDRrange,LDRmin-LDRrange, linewidth=2.0, color='g')

plt.xlabel('LDRtrue', fontsize=18)
plt.ylabel('LDRTrue-LDRmin, LDRTrue-LDRmax', fontsize=14)
plt.title(LID + ' ' + str(Type[TypeC]) + ' ' + str(Loc[LocC]), fontsize=18)
#plt.ylimit(-0.07, 0.07)
plt.show()
plt.close

# --- Save LDRmin, LDRmax to file
# http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
with open('LDR_min_max_ver7_' + LID + '.dat', 'w') as f:
    with redirect_stdout(f):
        print(LID)
        print("LDRtrue, LDRmin, LDRmax")
        for i in range(len(LDRrange)):
            print("{0:7.4f},{1:7.4f},{2:7.4f}".format(LDRrange[i], LDRmin[i], LDRmax[i]))

'''
# --- Plot K over LDRCal
fig3 = plt.figure()
plt.plot(LDRCal0+aLDRCal*dLDRCal/nLDRCal,aX[4,:], linewidth=2.0, color='b')

plt.xlabel('LDRCal', fontsize=18)
plt.ylabel('K', fontsize=14)
plt.title(LID, fontsize=18)
plt.show()
plt.close
'''

# Additional plot routines ======>
'''
#******************************************************************************
# 1. Plot LDRcorrected - LDR(measured Icross/Iparallel)
LDRa = np.arange(1.,100.)*0.005
LDRCorra = np.arange(1.,100.)
if Y == - 1.: LDRa = 1./LDRa
LDRCorra = (1./Eta*LDRa*(GT+HT)-(GR+HR))/((GR-HR)-1./Eta*LDRa*(GT-HT))
if Y == - 1.: LDRa = 1./LDRa
#
#fig = plt.figure()
plt.plot(LDRa,LDRCorra-LDRa)
plt.plot([0.,0.5],[0.,0.5])
plt.suptitle('LDRcorrected - LDR(measured Icross/Iparallel)', fontsize=16)
plt.xlabel('LDR', fontsize=18)
plt.ylabel('LDRCorr - LDR', fontsize=16)
#plt.savefig('test.png')
#
'''
'''
#******************************************************************************
# 2. Plot LDRsim (simulated measurements without corrections = Icross/Iparallel) over LDRtrue
LDRa = np.arange(1.,100.)*0.005
LDRsima = np.arange(1.,100.)

atruea = (1.-LDRa)/(1+LDRa)
Ita = TiT*TiO*IinL*(GT+atruea*HT)
Ira = TiR*TiO*IinL*(GR+atruea*HR)
LDRsima = Ira/Ita  # simulated uncorrected LDR with Y from input file
if Y == -1.: LDRsima = 1./LDRsima
#
#fig = plt.figure()
plt.plot(LDRa,LDRsima)
plt.plot([0.,0.5],[0.,0.5])
plt.suptitle('LDRsim (simulated measurements without corrections = Icross/Iparallel) over LDRtrue', fontsize=10)
plt.xlabel('LDRtrue', fontsize=18)
plt.ylabel('LDRsim', fontsize=16)
#plt.savefig('test.png')
#
'''
