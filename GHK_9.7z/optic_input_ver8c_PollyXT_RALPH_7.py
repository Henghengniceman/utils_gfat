# This Python script will be executed from within the main calc_lidar_correction_parameters_G_H_K.py
# Probably it will be better in the future to let the main script rather read a conguration file, which might improve the portability of the code within an executable.
# Due to problems I had with some two letter variables, most variables are now with at least three letters mixed small and capital.

# Header to identify the lidar system
# Values of DO, DT, and DR etc. from fit to lamp calibrations in Leipzig (LampCalib_2_invers_c_D0=0.opj)
EID = "oh"				# Earlinet station ID
LID = "POLLY_XT_RALPH LampCalib_2_invers_c_DO=0.opj ver8c-7" 	# Additional lidar ID (short descriptive text)
# firet fit intern (FITLN1) => DO = 0, DT fixed -0.9998, eta and DR fitted,
# => internal calib with LinPol before the receiver
print("    Lidar system :", EID, ", ", LID)


# --- IL Laser IL and +-Uncertainty
bL = 1.	#degree of linear polarization; default 1
RotL, dRotL, nRotL 	= 90, 	1., 	0	#alpha; rotation of laser polarization in degrees; default 0
# --- ME Emitter and +-Uncertainty
DiE, dDiE, nDiE 	= 0., 	0.1, 	0	# Diattenuation
TiE 		= 1.		# Unpolarized transmittance
RetE, dRetE, nRetE 	= 0., 	180.0, 	0	# Retardance in degrees
RotE, dRotE, nRotE 	= 0., 	1.0, 	0	# beta: Rotation of optical element in degrees

# --- MO Receiver Optics including telescope 
DiO,  dDiO, nDiO 	= 0.0, 	0.0022, 0
TiO 				= 1.0 				
RetO, dRetO, nRetO 	= 0., 	180.0, 	0 
RotO, dRotO, nRotO 	= 0., 	0.5, 	0	#gamma

# --- PBS MT transmitting path defined with (TS,TP);  and +-Uncertainty
# --- Pol.Filter
ERaT, dERaT, nERaT	 = 0.0001, 0.0001, 1 # Extinction ratio
RotaT, dRotaT, nRotaT = 90., 	2., 	0 # Rotation of the pol.-filter in degrees
DaT = (1-ERaT)/(1+ERaT)
TaT 		= 0.5*(1+ERaT)
# --- PBS combined with Pol.Filter
TP,   dTP, nTP	 	= 0.512175,	0.0024, 1
TS,   dTS, nTS	 	= 1-TP,	0.02, 0
TiT = 0.5 * (TP + TS)
DiT = (TP-TS)/(TP+TS)
RetT, dRetT, nRetT	 = 0., 		180., 	0 # Retardance in degrees

# --- PBS MR reflecting path defined with (RS,RP);  and +-Uncertainty
# --- Pol.Filter
ERaR, dERaR, nERaR	  = 1,	0.003,	0
RotaR, dRotaR, nRotaR = 0., 	2.,		0
DaR = (1-ERaR)/(1+ERaR)
TaR 		= 0.5*(1+ERaR)
# --- PBS 50/50
RP, dRP, nRP        = 1-TP,  0.02, 0
RS, dRS, nRS        = 1-TS,  0.00, 0
RetR, dRetR, nRetR	= 0.,		180., 	0
TiR = 0.5 * (RP + RS)
DiR = (RP-RS)/(RP+RS)

# --- Parallel signal detected in the transmitted channel => Y = 1, or in the reflected channel => Y = -1
Y = -1.

# --- Calibrator Location
LocC = 3 #location of calibrator: 1 = behind laser; 2 = behind emitter; 3 = before receiver; 4 = before PBS
# --- Calibrator Type used; defined by matrix values below
TypeC = 3	#Type of calibrator: 1 = mechanical rotator; 2 = hwp rotator (fixed retardation); 3 = linear polarizer; 4 = qwp; 5 = circular polarizer; 6 = real HWP calibration +-22.5°
# --- MC Calibrator
if TypeC == 1:  #mechanical rotator
	DiC, dDiC, nDiC 	= 0., 	0., 	0
	TiC = 1.
	RetC, dRetC, nRetC 	= 0., 	0., 	0
	RotC, dRotC, nRotC 	= 0., 	0.1, 	1	#constant calibrator offset epsilon
	# Rotation error without calibrator: if False, then epsilon = 0 for normal measurements	
	RotationErrorEpsilonForNormalMeasurements = True	# 	is in general True for TypeC == 1 calibrator
elif TypeC == 2:   # HWP rotator
	DiC, dDiC, nDiC 	= 0., 	0., 	0
	TiC = 1.
	RetC, dRetC, nRetC 	= 180., 0., 	0
	#NOTE: use here twice the HWP-rotation-angle
	RotC, dRotC, nRotC 	= 0.0, 	0.1, 	1	#constant calibrator offset epsilon
	RotationErrorEpsilonForNormalMeasurements = True	# 	is in general True for TypeC == 2 calibrator
elif TypeC == 3:   # linear polarizer calibrator
	DiC, dDiC, nDiC 	= 0.9998, 0.0001, 1	# ideal 1.0
	TiC = 0.505	# ideal 0.5
	RetC, dRetC, nRetC 	= 0., 	0., 	0
	RotC, dRotC, nRotC 	= 0.0, 	0.1, 	1	#constant calibrator offset epsilon
	RotationErrorEpsilonForNormalMeasurements = False	# 	is in general False for TypeC == 3 calibrator
elif TypeC == 4:   # QWP calibrator
	DiC, dDiC, nDiC 	= 0.0, 	0., 	0	# ideal 1.0
	TiC = 1.0	# ideal 0.5
	RetC, dRetC, nRetC 	= 90., 	0., 	0
	RotC, dRotC, nRotC 	= 0.0, 	0.1, 	1	#constant calibrator offset epsilon
	RotationErrorEpsilonForNormalMeasurements = False	# 	is  False for TypeC == 4 calibrator
elif TypeC == 6:   # real half-wave plate calibration at +-22.5°  => rotated_diattenuator_X22x5deg.odt
	DiC, dDiC, nDiC 	= 0., 	0., 	0
	TiC = 1.
	RetC, dRetC, nRetC 	= 180., 0., 	0
    #Note: use real HWP angles here
	RotC, dRotC, nRotC 	= 0.0, 	0.1, 	1	#constant calibrator offset epsilon -1.15
	RotationErrorEpsilonForNormalMeasurements = True	# 	is in general True for TypeC == 6 calibrator
else:
    print ('calibrator not implemented yet')
    sys.exit()

# --- LDRCal assumed atmospheric linear depolarization ratio during the calibration measurements (first guess)
LDRCal,dLDRCal,nLDRCal= 0.006, 0.02, 1

# ====================================================
# NOTE: there is no need to change anything below.

# --- LDRtrue for simulation of measurement => LDRsim
LDRtrue = 0.4
LDRtrue2 = 0.004

# --- measured LDRm will be corrected with calculated parameters GHK
LDRmeas = 0.3

# --- this is just for correct transfer of the variables to the main file 
RotL0, dRotL, nRotL = RotL, dRotL, 	nRotL 
# Emitter
DiE0,  dDiE,  nDiE  = DiE,  dDiE, 	nDiE  
RetE0, dRetE, nRetE = RetE, dRetE, 	nRetE 
RotE0, dRotE, nRotE = RotE, dRotE, 	nRotE 
# Receiver
DiO0,  dDiO,  nDiO  = DiO,  dDiO, 	nDiO  
RetO0, dRetO, nRetO = RetO, dRetO, 	nRetO 
RotO0, dRotO, nRotO = RotO, dRotO, 	nRotO 
# Calibrator
DiC0,  dDiC,  nDiC  = DiC,  dDiC, 	nDiC  
RetC0, dRetC, nRetC = RetC, dRetC, 	nRetC 
RotC0, dRotC, nRotC = RotC, dRotC, 	nRotC 
# PBS
TP0,   dTP,   nTP   = TP,   dTP, 	nTP   
TS0,   dTS,   nTS   = TS,   dTS, 	nTS 
RetT0, dRetT, nRetT	= RetT, dRetT, nRetT

ERaT0, dERaT, nERaT	= ERaT, dERaT, nERaT
RotaT0,dRotaT,nRotaT= RotaT,dRotaT,nRotaT

RP0,   dRP,   nRP   = RP,   dRP,   nRP
RS0,   dRS,   nRS   = RS,   dRS,   nRS
RetR0, dRetR, nRetR	= RetR, dRetR, nRetR

ERaR0, dERaR, nERaR	= ERaR, dERaR, nERaR
RotaR0,dRotaR,nRotaR= RotaR,dRotaR,nRotaR

LDRCal0,dLDRCal,nLDRCal=LDRCal,dLDRCal,nLDRCal