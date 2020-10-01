#!/usr/bin/env python
#
# List of global parameters and constants.
#
# Tristan Bereau (2017)

# Atomic units to kcal/mol
au2kcalmol    = 627.5095
# angstrom to bohr
a2b           = 1.8897268
# Bohr to Angstrom
b2a           = 0.529177
# Multipole conversion: Hartree and bohr to kcal/mol (332.064)
hbohr2kcalmol = 332.063595

# Some mathematical constants
sqrt_3      = 1.732050808

# Degrees to radian
deg2rad     = 0.017453292519943295

# au to debye
au2debye    = 2.541746231
debye2au    = 0.393430307

atomic_number = {
  'H'  : 1,
  'B'  : 5,
  'C'  : 6,
  'N'  : 7,
  'O'  : 8,
  'F'  : 9,
  'P'  :15,
  'S'  :16,
  'Cl' :17,
  'Br' :35,
  'I'  :53,
}

Z_val = {
    'H' : 1,
    'B' : 3,
    'C' : 4,
    'N' : 5,
    'O' : 6,
    'F' : 7,
    'P' : 5,
    'S' : 6,
    'Cl': 7,
    'Br': 7,
    'I' : 7,
}

atom_row = {
    'H' : 0,
    'B' : 1,
    'C' : 1,
    'N' : 1,
    'O' : 1,
    'F' : 1,
    'P' : 2,
    'S' : 2,
    'Cl': 2,
    'Br': 3,
    'I' : 4,
}

pol_free = {
  'H' :  4.50,
  'He':  1.38,
  'C' : 12.00,
  'N' :  7.40,
  'O' :  5.40,
  'F' :  3.80,
  'Ne':  2.67,
  'Si': 37.00,
  'P' : 25.00,
  'S' : 19.60,
  'Cl': 15.00,
  'Ar': 11.10,
  'Br': 20.00,
  'Kr': 16.80,
  'I' : 35.00,
}

csix_free  = {
  'H' :  6.50,
  'He':  1.46,
  'C' : 46.60,
  'N' : 24.20,
  'O' : 15.60,
  'F' :  9.52,
  'Ne':  6.38,
  'Si': 305.0,
  'P' : 185.0,
  'S' : 134.0,
  'Cl': 94.60,
  'Ar': 64.30,
  'Br': 162.0,
  'Kr': 130.0,
  'I' : 385.0,
}

rad_free = {
  'H' : 3.10,
  'He': 2.65,
  'C' : 3.59,
  'N' : 3.34,
  'O' : 3.19,
  'F' : 3.04,
  'Ne': 2.91,
  'Si': 4.20,
  'P' : 4.01,
  'S' : 3.86,
  'Cl': 3.71,
  'Ar': 3.55,
  'Br': 3.93,
  'Kr': 3.82,
  'I' : 4.39,
}

atomic_r2 = {
 'C': 14.046740332406095,
'Cl': 27.651441841502898,
 'F': 10.601441421661603,
 'H': 3.1589186627858976,
 'N': 12.419553635522538,
 'O': 11.566513469042524,
 'S': 29.17364957903178,
'Br':40.3810331885329
}


atomic_r4 = {
  'C':  109.39192409419,
 'Cl':  187.29318856137,
  'F':  41.707556384507,
  'H':  26.818416656404,
  'N':  71.475971390365,
  'O':  56.180301267329,
  'S': 239.014016575349,
 'Br': 288.05982633399606
}

atomic_weight = {
  'H'  : 1.008,
  'B'  : 10.81,
  'C'  : 12.01,
  'N'  : 14.01,
  'O'  : 16.00,
  'F'  : 19.00,
  'P'  : 31.00,
  'S'  : 32.06,
  'Cl' : 35.45,
  'Br' : 79.90,
  'I'  : 126.90,
}

# Covalent radii
cov_rad = {
    'H' : 0.31,
    'C' : 0.70,
    'N' : 0.71,
    'O' : 0.66,
    'F' : 0.57,
    'P' : 1.07,
    'S' : 1.05,
    'Cl': 1.02,
    'Br': 1.20,
    'I' : 1.39,
}

# Hbond interaction: Strength k_hbnd (according to Grimme)
k_hbnd = {
    'N' : 0.8,
    'O' : 0.3,
    'F' : 0.1,
    'P' : 2.0,
    'S' : 2.0,
    'Cl': 2.0,
    'Br': 2.0,
    'I' : 2.0,
}

# Map multipole coefficient to index
map_mtp_coeff = {
  'Q10'  : 2,
  'Q11c' : 0,
  'Q11s' : 1,
  'Q20'  : 0,
  'Q21c' : 1,
  'Q21s' : 2,
  'Q22c' : 3,
  'Q22s' : 4,
}

# Atoms for bag of bonds (correctly ordered)
bob_atoms = ['C','H','O','S','N','Br','Cl','F','I','P','B']

# Charge penetration
# Follows Wang et al. JCTC (2017) DOI: 10.1021/acs.jctc.5b00267
# effective core charge
cp_Z = {
  'H'  : 1,
  'B'  : 3,
  'C'  : 4,
  'N'  : 5,
  'O'  : 6,
  'F'  : 7,
  'P'  : 5,
  'S'  : 6,
  'Cl' : 7,
  'Br' : 7,
}

# valence-alpha set [Ang^-1]
cp_alpha = {
  'H'  : 2.0,
  'C'  : 4.0,
  'N'  : 5.0,
  'O'  : 6.0,
  'F'  : 7.0,
  'P'  : 5.0,
  'S'  : 6.0,
  'Cl' : 7.0,
  'Br' : 7.0,
}

# Free-atom valence widths [Bohr^-1]
val_width_free = {
    'H': 0.5094,
    'C': 0.5242,
    'N': 0.4415,
    'O': 0.3882,
}

# Free-atom valence charges
val_charge_free = {
    'H': -1.00000061,
    'C': -4.31910903,
    'N': -5.35306426,
    'O': -6.36289409,
}

ml_metric = {
    'gaussian': 'euclidean',
    'laplacian': 'cityblock'
}

ml_prefactor = {
    'gaussian': 2.0,
    'laplacian': 1.0
}

ml_power = {
    'gaussian': 2,
    'laplacian': 1
}


# This is how IPML re-shuffles charge when correcting
# for non-integer total charge. Instead of these parameters.
# we'll use atomic weights
#ml_chg_correct_error = {
#    'H': 1.,
#    'C': 1.,
#    'N': 1.,
#    'O': 1.,
#    'S': 1., # NOT SURE WHAT VALUE
#}
#ml_chg_correct_error = {
#    'H': 1.,
#    'C': 3.2,
#    'N': 1.7,
#    'O': 1.,
#    'S': 1., # NOT SURE WHAT VALUE
#}
# based on mae
#ml_chg_correct_error = {
#    'H': 0.144236,
#    'C': 0.362682,
#    'N': 0.463021,
#    'O': 0.290041,
#    'S': 0.741042, 
#}
ml_chg_correct_error = {
    'H'  : 1.0,
    'C'  : 2.5145,
    'N'  : 3.2101,
    'O'  : 2.0108,
    'S'  : 5.108, 
    'Cl' : 2.703,
    'F'  : 1.138, 
    'Br'  : 1.138 
}

## Default damping parameters 
##for CP-corrected electrostatics
elst_cp_exp = {
'Cl'  :  3.44010743 ,
'F'   :  4.31542815 ,
'S1'  :  3.0618362  ,
'S2'  :  3.10341896 ,
'HS'  :  3.59745201 ,
'HC'  :  3.59823675 ,
'HN'  :  3.25572817 , 
'HO'  :  3.12565888 ,
'C4'  :  3.39136671 ,
'C3'  :  3.3325365  ,
'C2'  :  3.1056848  ,
'N3'  :  3.43675725 , 
'N2'  :  3.03719323 ,
'N1'  :  3.37839207 ,
'O1'  :  3.60304401 , 
'O2'  :  3.87579112 ,
'Br'  :  3.69423067}

# Default short-range induction parameters
indu_sr_params = {
'Cl' :  0.846826222, 
'F'  :  1.52835207 , 
'S1' :  0.98640028 , 
'S2' :  0.769898525,
'HS' :  0.603114165,
'HC' :  0.377905016, 
'HN' :  0.595195215, 
'HO' :  0.68555725 , 
'C4' :  1.9007004e-5, 
'C3' :  0.284254122, 
'C2' :  0.784382989,
'N3' :  1.75474273 , 
'N2' :  1.52093681 , 
'N1' :  0.813639435, 
'O1' :  1.63728586 , 
'O2' :  1.14792616 , 
'Br' :  1.16122601
}

exch_int_params = {
'Cl'  :  3.81526151  , 
'F'   :  7.60403972  , 
'S1'  :  3.17730436  , 
'S2'  :  3.28424865  ,
'HS'  :  0.790922939 ,
'HC'  :  0.988983705 , 
'HN'  :  0.690947498 , 
'HO'  :  0.599556368 , 
'C4'  :  2.2647628   , 
'C3'  :  2.45660744  , 
'C2'  :  2.80246355  ,
'N3'  :  4.46638908  , 
'N2'  :  4.62522968  , 
'N1'  :  3.48983686  , 
'O1'  :  5.34364856  , 
'O2'  :  5.85426286  ,
'Br'  :  4.10063123
}

disp_coeffs = {
'Cl'  : 0.628883473  , 
'F'   : 0.593350598  , 
'S1'  : 0.724981123  , 
'S2'  : 0.68979661   ,
'HS'  : 9.97960769e-6 ,
'HC'  : 0.161836039  , 
'HN'  : 0.142032196  , 
'HO'  : 3.36623485e-7 , 
'C4'  : 0.349153282  , 
'C3'  : 0.380193754  , 
'C2'  : 0.474874035  ,
'N3'  : 0.2513869    , 
'N2'  : 0.921619075  , 
'N1'  : 0.814331203  , 
'O1'  : 0.779331151  , 
'O2'  : 0.547770643  , 
'Br'  : 0.499292816
}


# Default smearing coefficient for induction (Thole model)
indu_smearing_coeff = 0.38539063 


