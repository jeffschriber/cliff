#!/usr/bin/env python
#

import logging
import cliff.helpers.utils as util
import cliff.helpers.constants as constants
import configparser
import os

class Options:
    """
    Class to parse and store options

    """


    def __init__(self, config_file="config.ini", name=None):
        # Config parser
        self.Config = configparser.ConfigParser()
        # Load config file
        self.Config.read(config_file)
        # Set logger
        self.name = "output"
        if name is not None:
            self.name = name

        self.logger = logging.getLogger(__name__)
        fh = logging.FileHandler(self.name + '.log')
        self.logger.addHandler(fh)


        # Logger level
        self.logger_level = 20
        self.logger.setLevel(self.logger_level)
        util.set_logger_level(self.logger_level)

        self.test_mode = False

        # The options and default values

        models_path = os.path.dirname(os.path.realpath(__file__))
        models_path += "/../models/large/"
        # Hirshfeld train/learn options
        self.hirsh_training = models_path + "hirsh/"
        self.hirsh_krr_kernel = 'laplacian'
        self.hirsh_krr_sigma = 1000.0
        self.hirsh_krr_lambda = 1e-9
        self.hirsh_filepath = ""
        self.hirsh_file_read = False
        self.hirsh_cutoff = 4.0
        self.hirsh_save_to_disk = False
        self.hirsh_save_path = ""
    

        # Atomic Density options
        self.atomicdensity_training = models_path + "adens/" 
        self.atomicdensity_cutoff = 4.0
        self.atomicdensity_krr_kernel = 'laplacian'
        self.atomicdensity_krr_sigma = 1000.0
        self.atomicdensity_krr_lambda = 1e-9
        self.atomicdensity_ref_adens = False
        self.atomicdensity_refpath = ""
        self.atomicdensity_save_to_disk = False
        self.atomicdensity_save_path = ""

        # Defaults for multipoles
        self.multipole_training = models_path + "mtp/"
        self.multipole_ml_method = "KRR"
        self.multipole_kernel = 'laplacian'
        self.multipole_krr_sigma = 10.0
        self.multipole_krr_lambda = 1e-3
        self.multipole_correct_charge = True
        self.multipole_ref_mtp = False
        self.multipole_ref_path = ""
        self.multipole_save_to_disk = False
        self.multipole_save_path = ""
        self.multipole_rcut = 4.5

        # Defaults for electrostatics
        self.elst_type = "damped_mtp"
        self.elst_damping_exponents = constants.elst_cp_exp 

        # Defaults for Induction
        self.indu_sr_params = constants.indu_sr_params
        self.indu_smearing_coeff = 0.38539063
        self.indu_omega = 0.75 
        self.indu_conv = 1e-5

        # Defaults for Exchange
        self.exch_int_params = constants.exch_int_params 

        # Defaults for dispersion/polarizability

        self.disp_method    = 'TT'
        self.disp_coeffs    = constants.disp_coeffs
        self.pol_scs_cutoff = 5.01451 
        self.disp_beta      = 2.40871
        self.disp_radius    = 0.57785
        self.pol_exponent   = 0.177346
        #self.pol_scs_cutoff = 3.93473 
        #self.disp_beta      = 2.40871
        #self.disp_radius    = 0.58605
        #self.pol_exponent   = 0.177346


        # load the options
        self.load_hirsh_options()
        self.load_atomic_density_options()
        self.load_multipole_options()
        self.load_elst_options()
        self.load_indu_options()
        self.load_exch_options()
        self.load_disp_options()

    def load_hirsh_options(self):
        
        #test
        try:
            val = self.Config.get("output", "test_mode")
            if val in ["True", "true", "t", "1"]:
                self.test_mode = True 
            else:
                self.test_mode = False
        except:
            pass
        # log level
        try:
            self.logger_level = self.Config.getint("output", "log_level")
        except:
            pass

        ### Options for computing Hirshfeld ratios
        try:
            self.hirsh_training = self.Config.get("hirshfeld","training")
        except:
            pass

        # Return type of kernel
        try:
            self.hirsh_krr_kernel =  self.Config.get("hirshfeld","krr_kernel")
        except:
            pass

        # Return sigma parameter for #machinelearning
        try:
            self.hirsh_krr_sigma = self.Config.getfloat("hirshfeld","krr_sigma")
        except:
            pass

        # Return sigma parameter for machinelearning
        try:
            self.hirsh_krr_lambda = self.Config.getfloat("hirshfeld","krr_lambda")
        except:
            pass

        # cutoff for ml descriptor
        try:
            self.hirsh_cutoff = self.Config.getfloat("hirshfeld","cutoff")
        except:
            pass

        # no default
        try:
            self.hirsh_filepath = self.Config.get("hirshfeld","ref_path")
        except:
            pass

        # no default
        try:
            val = self.Config.get("hirshfeld","ref_hirsh")
            if val in ["True", "true", "t", "1"]:
                self.hirsh_file_read = True 
            else:
                self.hirsh_file_read = False
        except:
            pass


        try:
            val = self.Config.get("hirshfeld","save_to_disk")
            if val in ["True", "true", "t", "1"]:
                self.hirsh_save_to_disk = True 
        except:
            pass
        try:
            self.hirsh_save_path = self.Config.get("hirshfeld","ref_path")
        except:
            pass


    def set_logger_level(self, val):
        self.logger_level = val

    def set_hirshfeld_training(self, val):
        self.hirsh_training = val

    def set_hirshfeld_max_neighbors(self,val):
        self.hirsh_max_neighbors = val

    def set_hirshfeld_krr_kernel(self, val):
        self.hirsh_krr_kernel = val

    def set_hirshfeld_krr_sigma(self, val):
        self.hirsh_krr_sigma = val

    def set_hirshfeld_krr_lambda(self, val):
        self.hirsh_krr_lambda = val

    def set_hirshfeld_file_read(self, val):
        self.hirsh_file_read = val

    def set_hirshfeld_filepath(self, val):
        self.hirsh_filepath = val

    def set_hirshfeld_cutoff(self, val):
        self.hirsh_cutoff = val
    def set_hirshfeld_ref_hirsh(self, val):
        self.hirsh_file_read = val

    def set_hirsh_save_to_disk(self,val):
        self.hirsh_save_to_disk = val
    def set_hirsh_save_path(self,val):
        self.hirsh_save_path = val

    ### Options for atomic density computations 
    def load_atomic_density_options(self):
        try:
            self.atomicdensity_training = self.Config.get("atomicdensity","training")
        except:
            pass

        try:
            self.atomicdensity_cutoff = self.Config.getfloat("atomicdensity","cutoff")
        except:
            pass

        try:
            self.atomicdensity_krr_kernel = self.Config.get("atomicdensity","krr_kernel")
        except:
            pass

        try:
            self.atomicdensity_krr_sigma = self.Config.getfloat("atomicdensity","krr_sigma")
        except:
            pass

        try:
            self.atomicdensity_krr_lambda = self.Config.getfloat("atomicdensity","krr_lambda")
        except:
            pass

        try:
            val = self.Config.get("atomicdensity","ref_adens")
            if val in ["True", "true", "t", "1"]:
                self.atomicdensity_ref_adens = True 
        except:
            pass

        try:
            self.atomicdensity_refpath = self.Config.get("atomicdensity","ref_path")
        except:
            pass

        try:
            val = self.Config.get("atomicdensity","save_to_disk")
            if val in ["True", "true", "t", "1"]:
                self.atomicdensity_save_to_disk = True 
        except:
            pass
        try:
            self.atomicdensity_save_path = self.Config.get("atomicdensity","ref_path")
        except:
            pass



    def set_atomicdensity_training(self, val):
        self.atomicdensity_training = val

    def set_atomicdensity_cutoff(self, val):
        self.atomicdensity_cutoff = val

    def set_atomicdensity_krr_kernel(self, val):
        self.atomicdensity_krr_kernel = val

    def set_atomicdensity_krr_sigma(self, val):
        self.atomicdensity_krr_sigma = val

    def set_atomicdensity_krr_lambda(self, val):
        self.atomicdensity_krr_lambda = val

    def set_atomicdensity_ref_adens(self, val):
        self.atomicdensity_ref_adens = val 

    def set_atomicdensity_refpath(self, val):
        self.atomicdensity_refpath = val

    def set_atomicdensity_save_to_disk(self,val):
        self.atomicdensity_save_to_disk = val
    def set_atomicdensity_save_path(self,val):
        self.atomicdensity_save_path = val


    ### Options for multipoles

    def load_multipole_options(self):
        try:
            self.multipole_max_neighbors = self.Config.getint("multipoles","max_neighbors")
        except:
            pass

        try:
            self.multipole_ml_method = self.Config.get("multipoles","ml_method")
        except:
            pass

        try:
            self.multipole_training = self.Config.get("multipoles","training")
        except:
            pass

        try:
            self.multipole_kernel = self.Config.get("multipoles","kernel")
        except:
            pass

        try:
            self.multipole_krr_sigma = self.Config.getfloat("multipoles","krr_sigma")
        except:
            pass

        try:
            self.multipole_krr_lambda = self.Config.getfloat("multipoles","krr_lambda")
        except:
            pass
        try:
            self.multipole_rcut = self.Config.getfloat("multipoles","rcut")
        except:
            pass

        try:
            val = self.Config.get("multipoles","correct_charge")
            if val in ["True","true","t","1"]:
                self.multipole_correct_charge = True 
            else:
                self.multipole_correct_charge = False
        except:
            pass

        try:
            val = self.Config.get("multipoles","ref_mtp")
            if val in ["True","true","t","1"]:
                self.multipole_ref_mtp = True
            else:
                self.multipole_ref_mtp = False
        except:
            pass 

        try:
            self.multipole_ref_path = self.Config.get("multipoles","ref_path")
        except:
            pass

        try:
            val = self.Config.get("multipoles","save_to_disk")
            if val in ["True","true","t","1"]:
                self.multipole_save_to_disk = True
        except:
            pass

        try:
            self.multipole_save_path = self.Config.get("multipoles","ref_path")
        except:
            pass

    def set_multipole_training(self, val):
        self.multipole_training = val

    def set_multipole_ml_method(self, val):
        self.multipole_ml_method = val

    def set_multipole_kernel(self, val):
        self.multipole_kernel = val

    def set_multipole_krr_sigma(self, val):
        self.multipole_krr_sigma = val

    def set_multipole_krr_lambda(self, val):
        self.multipole_krr_lambda = val
    
    def set_multipole_correct_charge(self, val):
        self.multipole_correct_charge = val

    def set_multipole_ref_mtp(self, val):
        self.multipole_ref_mtp = val

    def set_multipole_ref_path(self, val):
        self.multipole_ref_path = val

    def set_mtp_to_disk(self, val):
        self.multipole_save_to_disk = val

    def set_multipole_save_path(self, val):
        self.multipole_save_path = val
    def set_multipole_rcut(self, val):
        self.multipole_rcut = val

    ### Options for Electrostatics
    
    def load_elst_options(self):
        try:
            self.elst_type =  self.Config.get("electrostatics","type")
        except:
            pass
        
        try:
            ret = {}
            at_types = ['Cl', 'F', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2',  'N3', 'N2', 'N1', 'O1', 'O2','Br']  
            # Grab exponents from config if provided
            for at in at_types:
                ret[at] = self.Config.getfloat("electrostatics", "exp["+at+"]")
            self.elst_damping_exponents = ret
        except:
            pass

    def set_elst_type(self, val):
        self.elst_type = val
    
    def set_damping_exponents(self, val):
        self.elst_damping_exponents = val

    ### Options for Induction
    def load_indu_options(self):
            
        try:
            at_types = ['Cl', 'F', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2',  'N3', 'N2', 'N1', 'O1', 'O2', 'Br']  
            ret = {}
            # Grab exponents from config if provided
            for at in at_types:
                ret[at] = self.Config.getfloat("induction", "sr["+at+"]")
            self.indu_sr_params = ret
        except:
            pass

        try:
            self.indu_smearing_coeff = self.Config.getfloat("induction","smearing_coeff")
        except:
            pass

        try:
            self.indu_omega = self.Config.getfloat("induction","omega")      
        except:
            pass

        try:
            self.indu_conv = self.Config.getfloat("induction","convergence_thrshld")      
        except:
            pass

    def set_induction_sr_params(self, val):
            self.indu_sr_params = val
    
    def set_indu_smearing_coeff(self, val):
        self.indu_smearing_coeff = val
        
    def set_induction_omega(self, val):
        self.indu_omega = val
    
    def set_induction_conv(self, val):
        self.indu_conv = val

    ### Options for Exchange
    def load_exch_options(self):
        try:
            at_types = ['Cl', 'F', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2',  'N3', 'N2', 'N1', 'O1', 'O2', 'Br']  
            ret = {}
            # Grab exponents from config if provided
            for at in at_types:
                ret[at] = self.Config.getfloat("repulsion", "rep["+at+"]")
            self.exch_int_params = ret
        except:
            pass 

    def set_exchange_int_params(self, val):
            self.exch_int_params = val

    ### Options for Disp/polarizability
    def load_disp_options(self):
        try:
            self.pol_scs_cutoff = self.Config.getfloat("polarizability","scs_cutoff")
        except:
            pass

        try:
            self.disp_beta = self.Config.getfloat("polarizability","beta")
        except:
            pass

        try:
            self.disp_radius = self.Config.getfloat("polarizability","radius")
        except:
            pass

        try:
            self.pol_exponent = self.Config.getfloat("polarizability","exponent")
        except:
            pass
    
        try:
            self.disp_method = self.Config.get("dispersion","method")
        except:
            pass

        try:
            at_types = ['Cl', 'F', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2',  'N3', 'N2', 'N1', 'O1', 'O2', 'Br']  
            ret = {}
            # Grab exponents from config if provided
            for at in at_types:
                ret[at] = self.Config.getfloat("dispersion", "disp["+at+"]")
            self.disp_coeffs = ret
        except:
            pass 

    def set_pol_scs_cutoff(self, val):
        self.pol_scs_cutoff = val

    def set_disp_beta(self, val):
        self.disp_beta = val

    def set_disp_coeffs(self, val):
        self.disp_coeffs = val

    def set_disp_radius(self, val):
        self.disp_radius = val

    def set_pol_exponent(self, val):
        self.pol_exponent = val
    
    def set_disp_method(self, val):
        self.disp_method = val

    def set_name(self, name):
        self.name = name


