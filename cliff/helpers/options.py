#!/usr/bin/env python
#

import logging
import cliff.helpers.utils as util
import cliff.helpers.constants as constants
import configparser

class Options:
    """
    Class to parse and store options

    """

    # Set logger
    logger = logging.getLogger(__name__)
    logging.basicConfig()

    def __init__(self, config_file="config.ini"):
        # Config parser
        self.Config = configparser.ConfigParser()
        # Load config file
        self.Config.read(config_file)
        # Logger level
        self.logger_level = 6
        self.logger.setLevel(self.logger_level)
        util.set_logger_level(self.logger_level)

        # The options and default values

        # Hirshfeld train/learn options
        self.hirsh_training = "" 
        self.hirsh_max_neighbors = 12
        self.hirsh_krr_kernel = 'laplacian'
        self.hirsh_krr_sigma = 1000.0
        self.hirsh_krr_lambda = 1e-9
        self.hirsh_svr_kernel = 'linear'
        self.hirsh_svr_C = 1.0
        self.hirsh_filepath = ""
        self.hirsh_file_read = False
        self.hirsh_svr_epsilon = 0.5

        # Atomic Density options
        self.atomicdensity_training = "" 
        self.atomicdensity_training_env = ""
        self.atomicdensity_max_neighbors = 12
        self.atomicdensity_max_neighbors_env = 2
        self.atomicdensity_krr_kernel = 'laplacian'
        self.atomicdensity_krr_sigma = 1000.0
        self.atomicdensity_krr_lambda = 1e-9
        self.atomicdensity_krr_sigma_env = 1000.0
        self.atomicdensity_krr_gamma_env = 1000.0
        self.atomicdensity_ref_adens = False
        self.atomicdensity_refpath = ""

        # Defaults for multipoles
        self.multipole_max_neighbors = 12
        self.multipole_training = ""
        self.multipole_kernel = 'laplacian'
        self.multipole_krr_sigma = 10.0
        self.multipole_krr_lambda = 1e-3
        self.multipole_correct_charge = True
        self.multipole_ref_mtp = False
        self.multipole_ref_path = ""
        self.multipole_save_to_disk = False
        self.multipole_save_path = ""

        # Defaults for electrostatics
        self.elst_type = "damped_mtp"
        self.elst_damping_exponents = constants.elst_cp_exp 

        # Defaults for Induction
        self.indu_sr_params = constants.indu_sr_params
        self.indu_smearing_coeff = 0.5478502
        self.indu_omega = 0.75 
        self.indu_conv = 1e-5

        # Defaults for Exchange
        self.exch_int_params = constants.exch_int_params 

        # Defaults for dispersion/polarizability
        self.pol_scs_cutoff = 5.01451 
        self.disp_beta      = 2.40871
        self.disp_radius    = 0.57785
        self.pol_exponent   = 0.177346


        # load the options
        self.load_hirsh_options()
        self.load_atomic_density_options()
        self.load_multipole_options()
        self.load_elst_options()
        self.load_indu_options()
        self.load_exch_options()
        self.load_disp_options()

    def load_hirsh_options(self):
        
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

        # neighbors for hirsh training
        try:
            self.hirsh_max_neighbors = self.Config.getint("hirshfeld","max_neighbors")
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

        # Return type of kernel
        try:
            self.hirsh_svr_kernel =  self.Config.get("hirshfeld","svr_kernel")
        except:
            pass

        # Return sigma parameter for machinelearning
        try:
            self.hirsh_svr_C = self.Config.getfloat("hirshfeld","svr_C")
        except:
            pass

        # no default
        try:
            self.hirsh_filepath = self.Config.get("hirshfeld","ref_path")
        except:
            pass

        # no default
        try:
            self.hirsh_file_read = self.Config.get("hirshfeld","ref_hirsh")
        except:
            pass

        # Return sigma parameter for machinelearning
        try:
            self.hirsh_svr_epsilon = self.Config.getfloat("hirshfeld","svr_epsilon")
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

    def set_hirshfeld_svr_kernel(self, val):
        self.hirsh_svr_kernel = val

    def set_hirshfeld_svr_C(self, val):
        self.hirsh_svr_C = val

    def set_hirshfeld_file_read(self, val):
        self._hirsh_file_read = val

    def set_hirshfeld_filepath(self, val):
        self.hirsh_filepath = val

    def set_hirshfeld_svr_epsilon(self, val):
        self.hirsh_svr_epsilon = val

    ### Options for atomic density computations 
    def load_atomic_density_options(self):
        try:
            self.atomicdensity_training = self.Config.get("atomicdensity","training")
        except:
            pass

        try:
            self.atomicdensity_training_env = self.Config.get("atomicdensity","training_env")
        except:
            pass

        try:
            self.atomicdensity_max_neighbors = self.Config.getint("atomicdensity","max_neighbors")
        except:
            pass
        try:
            self.atomicdensity_max_neighbors_env = self.Config.getint("atomicdensity","max_neighbors_env")
        except:
            pass

        try:
            self.atomicdensity_krr_kernel = self.Config.get("atomicdensity","krr_kernel")
        except:
            pass

        try:
            self.atomicdensity_krr_sigma = self.Config.get("atomicdensity","krr_sigma")
        except:
            pass

        try:
            self.atomicdensity_krr_lambda = self.Config.get("atomicdensity","krr_lambda")
        except:
            pass

        try:
            self.atomicdensity_krr_sigma_env = self.Config.get("atomicdensity","krr_sigma_env")
        except:
            pass

        try:
            self.atomicdensity_krr_gamma_env = self.Config.get("atomicdensity","krr_gamma_env")
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

    def set_atomicdensity_training(self, val):
        self.atomicdensity_training = val

    def set_atomicdensity_training_env(self, val):
        self.atomicdensity_training_env = val

    def set_atomicdensity_max_neighbors(self, val):
        self.atomicdensity_max_neighbors = val

    def set_atomicdensity_max_neighbors_env(self, val):
        self.atomicdensity_max_neighbors_env = val

    def set_atomicdensity_krr_kernel(self, val):
        self.atomicdensity_krr_kernel = val

    def set_atomicdensity_krr_sigma(self, val):
        self.atomicdensity_krr_sigma = val

    def set_atomicdensity_krr_lambda(self, val):
        self.atomicdensity_krr_lambda = val

    def set_atomicdensity_krr_sigma_env(self, val):
        self.atomicdensity_krr_sigma_env = val

    def set_atomicdensity_krr_gamma_env(self, val):
        self.atomicdensity_krr_gamma_env = val

    def set_atomicdensity_ref_adens(self, val):
        self.atomicdensity_ref_adens = val 

    def set_atomicdensity_refpath(self, val):
        self.atomicdensity_refpath = val

    ### Options for multipoles

    def load_multipole_options(self):
        try:
            self.multipole_max_neighbors = self.Config.getint("multipoles","max_neighbors")
        except:
            pass

        try:
            self.multipole_training = self.Config.getint("multipoles","training")
        except:
            pass

        try:
            self.multipole_kernel = self.Config.get("multipoles","kernel")
        except:
            pass

        try:
            self.multipole_krr_sigma = self.Config.get("multipoles","krr_sigma")
        except:
            pass

        try:
            self.multipole_krr_lambda = self.Config.get("multipoles","krr_lambda")
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
            self.multipole_save_path = self.Config.get("multipoles","mtp_save_path")
        except:
            pass

    def set_multipole_max_neighbors(self, val):
        self.multipoles_max_neighbors = val

    def set_multipole_training(self, val):
        self.multipole_training = val

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

    ### Options for Electrostatics
    
    def load_elst_options(self):
        try:
            self.elst_type =  self.Config.get("electrostatics","type")
        except:
            return "damped_mtp"
        
        try:
            ret = {}
            at_types = ['Cl1', 'F1', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2',  'N3', 'N2', 'N1', 'O1', 'O2']  
            # Grab exponents from config if provided
            for at in at_types:
                ret[at] = self.Config.getfloat("electrostatics", "exp["+at+"]")
            self.elst_damping_exponents = ret
        except:
            pass

    def set_elec_type(self, val):
        self.elst_type = val
    
    def set_damping_exponents(self, val):
        self.elst_damping_exponents = val

    ### Options for Induction
    def load_indu_options(self):
            
        try:
            at_types = ['Cl1', 'F1', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2',  'N3', 'N2', 'N1', 'O1', 'O2']  
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
            at_types = ['Cl1', 'F1', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2',  'N3', 'N2', 'N1', 'O1', 'O2']  
            ret = {}
            # Grab exponents from config if provided
            for at in at_types:
                ret[at] = self.Config.getfloat("repulsion", "rep["+at+"]")
            self.exch_int_params = ret
        except:
            pass 

    def set_exchange_int_params(self, val):
            self.exch_int_params = val
            ret = constants.exch_int_params 

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

    def set_pol_scs_cutoff(self, val):
        self.pol_scs_cutoff = val

    def set_disp_beta(self, val):
        self.disp_beta = val

    def set_disp_radius(self, val):
        self.disp_radius = val

    def set_pol_exponent(self, val):
        self.pol_exponent = val

