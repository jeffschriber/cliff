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
        self.logger.setLevel(self.get_logger_level())
        util.set_logger_level(self.get_logger_level())

    def get_logger_level(self):
        # Return logger level
        try:
            return self.Config.getint("output", "log_level")
        except:
            return 50    

    ### Options for computing Hirshfeld ratios

    def get_hirshfeld_training(self):
        # Return training file for Hirshfeld ratios
        # No default when called
        return self.Config.get(
            "hirshfeld","training")

    def get_hirshfeld_max_neighbors(self):
        # Return maximum number of neighbors in Coulomb matrix
        try:
            return self.Config.getint("hirshfeld","max_neighbors")
        except:
            return 6

    def get_hirshfeld_krr_kernel(self):
        # Return type of kernel
        try:
            return self.Config.get("hirshfeld","krr_kernel")
        except:
            return 'laplacian'

    def get_hirshfeld_krr_sigma(self):
        # Return sigma parameter for machinelearning
        try:
            return self.Config.getfloat("hirshfeld","krr_sigma")
        except:
            return 1000.0

    def get_hirshfeld_krr_lambda(self):
        # Return sigma parameter for machinelearning
        try:
            return self.Config.getfloat("hirshfeld","krr_lambda")
        except:
            return 1e-9    

    def get_hirshfeld_svr_kernel(self):
        # Return type of kernel
        try:
            return self.Config.get("hirshfeld","svr_kernel")
        except:
            return "linear"

    def get_hirshfeld_svr_C(self):
        # Return sigma parameter for machinelearning
        try:
            return self.Config.getfloat("hirshfeld","svr_C")
        except:
            return 1.0

    def get_hirshfeld_file_read(self):
        # no default
        return self.Config.get(
            "hirshfeld","ref_hirsh")

    def get_hirshfeld_filepath(self):
        # no default
        return self.Config.get(
            "hirshfeld","ref_path")

    def get_hirshfeld_svr_epsilon(self):
        # Return sigma parameter for machinelearning
        try:
            return self.Config.getfloat("hirshfeld","svr_epsilon")
        except:
            return 0.05

    ### Options for atomic density computations 

    def get_atomicdensity_training(self):
        # Return location of training file
        # no default
        return self.Config.get(
            "atomicdensity","training")

    def get_atomicdensity_training_env(self):
        # Return location of training file
        try:
            return self.Config.get(
                "atomicdensity","training_env")
        except:
            return False

    def get_atomicdensity_max_neighbors(self):
        # Return location of training file
        try:
            return self.Config.get("atomicdensity","max_neighbors")
        except:
            return 12

    def get_atomicdensity_max_neighbors_env(self):
        # Return location of training file
        try:
            return self.Config.get("atomicdensity","max_neighbors_env")
        except:
            return 2

    def get_atomicdensity_krr_kernel(self):
        # Return location of training file
        try:
            return self.Config.get("atomicdensity","krr_kernel")
        except:
            return 'laplacian'

    def get_atomicdensity_krr_sigma(self):
        # Return location of training file
        try:
            return self.Config.get("atomicdensity","krr_sigma")
        except:
            return 1000.0

    def get_atomicdensity_krr_lambda(self):
        # Return location of training file
        try:
            return self.Config.get("atomicdensity","krr_lambda")
        except:
            return 1e-9

    def get_atomicdensity_krr_sigma_env(self):
        # Return location of training file
        try:
            return self.Config.get("atomicdensity","krr_sigma_env")
        except:
            return 1000.0

    def get_atomicdensity_krr_gamma_env(self):
        # Return location of training file
        try:
            return self.Config.get("atomicdensity","krr_gamma_env")
        except:
            return 1000.0

    def get_atomicdensity_ref_adens(self):
        try:
            val = self.Config.get("atomicdensity","ref_adens")
            if val in ["True", "true", "t", "1"]:
                return True
        except:
            return False

    def get_atomicdensity_refpath(self):
        # no default
        return self.Config.get("atomicdensity","ref_path")

    ### Options for multipoles

    def get_multipole_max_neighbors(self):
        try:
            return self.Config.get("multipoles","max_neighbors")
        except:
            return 12


    def get_multipole_training(self):
        # Return training file for multipoles
        return self.Config.get(
            "multipoles","training")

    def get_multipole_kernel(self):
        try:
            return self.Config.get("multipoles","kernel")
        except:
            return 'laplacian'

    def get_multipole_krr_sigma(self):
        try:
            return self.Config.get("multipoles","krr_sigma")
        except:
            return 10.0

    def get_multipole_krr_lambda(self):
        try:
            return self.Config.get("multipoles","krr_lambda")
        except:
            return 1e-3
    
    def get_multipole_krr_lambda(self):
        try:
            return self.Config.get("multipoles","krr_lambda")
        except:
            return 1e-3

    def get_multipole_krr_lambda(self):
        try:
            return self.Config.get("multipoles","krr_lambda")
        except:
            return 1e-3

    def get_multipole_correct_charge(self):
        try:
            val = self.Config.get("multipoles","correct_charge")
            if val in ["True","true","t","1"]:
                return True
            else:
                return False
        except:
            return True

    def get_multipole_ref_mtp(self):
        try:
            val = self.Config.get("multipoles","ref_mtp")
            if val in ["True","true","t","1"]:
                return True
            else:
                return False
        except:
            return False

    def get_multipole_ref_path(self):
        #no default
        return self.Config.get("multipoles","ref_path")

    def save_mtp_to_disk(self):
        try:
            val = self.Config.get("multipoles","save_to_disk")
            if val in ["True","true","t","1"]:
                return True
        except:
            return False
    def get_multipole_save_path(self):
        #no default
        return self.Config.get("multipoles","mtp_save_path")

    ### Options for Electrostatics
    
    def get_elec_type(self):
        try:
            return self.Config.get("electrostatics","type")
        except:
            return "damped_mtp"
    
    def get_damping_exponents(self):
        at_types = ['Cl1', 'F1', 'S1', 'S2', 'HS', 'HC', 'HN', 'HO', 'C4', 'C3', 'C2',  'N3', 'N2', 'N1', 'O1', 'O2']  
            
        ret = {}
        try:
            # Grab exponents from config if provided
            for at in at_types:
                ret[at] = self.Config.getfloat("electrostatics", "exp["+at+"]")
            return ret
        except:
            # Use the default ones           
            ret = constants.elst_cp_exp            
            return ret

    ###

    def get_mbd_beta(self):
        # Return beta coefficient in MBD
        return self.Config.getfloat(
            "polarizability","beta")

    def get_mbd_radius(self):
        # Return radius coefficient in MBD
        return self.Config.getfloat(
            "polarizability","radius")

    def get_pol_exponent(self):
        # Return exponent coefficient for anisotropic polarizability
        return self.Config.getfloat(
            "polarizability","exponent")

