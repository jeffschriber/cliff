#!/usr/bin/env python
#
# Calculator class. Initialize calculator.
#
# Tristan Bereau (2017)

import logging
import utils
import configparser

class Calculator:
    'Main calculator'
    # Config parser
    Config = configparser.ConfigParser()
    Config
    # Set logger
    logger = logging.getLogger(__name__)
    logging.basicConfig()

    def __init__(self, config_file="config.ini"):
        # Load config file
        self.Config.read(config_file)
        # Logger level
        self.logger.setLevel(self.get_logger_level())
        utils.set_logger_level(self.get_logger_level())
        self.energy = 0.0

    def get_logger_level(self):
        # Return logger level
        return self.Config.getint("output", "log_level")

    def get_hirshfeld_training(self):
        # Return training file for Hirshfeld ratios
        return self.Config.get(
            "hirshfeld","training")

    def get_hirshfeld_max_neighbors(self):
        # Return maximum number of neighbors in Coulomb matrix
        return self.Config.getint(
            "hirshfeld","max_neighbors")

    def get_hirshfeld_krr_kernel(self):
        # Return type of kernel
        return self.Config.get(
            "hirshfeld","krr_kernel")

    def get_hirshfeld_krr_sigma(self):
        # Return sigma parameter for machinelearning
        return self.Config.getfloat(
            "hirshfeld","krr_sigma")

    def get_hirshfeld_krr_lambda(self):
        # Return sigma parameter for machinelearning
        return self.Config.getfloat(
            "hirshfeld","krr_lambda")

    def get_hirshfeld_svr_kernel(self):
        # Return type of kernel
        return self.Config.get(
            "hirshfeld","svr_kernel")

    def get_hirshfeld_svr_C(self):
        # Return sigma parameter for machinelearning
        return self.Config.getfloat(
            "hirshfeld","svr_C")

    def get_hirshfeld_file_read(self):
        return self.Config.get(
            "hirshfeld","ref_hirsh")

    def get_hirshfeld_filepath(self):
        return self.Config.get(
            "hirshfeld","ref_path")

    def get_hirshfeld_svr_epsilon(self):
        # Return sigma parameter for machinelearning
        return self.Config.getfloat(
            "hirshfeld","svr_epsilon")

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

    def get_multipole_training(self):
        # Return training file for multipoles
        return self.Config.get(
            "multipoles","training")
