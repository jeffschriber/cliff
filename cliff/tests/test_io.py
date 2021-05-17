"""
Unit and regression test for the cliff package.
"""

# Import package, test suite, and other packages as needed
import cliff
import pytest
import sys
import os
from cliff.helpers.options import Options
import numpy as np


import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]


def test_cliff_io():

    refs = np.array([[-1.57184919,  1.94681627, -0.28760022, -1.34216795],[-0.99218904,  1.08849426, -0.15499095, -1.02014158],[-0.48795822,  0.43990854, -0.05996322, -0.67700809]])


    monomerA = testpath + "/monomer_data/monomerA.xyz"
    monomerB = testpath + "/monomer_data/monomerB.xyz"
    
    save_path = testpath + "/monomer_data/atomic_data/"

    monA = cliff.load_monomer_xyz(monomerA)[0]
    monB = cliff.load_monomer_xyz(monomerB)

    options = Options()
    options.set_multipole_training(testpath + '/../models/small/mtp')
    models = cliff.load_krr_models(options)

    sysa = cliff.mol_to_sys(monA, options)
    sysa = cliff.predict_atomic_properties(sysa,models)    
    sysb_list = []
    for mb in monB:
        sysb = cliff.mol_to_sys(mb, options)
        sysb = cliff.predict_atomic_properties(sysb,models)    

        sysb_list.append(sysb)


    cliff.save_atomic_properties(sysa,save_path)
    for sb in sysb_list:
        cliff.save_atomic_properties(sb,save_path)



    monA = cliff.load_monomer_xyz(monomerA)
    monB = cliff.load_monomer_xyz(monomerB)
    energies = cliff.predict_from_monomer_list(monA,monB,load_path=save_path)

    for n in range(3):
        ref = refs[n]
        en  = energies[n]
        assert abs(ref[0] - en[0]) < 1e-5        
        assert abs(ref[1] - en[1]) < 1e-5        
        assert abs(ref[2] - en[2]) < 1e-5        
        assert abs(ref[3] - en[3]) < 1e-5        


