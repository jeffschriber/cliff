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

    print("Testing io")
    refs = np.array([[-2.12674195,  1.94681627, -0.38330545, -1.34216795],[-1.44042857,  1.08849426, -0.22083751, -1.02014158],[-0.80034005,  0.43990854, -0.09846555, -0.67700809]])


    monomerA = testpath + "/monomer_data/monomerA.xyz"
    monomerB = testpath + "/monomer_data/monomerB.xyz"
    
    save_path = testpath + "/monomer_data/atomic_data/"

    print(save_path)

    monA = cliff.load_monomer_xyz(monomerA)[0]
    monB = cliff.load_monomer_xyz(monomerB)

    print(monA,monB)
    #energies_1 = cliff.predict_from_monomer_list(monA,monB)
    
    options = Options()
    models = cliff.load_krr_models(options)

    print("models loaded")

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
        assert (ref[0] - en[0]) < 1e-5        
        assert (ref[1] - en[1]) < 1e-5        
        assert (ref[2] - en[2]) < 1e-5        
        assert (ref[3] - en[3]) < 1e-5        


