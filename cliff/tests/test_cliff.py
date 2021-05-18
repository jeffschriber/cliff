"""
Unit and regression test for the cliff package.
"""

# Import package, test suite, and other packages as needed
import cliff
import pytest
import sys
import os
import numpy as np
import glob

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]

def test_cliff_imported():
    """Sample test, will always pass so long as import statement worked"""
    print("Test import cliff")
    
    assert "cliff" in sys.modules

#def test_cliff_dimers():
#    dimer_xyz = glob.glob(testpath + "/dimer_data/*.xyz")
#    dimers = [cliff.load_dimer_xyz(f) for f in dimer_xyz]
#    energies = cliff.predict_from_dimers(dimers, infile=testpath + '/config.ini')
#
#    refs = np.asarray([[-2.36940907, 3.35565678,-0.45133031,-3.90665939],
#                       [-4.21149822, 6.35877881,-1.22398329,-2.14624461],
#                       [-1.62012678, 2.22211712,-0.25787379,-2.86988216],
#                       [-4.08143058, 7.62044375,-1.97118627,-3.27760226]])
#
#    for n in range(4):
#        ref = refs[n]
#        en  = energies[n]
#        assert abs(ref[0] - en[0]) < 1e-5        
#        assert abs(ref[1] - en[1]) < 1e-5        
#        assert abs(ref[2] - en[2]) < 1e-5        
#        assert abs(ref[3] - en[3]) < 1e-5        
#

def test_cliff_monomers():

    refs = np.array([[-1.57184919,  1.94681627, -0.28760022, -1.34216795],[-0.99218904,  1.08849426, -0.15499095, -1.02014158],[-0.48795822,  0.43990854, -0.05996322, -0.67700809]])
    
    monomerA = testpath + "/monomer_data/monomerA.xyz"
    monomerB = testpath + "/monomer_data/monomerB.xyz"

    monA = cliff.load_monomer_xyz(monomerA)[0]
    monB = cliff.load_monomer_xyz(monomerB)

    energies = cliff.predict_from_monomer_list(monA,monB, infile=testpath +'/config.ini')

    for n in range(3):
        ref = refs[n]
        en  = energies[n]
        assert abs(ref[0] - en[0]) < 1e-5        
        assert abs(ref[1] - en[1]) < 1e-5        
        assert abs(ref[2] - en[2]) < 1e-5        
        assert abs(ref[3] - en[3]) < 1e-5        

#def test_cliff_dimer_runscript():
#    """Test execution of run_cliff.py"""
#    import cliff.run_cliff as rc
#    
#    ret = rc.main(dimer = testpath + '/dimer_data/', name='test' )
#    labels = ret[0]
#    energy = ret[1]
#    ref = {'NBC-4-A' :[-2.36940907,  3.35565678, -0.45133031, -3.90665939],
#           'S66-1-A' :[-4.21149822,  6.35877881, -1.22398329, -2.14624461],
#           'NBC-13-A':[-1.62012678,  2.22211712, -0.25787379, -2.86988216],
#           'S66-10-A':[-4.08143058,  7.62044375, -1.97118627, -3.27760226]}
#
#
#
#    for lab,en in zip(labels,energy):
#        l = lab[0]
#        ref_e = ref[l]
#        assert abs(ref_e[0] - en[0]) < 1e-5 
#        assert abs(ref_e[1] - en[1]) < 1e-5 
#        assert abs(ref_e[2] - en[2]) < 1e-5 
#        assert abs(ref_e[3] - en[3]) < 1e-5 
    
#def test_cliff_mon_runscript():
#    import cliff.run_cliff as rc
#    ret = rc.main(monA = testpath + '/monomer_data/monomerA.xyz', monB = testpath + '/monomer_data/monomerB.xyz', name='test',units='angstrom')
#    ref = [[-1.57184919,  1.94681627, -0.28760022, -1.34216795], 
#           [-0.99218904,  1.08849426, -0.15499095, -1.02014158],
#           [-0.48795822,  0.43990854, -0.05996322, -0.67700809]]
#    labels = ret[0]
#    energy = ret[1]
#    for lab,en in zip(labels,energy):
#        l = int(lab[1])
#        ref_e = ref[l]
#        assert abs(ref_e[0] - en[0]) < 1e-5 
#        assert abs(ref_e[1] - en[1]) < 1e-5 
#        assert abs(ref_e[2] - en[2]) < 1e-5 
#        assert abs(ref_e[3] - en[3]) < 1e-5 

