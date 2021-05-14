"""
Unit and regression test for the cliff package.
"""

# Import package, test suite, and other packages as needed
import cliff
import pytest
import sys
import os

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]

def test_cliff_imported():
    """Sample test, will always pass so long as import statement worked"""
    print("Test import cliff")
    
    assert "cliff" in sys.modules

#def test_cliff_dimer_runscript():
#    """Test execution of run_cliff.py"""
#    import cliff.run_cliff as rc
#    
#    ret = rc.main(dimer = testpath + '/dimer_data/', name='test' )
#    labels = ret[0]
#    energy = ret[1]
#    ref = {'NBC-4-A' :[-2.14429792,  3.35565678, -0.40030537, -3.90665939],
#           'S66-1-A' :[-7.35151536,  6.35877881, -1.99789796, -2.14624461],
#           'NBC-13-A':[-1.76418241,  2.22211712, -0.27161155, -2.86988216],
#           'S66-10-A':[-4.86971584,  7.62044375, -2.18159367, -3.27760226]}
#
#    for lab,en in zip(labels,energy):
#        l = lab[0]
#        ref_e = ref[l]
#        assert abs(ref_e[0] - en[0]) < 1e-5 
#        assert abs(ref_e[1] - en[1]) < 1e-5 
#        assert abs(ref_e[2] - en[2]) < 1e-5 
#        assert abs(ref_e[3] - en[3]) < 1e-5 
#    
#def test_cliff_mon_runscript():
#    import cliff.run_cliff as rc
#    ret = rc.main(monA = testpath + '/monomer_data/monomerA.xyz', monB = testpath + '/monomer_data/monomerB.xyz', name='test',units='angstrom')
#
#    ref = [[-2.12674195,  1.94681627, -0.38330545, -1.34216795],
#           [-1.44042857,  1.08849426, -0.22083751, -1.02014158],
#           [-0.80034005,  0.43990854, -0.09846555, -0.67700809]]
#
#    labels = ret[0]
#    energy = ret[1]
#    
#    for lab,en in zip(labels,energy):
#        l = int(lab[1])
#        ref_e = ref[l]
#        assert abs(ref_e[0] - en[0]) < 1e-5 
#        assert abs(ref_e[1] - en[1]) < 1e-5 
#        assert abs(ref_e[2] - en[2]) < 1e-5 
#        assert abs(ref_e[3] - en[3]) < 1e-5 
#
#
#
