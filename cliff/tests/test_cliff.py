"""
Unit and regression test for the cliff package.
"""

# Import package, test suite, and other packages as needed
import cliff
import pytest
import sys
import json
import os

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]

def test_cliff_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cliff" in sys.modules

def test_cliff_runscript():
    """Test execution of run_cliff.py"""
    import cliff.run_cliff as rc
    
    ret = rc.main(testpath + '/cliff_test/config.ini', testpath + '/s30/xyzs/' )
    with open(testpath + 's30/s30_ref_TT.json','r') as f:
        refs = json.load(f)

        rc.print_ret(refs)

        # Make sure values match up
        for k,v in ret.items():
            r = refs[k]
            assert abs(v[0] - r[0]) < 1e-5 
            assert abs(v[1] - r[1]) < 1e-5 
            assert abs(v[2] - r[2]) < 1e-5 
            assert abs(v[3] - r[3]) < 1e-5 
            assert abs(v[4] - r[4]) < 1e-5 
            #if abs(v[2] - r[2]) > 1e-5:
            #    print(k, abs(v[2] - r[2]))# < 1e-5 
            #print(abs(v[1] - r[1]))# < 1e-5 #fail 
            #print(abs(v[2] - r[2]))# < 1e-5 #fail 
            #print(abs(v[3] - r[3]))# < 1e-5 
            #print(abs(v[4] - r[4]))# < 1e-5 
    
#test_cliff_runscript()
