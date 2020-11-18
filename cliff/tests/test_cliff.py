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
    
    ret = rc.main(testpath + '/cliff_test/config.ini', testpath + '/s18/xyzs/', name='test' )
    with open(testpath + 's18/xyzs/s18_test.json','r') as f:
        refs = json.load(f)
        # Make sure values match up
        for k,v in ret.items():
            r = refs[k]
            assert abs(v[0] - r['elst']) < 1e-5 
            assert abs(v[1] - r['exch']) < 1e-5 
            assert abs(v[2] - r['indu']) < 1e-5 
            assert abs(v[3] - r['disp']) < 1e-5 
            assert abs(v[4] - r['total']) < 1e-5 
    
