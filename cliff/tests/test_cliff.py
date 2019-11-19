"""
Unit and regression test for the cliff package.
"""

# Import package, test suite, and other packages as needed
import cliff
import pytest
import sys

def test_cliff_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cliff" in sys.modules
