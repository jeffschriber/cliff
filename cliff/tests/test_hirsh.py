
#!/usr/bin/env python

import cliff
from cliff.helpers.options import Options
from cliff.helpers.cell import Cell
from cliff.helpers.system import System
from cliff.atomic_properties.hirshfeld import Hirshfeld
import os
import pickle
import glob
import numpy as np

import cliff.tests as t
testpath = os.path.abspath(t.__file__).split('__init__')[0]


h_at = np.array([ 1.35089328e+00,-1.00033999e+01,-2.70953856e+00,-1.41385575e+01, 
 -1.64965434e+06, 8.51258356e+01, 4.33263644e+00,-2.06028723e+00,
 -9.95991265e+00,-6.04274711e+05, 5.93758471e+04,-3.55740990e+05,
 -3.36312810e+00,-1.32575836e+00, 1.45122277e-01, 2.38044972e-01,
  1.32559118e+00, 1.45666998e+00, 1.16145981e+01, 6.04290766e+05,
  1.59027930e+06, 3.55658093e+05,-1.20047377e+00,-7.42976025e-01,
  5.53265101e+01, 2.93665592e+00,-1.61305581e-01, 1.26352695e-01,
  5.52644074e+00,-2.11546512e+01,-1.91259135e+00, 7.24538349e+00,
 -4.79979027e+01,-3.69909523e+00, 3.21431654e-01, 2.69521468e-01,
  1.33139830e+00,-4.62663363e+00, 1.50076910e+01, 1.16349707e+00,
 -3.21007935e-01,-5.64219180e+00,-1.35696703e+00,-1.00147947e+00,
 -1.72146411e+00,-7.89556830e-01, 1.06246425e+00, 6.91806469e+00,])

def test_hirsh_ml():
    
    options = Options(testpath + '/hirsh_train/config.ini')
    # 1. Load hirshfeld with options in config
    hirsh = Hirshfeld(options)
    with open(testpath + "/../models/mbtypes.pkl", 'rb') as f:
        hirsh.mbtypes = pickle.load(f)
    
    # 2. Parse reference data and initialize training sets
    pathlist = sorted(glob.glob(testpath + 'hirsh_train/*.xyz'))

    # 3. Build the descriptors. This is done using
    ##   a Coulomb matrix representation where the number of
    ##   nearest neighbors is defined in the config file
    for path in pathlist:
        mol = System(options, path)
        ref_hirsh = []
        with open(path, 'r') as p:
            for line in p:
                line = line.split()
                if len(line) == 6:
                    ref_hirsh.append(float(line[4]))

        hirsh.add_mol_to_training(mol, ref_hirsh, atom='C')
    
    # 4. Train the model using kernel ridge regression
    hirsh.train_ml()
    res = np.abs(np.divide(np.subtract(h_at,hirsh.alpha_train['C']), h_at))

    # avg rel error
    res = np.sum(res) / len(res)   

    assert res < 1e-6
