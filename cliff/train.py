#! /usr/bin/env python


"""
train.py
Component-based Learned Intermolecular Force Field

Contains functions for training ML models for atomic
properties

"""
import os
import numpy as np
import glob
import qcelemental as qcel

using_apnet = True
try:
    import apnet
except:
    using_apnet = False


from cliff.helpers.options import Options
from cliff.helpers.system import System
from cliff.atomic_properties.hirshfeld import Hirshfeld
from cliff.atomic_properties.atomic_density import AtomicDensity
from cliff.atomic_properties.multipole import Multipole

def train_atomic_properties(reference_properties, train_fraction, save_file):
    """
    Trains NN-based model for atomic properties using APNET

    Parameters
    ----------

    reference_properties : :class: `str`
        Location of .pkl file containing reference properties. The pkl file
        contains a pandas dataframe with the following columns:
            - 'Z' : The atom types (:class: `~numpy.ndarray` of int with shape (n,))
            - 'R' : Cartesian coordinates (:class: `~numpy.ndarray` of float with shape (n,3)
            - 'total_charge' : The total monomer charge (int)
            - 'cartesian_multipoles' : Reference atomic charges, dipoles, and quadrupoles (:class: `~numpy.ndarray` of `float` with shape (n,10), in a.u.)
            - 'volume_ratios' : Reference volume ratios (:class: `~numpy.ndarray` of `float` with shape (n,), no units)
            - 'valence_widths' : Reference valence widths (:class: `~numpy.ndarray` of `float` with shape (n,))
    train_fraction : :class: `float`
        Fraction of data set in reference_properties to use for training. Default is 0.9 
    save_file : :class: `str`
        location and filename of ML model. Needs to end in .h5, or else code will enforce.
    """
    if using_apnet:
        monomers, multipoles, ratios, widths = apnet.load_monomer_pickle(reference_properties)
        
        # randomly shuffle the monomers
        N = len(monomers)
        inds = np.arange(N)
        np.random.seed(4201)
        np.random.shuffle(inds)
            
        # 90% of the monomers are used for training
        Nt = int(train_fraction * N)
        indst = inds[:Nt]
        monomers_t = [monomers[i] for i in indst]
        multipoles_t = multipoles[indst]
        ratios_t = ratios[indst]
        widths_t = widths[indst]
        
        # the other 10% are used for validation
        indsv = inds[Nt:]
        monomers_v = [monomers[i] for i in indsv]
        multipoles_v = multipoles[indsv]
        ratios_v = ratios[indsv]
        widths_v = widths[indsv]
        
        # train the property model and save weights to .h5 file
        if not save_file.endswith(".h5"):
            save_file += ".h5"

        apnet.train_cliff_model(monomers_t, multipoles_t, ratios_t, widths_t, monomers_v, multipoles_v, ratios_v, widths_v, save_file)
    else:
        raise Exception("Calling train_atomic_properties but APNET not found!")
    

def get_mols(pathname, frac, max_test=None):

    # 1. get list of xyzs
    mol_list = glob.glob(pathname + "/*.xyz")
    
    # 3. Split geometries into training and testing sets
    nmol = len(mol_list)
    train_size = int(round(frac*nmol))
    test_size = nmol - train_size
    if max_test is not None:
        test_size = max_test

    shuffle(pathlist)
    train = pathlist[:train_size]
    test = pathlist[train_size: (train_size + test_size)]
    return train, test

def run_multipole_krr(ref_file, train_xyzs, ele=None, save=None,
                hyperparams=None,frac=None, cutoff=None):
    
    options.set_multipole_rcut(cutoff)
    mtp_model = Multipole(options)
    # Load mbtypes
    with open("../mbtypes.pkl", 'rb') as f:
        mtp_model.mbtypes = pickle.load(f)
       # print("# Loaded mbtypes")
    if hyperparams != None:
        assign_params(hyperparams, mtp_model)
        assign_params(hyperparams, mtp_model)
    ele_set = set([])
    tr_file = open(ele + "_" + str(frac) +  "_training.txt",'w')
    for xyz in xyz_training:

        mtp = ref_file[xyz]
    
        try:
            mol1 = System(options,xyz=xyz)
        except:
            print("Skipping mol  ", xyz)
            continue

        mtp_model.add_mol_to_training(mol1, mtp, ele, xyz=xyz)
    tr_file.close()
    mtp_model.train_mol()
    if save != None:
        mtp_model.save_ml("models/" + save)
    if ele is not None:
        print("# Training:",len(mtp_model.target_train[ele]),"atoms,",mtp_model.num_mols_train[ele],"molecules")
    else:
        for e in ["H","C","O","N","S", "Cl", "F","Br"]:
            print("# Training:",len(mtp_model.target_train[e]),"atoms,",mtp_model.num_mols_train[e],"molecules")
    return mtp_model
    
    

def train_multipole_krr(pathname, ref_file, frac_training,options, element=None, save=None, load=None, max_test=None, cutoff=None):
    """Run training/prediction with frac_training ratio"""
    train_xyzs, test_xyzs  = get_mols(pathname, frac_training, max_test)
    mtp_model = run_multipole_krr(ref_file, train_xyzs, element, save, load, None, frac_training, cutoff)
    mae_avg = predict_ml(mtp_model, xyz_predict,frac_training, ele)

def train_hirshfeld_krr(pathname, ref_file, frac, ele, max_test=None):

    train, test = get_mols(pathname, frac_training, max_test)

    # 1. Load hirshfeld with options in config
    options.set_hirshfeld_cutoff(cutoff)
    hirsh = Hirshfeld(options)
    with open("../models/mbtypes.pkl", 'rb') as f:
        hirsh.mbtypes = pkl.load(f)
 
    ntrain = 0
    for path in train:
        mol = System(options,path)
        ref_hirsh = ref_file[path]
        ntrain += hirsh.add_mol_to_training(mol, ref_hirsh, atom=ele)

    h_outfile = open(ele + "_"+str(100*frac)+ "_" + str(cutoff) + "_h.dat", 'w')
    ostr = ''

    ostr += "Training set size: " + str(len(train)) 
    ostr += "\nTesting set size:  " + str(len(test)) 
    
    # 4. Train the model using kernel ridge regression
    hirsh.train_ml()
    
    # 5. Save the model
    save_file = "hirshfeld_model_" + str(ele) + "_" + str(frac) + "_" + str(cutoff) + ".pkl"
    hirsh.save_ml(save_file)

    # Test the model
    mae = 0.0
    rms = 0.0
    hirsh_test = []
    for path in test:
        test_mol = System(options,path)
        h_val = ref_file[path]

        if ele not in test_mol.elements:
            continue
        hirsh.predict_mol(test_mol, force_test=True)
        idx = 0

        for i in range(len(test_mol.hirshfeld_ratios)):
            if test_mol.elements[i] == ele:
                hirsh_test.append( (test_mol.elements[i], test_mol.hirshfeld_ratios[i], h_val[idx]) )
                mae += abs(test_mol.hirshfeld_ratios[i] - h_val[idx])
                rms += np.square(test_mol.hirshfeld_ratios[i] - h_val[idx])
                idx += 1
        ntest += idx

        if ntest >= max_test:
            break
    h_outfile.write( "\nTesting set size:  " + str(ntest)) 
 
    h_outfile.write("\nHirshfeld Ratios")
    for pair in hirsh_test:
        h_outfile.write(f"\n{pair[0]} {pair[1]} {pair[2]}")

    h_outfile.close()

    mae /= ntest
    rms = np.sqrt(rms/ntest)
    print(options.hirsh_krr_sigma, options.hirsh_krr_lambda, mae, rms)
    return rms


def train_atomic_volume_krr(pathname, ref_file, frac_training,options, element=None, save=None, load=None, max_test=None, cutoff=None):

    train, test = get_mols(pathname, frac_training, max_test)
    # 1. Load the config to initialize atomic density class
    options.set_atomicdensity_cutoff(cutoff)
    adens = AtomicDensity(options) 
    #pop_test = []
    width_test = []    
    data_dir = "../multipoles_and_valence_parameters/"
    o1 = open(ele + '_'+str(frac*100)+'_' + str(cutoff) + '_w.dat', 'w')

    with open("models/mbtypes.pkl", 'rb') as f:
        adens.mbtypes = pkl.load(f)
    # 4. Get reference atomic widths and populations
    ntrain = 0
    for path in train:
       # print(path)
        mol = System(options,path)
        width = ref_file[path] 
        ntrain += adens.add_mol_to_training(mol, width, atom=ele)
    
    if ntrain == 0:
        print(f"Cannot find atom type {ele} in training set given!")
        exit()

    o1.write( "Training set size: " + str(ntrain)) 

    # 5. Train the model, KRR only options
    adens.train_ml()
    
    # 6. Save the model
    savefile = "atomic_width_" +ele+"_"+ str(frac) + "_" + str(cutoff) + "_cliff.pkl"
    adens.save_ml(savefile)

    # 7. Test the model
    ntest = 0
    mae = 0.0
    rms = 0.0
    for path in test:
        test_mol = System(options,path)

        if ele not in test_mol.elements:
            continue
        adens.predict_mol(test_mol)
        width = ref_file[path] 

        idx = 0
        for i in range(len(test_mol.valence_widths)):
            if test_mol.elements[i] == ele:
                width_test.append((test_mol.elements[i], test_mol.valence_widths[i], width[idx]))
                mae += abs(test_mol.valence_widths[i] - width[idx])
                rms += np.square(test_mol.valence_widths[i] - width[idx])
                idx += 1
        ntest += idx

        if ntest >= max_test:
            break

    o1.write( "\nTesting set size:  " + str(ntest)) 
    o1.write("\nWidths")
    for pair in width_test:
        o1.write(f"\n{pair[0]} {pair[1]} {pair[2]}")

    o1.close()

    mae /= ntest
    rms = np.sqrt(rms/ntest)
    print(options.atomicdensity_krr_sigma, options.atomicdensity_krr_lambda, mae, rms)
    return rms

