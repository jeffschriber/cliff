#!/usr/bin/env python
#
# Utils Class. All sorts of random functions.
#
# Tristan Bereau (2017)

import numpy as np
import re
import logging
import math
import copy
import glob
import os
import cliff.helpers.constants as constants
from numba import jit

# Set logger
logger = logging.getLogger(__name__)

def file_finder(jobdir = None):
    """
    Compiles xyz files for multiple computations
    
    dirs points to directories, one for each computation
    """
    
    # check current directory if none is passed
    if jobdir == None:
        jobdir = './'

    # Gather directories in jobdir
    jobdirs = [ os.path.join(jobdir,name) for name in os.listdir(jobdir) if os.path.isdir(os.path.join(jobdir, name)) ]
    

    if len(jobdirs) == 0:
        raise Exception("Cannot find job directories!")
    else:
        print("    Found {} job directories".format(len(jobdirs)))        
        
    master_xyzs = []
    filesum = 0
    for n, jd in enumerate(jobdirs):
        filelist = glob.glob(jd + '/*.xyz')
        if len(filelist) == 0:
            print("WARNING: Job {} does not contain any files!".format(n))

        filesum += len(filelist)
        master_xyzs.append(filelist)

    if filesum == 0:
        raise Exception("Cannot find xyz files!")
    else:
        print("    Found {} xyz files".format(filesum))        


    return master_xyzs


def read_file(infile):
    'Read file and return content'
    try:
        f = open(infile, 'r')
        s = f.readlines()
        f.close()
    except IOError as e:
        print('Cannot open file %s' % infile)
        logger.error(e)
        exit(1)
    # Remove all empty header lines
    while len(str(s[0].rstrip())) == 0:
        s = s[1:]
    return s

def set_logger_level(level):
    logger.setLevel(level)

#@jit
def build_coulomb_matrix(coords, atom_types,
    central_atom_id, max_neighbors, direction=None):
    '''Build Coulomb matrix ordered by distance to central_atom_id'''
    # First sort list of atoms; then build distance matrix; then
    # invert elements; then multiply by numerator (atomic numbers)
    # Return upper triangular part of the (N,N) array
    #
    # First compute distance of all atoms to central_atom_id
    N = len(coords)
    cart_ord, atom_typ_ord, reorder = reorder_atoms(coords, atom_types,
        central_atom_id, max_neighbors)
    Z = extract_atomic_numbers(coords,atom_typ_ord)
    Z.resize(max_neighbors)
    # Compute distance matrix
    d = np.zeros((max_neighbors,max_neighbors))
    for i in range(min(N,max_neighbors)):
        for j in range(3):
            diff2 = cart_ord[i,j] - cart_ord[:,j]
            diff2 **= 2
            d[i,:] += diff2
    np.sqrt(d,d)
    for i in range(min(N,max_neighbors)):
        for j in range(min(N,max_neighbors)):
            if i != j:
                d[i,j] = Z[i]*Z[j]/d[i,j]
            else:
                d[i,i] = 0.5*Z[i]**(2.4)
    if direction is not None:
        # Optional orientation dot product
        for i in range(min(N,max_neighbors)):
            for j in range(min(N,max_neighbors)):
                if i != j:
                    d[i,j] *= np.sign(np.dot(
                        cart_ord[i,:]-cart_ord[j,:],direction))
    d[N:,:] = 0.0
    d[:,N:] = 0.0
    # Trying to reduce Coul Mat
    # d[3:,3:] = 0.0
    return d[np.triu_indices(max_neighbors)],reorder


def reorder_atoms(coords, atom_types, central_atom_id, max_neighbors):
    '''Reorder list of atoms from the central atom and by its distance'''
    distMain = sum([(coords[central_atom_id][j] - coords[:,j])**2
                    for j in range(3)])
    reorder = np.argsort(distMain)
    cart_ord = np.zeros((max_neighbors,3))
    for i in range(min(len(coords),max_neighbors)):
        cart_ord[i,:] = coords[reorder[i],:]
    atom_typ_ord = []
    for i in range(len(coords)):
        atom_typ_ord.append(atom_types[reorder[i]])
    return cart_ord, atom_typ_ord, reorder


def neighboring_vectors(coords, atom_types, central_atom_id):
    '''Returns neighboring pairwise vectors starting from central atom up
    to 3. If it's less than 3, complete by orthogonal vectors.'''
    cart_ord, atom_typ_ord, reorder = reorder_atoms(coords, atom_types,
                                                    central_atom_id, 4)
    ngb = len(cart_ord)-1
    ngb_vecs = np.zeros((3,3))
    if atom_types[central_atom_id] in ["H","O"]:
        for i in range(min(len(ngb_vecs),len(coords)-1)):
            ngb_vecs[i] = cart_ord[i+1]-cart_ord[0]
        ngb_vecs[2] = np.cross(ngb_vecs[0],ngb_vecs[1])
        ngb_vecs[1] = np.cross(ngb_vecs[2],ngb_vecs[0])
    else:
        # Rearrange atoms if we have two H atoms first
        # if len(atom_types) > 3 \
        #    and atom_typ_ord[1] == "H" \
        #    and atom_typ_ord[2] == "H":
        #     # Exchange #3 with #1
        #     cart_ord[1], cart_ord[3] = cart_ord[3], cart_ord[1]
        # Z-vector
        ngb_vecs[0] = cart_ord[1]-cart_ord[0]
        if len(coords) < 4:
            ngb_vecs[1] = cart_ord[2]-cart_ord[0]
        else:
            ngb_vecs[1] = cart_ord[3]-cart_ord[2]
        ngb_vecs[2] = np.cross(ngb_vecs[0],ngb_vecs[1])
        ngb_vecs[1] = np.cross(ngb_vecs[2],ngb_vecs[0])
    vec_all_dir = True
    for j in range(3):
        if np.linalg.norm(ngb_vecs[:,j]) < 1e-3:
            vec_all_dir = False
    for i in range(len(ngb_vecs)):
        if np.linalg.norm(ngb_vecs[i]) > 1e-6:
            ngb_vecs[i] /= np.linalg.norm(ngb_vecs[i])
    ngb_vecs.resize((3,3))
    return ngb_vecs, vec_all_dir

def extract_atomic_numbers(cartesian,atom_types):
    '''Extract atomic number from name of atom type'''
    # Parse atom type, then read from dictionary
    N = len(cartesian)
    Z = np.zeros(N)
    for i in range(N):
        name = re.sub('[0-9+-]','',atom_types[i])
        ele  = re.findall('[A-Z][^A-Z]*',name)[0]
        try:
            Z[i] = constants.atomic_number[ele]
        except:
            logger.error("Can't find element %s in dictionary." % ele)
            print("Can't find element %s in dictionary." % ele)
            exit(1)
    return Z

@jit
def spher_to_cart(quad, stone_convention=True):
    'convert spherical to cartesian multipoles'
    cart = np.zeros((3,3))
    if not stone_convention:
        cart[2,2] = 0.5*quad[0]
        cart[0,2] = cart[2,0] = 0.5/constants.sqrt_3*quad[1]
        cart[1,2] = cart[2,1] = 0.5/constants.sqrt_3*quad[2]
        cart[0,1] = cart[1,0] = 0.5/constants.sqrt_3*quad[4]
        cart[1,1] = -0.5/constants.sqrt_3*quad[3]
        cart[0,0] = +0.5/constants.sqrt_3*quad[3]
    else:
        cart[2,2] = quad[0]
        cart[0,2] = cart[2,0] = constants.sqrt_3*.5*quad[1]
        cart[1,2] = cart[2,1] = constants.sqrt_3*.5*quad[2]
        cart[0,1] = cart[1,0] = constants.sqrt_3*.5*quad[4]
        cart[1,1] = -.5*(constants.sqrt_3*quad[3] + quad[0])
        cart[0,0] =      constants.sqrt_3*quad[3] + cart[1,1]
    return cart

@jit
def cart_to_sphere(cart):
    'Convert cartesian to spherical multipoles'
    spher = np.zeros(5)
    # use Stone convention
    spher[0] = cart[2,2]
    spher[1] = 2./constants.sqrt_3*cart[0,2]
    spher[2] = 2./constants.sqrt_3*cart[1,2]
    spher[3] = 1./constants.sqrt_3*(cart[0,0]-cart[1,1])
    spher[4] = 2./constants.sqrt_3*cart[0,1]
    return spher

def slater_mbis(cell, coord_i, N_i, v_i, U_i, coord_j, N_j, v_j, U_j):  
    """
    Returns overlap of two atoms scaled by global parameters

    @params:

    """
    
    # Get interatomic distance
    vec = cell.pbc_distance(coord_i, coord_j)
    rij = np.linalg.norm(vec)
    # Call function externally to enable jit
    return U_i * U_j * slater_mbis_funcform(rij, N_i, v_i, N_j, v_j) 
    #return U_i * U_j * slater_mbis_funcform(rij, v_i, v_j) 

#@jit
#def slater_mbis_funcform(rij,vi, vj):
#    Bij = np.sqrt(1.0/(vi*vj))
#
#    return ((1./3)*Bij*Bij*rij*rij + Bij*rij + 1) * np.exp(-Bij*rij) 

@jit
def slater_mbis_funcform(rij, N_i, v_i, N_j, v_j):
    v_i2, v_j2 = v_i**2, v_j**2
    if abs(v_i-v_j) > 1e-4: # Original IPML uses 1e-3
        # regular formula
        g0ab = -4*v_i2*v_j2/(v_i2-v_j2)**3
        g1ab = v_i/(v_i2-v_j2)**2
        g0ba = -4*v_j2*v_i2/(v_j2-v_i2)**3
        g1ba = v_j/(v_j2-v_i2)**2
        return N_i*N_j/(8*np.pi*rij) * \
                            ((g0ab+g1ab*rij)*np.exp(-rij/v_i) \
                            + (g0ba+g1ba*rij)*np.exp(-rij/v_j))
    else:
        v = 0.5 * (v_i + v_j)
        rv = rij/v
        rv2, rv3, rv4 = rv**2, rv**3, rv**4
        exprv = np.exp(-rv)
        return N_i*N_j * \
            (1./(192*np.pi*v**3) * (3+3*rv+rv2) * exprv \
            + abs(v_j-v_i)/(384*v**4) * (-9-9*rv-2*rv2+rv3) * exprv \
            + (v_j-v_i)**2/(3840*v**5) * (90+90*rv+5*rv2-25*rv3+3*rv4) * exprv)



