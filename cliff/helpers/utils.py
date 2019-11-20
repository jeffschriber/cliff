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
import cliff.helpers.constants as constants
from numba import jit

# Set logger
logger = logging.getLogger(__name__)

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

def merge_two_dicts(x, y):
    """Merge two dictionaries."""
    z = x.copy()
    z.update(y)
    return z

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

def build_coulomb_matrix_env(coords, atom_types,
    central_atom_id, max_neighbors, coords_env, attyp_env):
    '''Build Coulomb matrix between central_atom_id and coords_env'''
    # Consider coordinates of central_atom_id and coords_env
    coords_sys = np.array([coords[central_atom_id]])
    coords_sys = np.append(coords_sys, coords_env, axis=0)
    attyp_sys  = np.array([atom_types[central_atom_id]])
    attyp_sys  = np.append(attyp_sys, attyp_env, axis=0)
    # Delete any duplicate atom
    dupl = []
    for i in range(1,len(coords_sys)):
        # Remove any element in coords
        for j in range(len(coords)):
            d = np.linalg.norm(coords_sys[i] - coords[j])
            if d < 1e-3:
                dupl.append(i)
    for m in reversed(dupl):
        coords_sys = np.delete(coords_sys, m, 0)
        attyp_sys  = np.delete(attyp_sys, m,0)

    return build_coulomb_matrix(coords_sys, attyp_sys, 0, max_neighbors)[0]


def coulomb_with_grads(coords, atom_types,
    central_atom_id, max_neighbors):
    """Build Coulomb matrix with first and second derivatives."""
    N = len(coords)
    cart_ord, atom_typ_ord, reorder = reorder_atoms(coords, atom_types,
        central_atom_id, max_neighbors)
    Z = extract_atomic_numbers(coords,atom_typ_ord)
    Z.resize(max_neighbors)
    # Compute distance matrix
    d0 = np.zeros((max_neighbors,max_neighbors))
    # Matrix of first derivatives
    d1 = np.zeros((3,max_neighbors,max_neighbors))
    # Matrix of second derivatives
    d2 = np.zeros((9,max_neighbors,max_neighbors))
    for i in range(min(N,max_neighbors)):
        for j in range(3):
            diff2 = cart_ord[i,j] - cart_ord[:,j]
            d1[j,i,:] = -diff2
            diff2 **= 2
            d0[i,:] += diff2
    np.sqrt(d0,d0)
    for i in range(min(N,max_neighbors)):
        for j in range(min(N,max_neighbors)):
            if i != j:
                for k in range(3):
                    d1[k,i,j] = -Z[i]*Z[j]*d1[k,i,j]/d0[i,j]**3
                    for l in range(3):
                        # 2nd derivative
                        d2[3*k+l,i,j] = 3*(cart_ord[j,k]-cart_ord[i,k]) * (cart_ord[j,l]-cart_ord[i,l])
                        if k == l:
                            d2[3*k+l,i,j] -= d0[i,j]
                        d2[3*k+l,i,j] *= Z[i]*Z[j]/d0[i,j]**5
                d0[i,j] = Z[i]*Z[j]/d0[i,j]
            else:
                d0[i,i] = 0.5*Z[i]**(2.4)
                for k in range(3):
                    d1[k,i,i] = 0.5*Z[i]**(2.4)
                    for l in range(3):
                        d2[3*k+l,i,j] = 0.5*Z[i]**(2.4)
    d0[N:,:] = 0.0
    d0[:,N:] = 0.0
    d1[:,N:,:] = 0.0
    d1[:,:,N:] = 0.0
    d2[:,N:,:] = 0.0
    d2[:,:,N:] = 0.0
    # Matrix of second derivatives in spherical coordinates
    s2 = np.zeros((5,max_neighbors,max_neighbors))
    s2[0,:,:] = d2[8,:,:]
    s2[1,:,:] = 2./math.sqrt(3.)*d2[2,:,:]
    s2[2,:,:] = 2./math.sqrt(3.)*d2[5,:,:]
    s2[3,:,:] = 1./math.sqrt(3.)*(d2[0,:,:]-d2[4,:,:])
    s2[4,:,:] = 2./math.sqrt(3.)*d2[1,:,:]
    return [d0[np.triu_indices(max_neighbors)]], \
            [d1[j,:,:][np.triu_indices(max_neighbors)] for j in range(3)], \
            [s2[j,:,:][np.triu_indices(max_neighbors)] for j in range(5)], \
            reorder


def coulomb_matrix_com(coords, elements, max_neighbors, ordered=False):
    """Build coulomb matrix centered at the center of mass."""
    # Index masses
    masses = np.array([float(constants.atomic_weight[ele])
                    for _,ele in enumerate(elements)])
    # Center of mass
    com = np.sum([m*c for m,c in zip(masses,coords)],axis=0)/np.sum(masses)
    N = len(coords)
    cart_ord, atom_typ_ord, reorder = reorder_atoms_com(coords, elements,
        com, max_neighbors, ordered)
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
    d[N:,:] = 0.0
    d[:,N:] = 0.0
    # print d
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

def reorder_atoms_com(coords, atom_types, com, max_neighbors, ordered=False):
    '''Reorder list of atoms from the center of mass and by its distance'''
    distMain = sum([(com[j] - coords[:,j])**2
                    for j in range(3)])
    reorder = np.argsort(distMain)
    cart_ord = np.zeros((max_neighbors,3))
    for i in range(min(len(coords),max_neighbors)):
        cart_ord[i,:] = coords[reorder[i],:]
    atom_typ_ord = []
    for i in range(len(coords)):
        atom_typ_ord.append(atom_types[reorder[i]])
    if ordered:
        Z = extract_atomic_numbers(cart_ord,atom_typ_ord)
        if max_neighbors > 1:
            if Z[1] > Z[0]:
                # Swap 1 and 0
                cart_ord[0], cart_ord[1] = cart_ord[1], cart_ord[0].copy()
                atom_typ_ord[0], atom_typ_ord[1] = atom_typ_ord[1], atom_typ_ord[0]
                reorder[0], reorder[1] = reorder[1], reorder[0]
        if max_neighbors > 3:
            # Keep #2 and 3 close to either #0 or 1, depending on the distance
            if np.linalg.norm(cart_ord[2,:]-cart_ord[0,:]) > \
                np.linalg.norm(cart_ord[2,:]-cart_ord[1,:]):
                # Swap 2 and 3
                cart_ord[3], cart_ord[2] = cart_ord[2], cart_ord[3].copy()
                atom_typ_ord[3], atom_typ_ord[2] = atom_typ_ord[2], atom_typ_ord[3]
                reorder[3], reorder[2] = reorder[2], reorder[3]
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

def find_orthogonal_vec(vec):
    """Find a vector orthogonal to vec"""
    aligned = True
    vec_tmp = np.ones(3)
    while aligned is True:
        for i in range(3):
            vec_tmp[i] += np.random.rand()
        vec_tmp /= np.linalg.norm(vec_tmp)
        # 0 vector -> any orthogonal vector will do
        if np.linalg.norm(vec) < 1e-8:
            aligned = False
        elif abs(np.dot(vec_tmp,vec)/np.linalg.norm(vec) - 1.0) > 0.1:
            aligned = False
    return np.cross(vec,vec_tmp)

def build_bag_of_bonds(cartesian, atom_types, main_atom_id, bob_struct,
    max_neighbors):
    '''Build bag of bonds descriptor ordered by distance to mainAtomID'''
    # First compute distance of all atoms to mainAtomID
    N = len(cartesian)
    dist_main = np.zeros(N)
    for j in range(3):
        diff2 = cartesian[main_atom_id,j] - cartesian[:,j]
        diff2 **= 2
        dist_main += diff2
    reorder = np.argsort(dist_main)
    cart_ord = np.zeros((N,3))
    for i in range(N):
        cart_ord[i,:] = cartesian[reorder[i],:]
    atom_typ_ord = []
    for i in range(N):
        atom_typ_ord.append(atom_types[reorder[i]])
    Z = extract_atomic_numbers(cartesian,atom_typ_ord)
    # BoB descriptor
    bob_desc = np.zeros(len(bob_struct)*max_neighbors)
    bob_fill = np.zeros(len(bob_struct))
    main_atom_type = atom_types[0]
    for i in range(1,len(atom_typ_ord)):
        for j in range(len(bob_struct)):
            pair = bob_struct[j]
            if (pair[0] == main_atom_type and pair[1] == atom_typ_ord[i]) or \
                (pair[1] == main_atom_type and pair[0] == atom_typ_ord[i]):
                fill = bob_fill[j]
                if fill < max_neighbors:
                    bob_desc[j*max_neighbors+fill] = Z[i]*Z[0]/np.linalg.norm(
                        cart_ord[i,:] - cart_ord[0,:])
                    bob_fill[j] += 1
    return bob_desc,reorder

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
def atom_dens_free(at_coord, at_typ, pos, atom):
    '''Free-atom Gaussian density'''
    ra = constants.rad_free[at_typ]
    return 1/((2*math.pi)**(1.5)*ra**3)*math.exp(
        -np.linalg.norm(pos-at_coord)**2/(2.*ra**2))

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
def cart_to_spher(cart, stone_convention=True):
    'Convert cartesian to spherical multipoles'
    spher = np.zeros(5)
    if not stone_convention:
        spher[0] = 2.*cart[2,2]
        spher[1] = 2.*constants.sqrt_3*cart[0,2]
        spher[2] = 2.*constants.sqrt_3*cart[1,2]
        spher[3] = 1.*constants.sqrt_3*(cart[0,0]-cart[1,1])
        spher[4] = 2.*constants.sqrt_3*cart[0,1]
    else:
        spher[0] = cart[2,2]
        spher[1] = 2./constants.sqrt_3*cart[0,2]
        spher[2] = 2./constants.sqrt_3*cart[1,2]
        spher[3] = 1./constants.sqrt_3*(cart[0,0]-cart[1,1])
        spher[4] = 2./constants.sqrt_3*cart[0,1]
    return spher

def inertia_tensor(coords, elements):
    """
    Compute inertia tensor for coordinates coords and chemical
    elements elements. Return sorted eigenvectors and eigenvalues.
    """
    # Index masses
    masses = np.array([float(constants.atomic_weight[ele])
                    for _,ele in enumerate(elements)])
    # Center of mass
    com = np.sum([m*c for m,c in zip(masses,coords)],axis=0)/np.sum(masses)
    # Build inertia tensor
    coords_relative = coords - com
    inertia = np.dot(masses*coords_relative.T,coords_relative)
    eigvals, eigvecs = np.linalg.eig(inertia)
    sort_index = (-eigvals).argsort()
    eigvals = eigvals[sort_index]
    eigvecs = eigvecs[sort_index]
    return eigvals,eigvecs

def rotate_mtps(glob_coeffs, princ_axes):
    '''Rotate global multipoles using principal axes'''
    rot_coeffs = [np.zeros(9)]*len(glob_coeffs)
    for i in range(len(princ_axes)):
        # charge -- no rotation
        chg  = glob_coeffs[i][0]
        # dipole
        dip  = np.dot(glob_coeffs[i][1],princ_axes[i])
        # Quadrupole
        quad = cart_to_spher(np.dot(princ_axes[i].transpose(),
            np.dot(spher_to_cart(glob_coeffs[i][2]),princ_axes[i])))
        rot_coeffs[i] = np.hstack([chg, dip, quad])
    return rot_coeffs

def rotate_mtps_back(loc_coeffs, princ_axes):
    '''Rotate back local coefficients using principal axes'''
    rot_coeffs = [np.zeros(len(loc_coeffs[0]))]*len(loc_coeffs)
    for i in range(len(princ_axes)):
        tmp_rot_cf = np.zeros(9)
        idx_coeffs = 0
        while idx_coeffs < len(loc_coeffs[0]):
            # charge -- no rotation
            chg  = loc_coeffs[i][0+idx_coeffs]
            # dipole
            dip  = np.dot(loc_coeffs[i][1+idx_coeffs:4+idx_coeffs],
                princ_axes[i].transpose())
            # quadrupole
            quad = cart_to_spher(np.dot(princ_axes[i],
                np.dot(spher_to_cart( \
                    loc_coeffs[i][4+idx_coeffs:9+idx_coeffs]),
                    princ_axes[i].transpose())))
            tmp_rot_cf[0+idx_coeffs] = chg
            for k in range(3):
                tmp_rot_cf[1+k+idx_coeffs] = dip[k]
            for k in range(5):
                tmp_rot_cf[4+k+idx_coeffs] = quad[k]
            idx_coeffs += 9
        rot_coeffs[i] = tmp_rot_cf
    return rot_coeffs

def cosangle_two_atoms_inter(ele_i, crd_i, bonded_i, ele_j, crd_j, bonded_j, rij):
    '''Returns the cosine of the angle between the two atoms
    at_i_id-at_j_id and the vector rij. There can be two angles of
    interest if both atoms are hydrogen bonding.'''
    z_vec1, z_vec2 = np.zeros((3)), np.zeros((3))
    cosangle1 = 0.0
    cosangle2 = 0.0
    if len(bonded_i) == 1 and ele_i == "H" \
        and bonded_i[0][0] in ["O","N"] and ele_j in ["O","N"]:
        z_vec1 = crd_i - bonded_i[0][1]
        cosangle1 = abs(np.dot(z_vec1,rij)/(np.linalg.norm(z_vec1)*np.linalg.norm(rij)))
        if len(bonded_j) == 1 and ele_j == "H" \
            and bonded_j[0][0] in ["O","N"]:
            z_vec2 = crd_j - bonded_j[0][1]
            cosangle2 = abs(np.dot(z_vec1,z_vec2)/(np.linalg.norm(z_vec1)*
                                                    np.linalg.norm(z_vec2)))
    return cosangle1, cosangle2

@jit
def ab_rotation(vec1, vec2):
    '''Returns rotation matrix that takes vec1 into vec2'''
    r1, r2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if abs(r1) < 1e-4 or abs(r2) < 1e-4:
        return np.identity(3)
    a, b = vec1/r1, vec2/r2
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if abs(s) < 1e-4:
        return np.identity(3)
    c = np.dot(a, b)
    vx = np.zeros((3,3))
    vx[0,1] = -v[2]
    vx[1,0] =  v[2]
    vx[0,2] =  v[1]
    vx[2,0] = -v[1]
    vx[1,2] = -v[0]
    vx[2,1] =  v[0]
    return np.identity(3) + vx + vx.dot(vx)*(1-c)/s**2

@jit
def rotated_quadrupoles(gamma, g2, expa, shg, chg, norm, roti, rotj):
    """Compute rotated second-rank tensor kernel"""
    d_rot, d_tmp = np.zeros((9,9)), np.zeros((9,9))
    ent1 = expa/(4*g2) * (g2*shg - gamma*chg + shg) * norm
    ent2 = expa/(4*gamma) * (gamma*chg - shg) * norm
    ent3 = expa/(2*g2) * (gamma*chg - shg) * norm
    ent4 = expa/g2 * (g2/2.*shg - gamma*chg + shg) * norm
    d_tmp[0][0] =  ent1
    d_tmp[0][4] =  ent2
    d_tmp[1][1] =  ent1
    d_tmp[1][3] = -ent2
    d_tmp[2][2] =  ent3
    d_tmp[3][1] = -ent2
    d_tmp[3][3] =  ent1
    d_tmp[4][0] =  ent2
    d_tmp[4][4] =  ent1
    d_tmp[5][5] =  ent3
    d_tmp[6][6] =  ent3
    d_tmp[7][7] =  ent3
    d_tmp[8][8] =  ent4
    # Build R ^ R.T
    for k in range(9):
        k1, k2 = int(np.floor(k/3.)), int(k-3*np.floor(k/3.))
        for l in range(9):
            l1, l2 = int(np.floor(l/3.)), int(l-3*np.floor(l/3.))
            for m in range(3):
                for n in range(3):
                    for o in range(3):
                        for p in range(3):
                            d_rot[k][l] += roti.T[k1][m] * rotj[n][l1] * \
                                            d_tmp[3*m+p][3*n+o] * \
                                            rotj.T[l2][o] * roti[p][k2]
    return d_rot

def compare_two_atomic_envs(moli, molj, at_i, at_j, rk):
    """Compare atomic environments of at_i in moli and at_j in molj"""
    dim = 3**rk
    d = np.zeros((dim,dim)) if dim > 1 else 0.
    sig = 1.
    norm = 1./(2*np.sqrt(np.pi*sig**2))**3
    for i in range(moli.num_atoms):
        if at_i is not i:
            for j in range(molj.num_atoms):
                if at_j is not j:
                    if moli.elements[i] == molj.elements[j]:
                        veci, vecj = moli.pairwise_vec[at_i][i], molj.pairwise_vec[at_j][j]
                        ri, rj = moli.pairwise_norm[at_i][i], molj.pairwise_norm[at_j][j]
                        alpha, gamma = (ri**2+rj**2)/(4*sig**2), ri*rj/(2*sig**2)
                        g2 = gamma**2
                        expa, chg, shg = np.exp(-alpha), np.cosh(gamma), np.sinh(gamma)
                        base = expa/g2*(gamma * chg - shg)
                        ele = moli.elements[i]
                        if rk == 0:
                            d += base * norm
                        elif rk == 1:
                            d += base * np.outer(veci, vecj.T)/(ri*rj) * norm
                        else: #rk == 2
                            roti, rotj = moli.rot_mat[at_i][i], molj.rot_mat[at_j][j]
                            d += rotated_quadrupoles(gamma, g2, expa, shg, chg, norm, roti, rotj)
    return d

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())
