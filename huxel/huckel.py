import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, value_and_grad
from jax.tree_util import tree_flatten, tree_multimap
from jax.nn import softmax

from huxel.parameters import H_X, N_ELECTRONS, H_X, H_XY
from huxel.molecule import myMolecule
from huxel.beta_functions import _f_beta

from typing import Any, Tuple


def f_homo_lumo_gap(params_b: dict, params_extra: dict, molecule: myMolecule, f_beta: callable) -> float:
    """Hückel model HOMO-LUMO gap prediction (linear transformation)

    Args:
        params_b (dict): parameters b
        params_extra (dict): additional parameters (Hückel model)
        molecule (class): molecule class
        f_beta (callable): atom-atom interaction

    Returns:
        float: value of the HOMO-LUMO gap predicted with the Hückel model
    """

    z_pred, extra = _homo_lumo_gap(params_b, params_extra, molecule, f_beta)
    h_m, e_ = extra
    y_pred = params_extra["hl_params"]["a"] * \
        z_pred + params_extra["hl_params"]["b"]
    return jnp.sum(y_pred)  # , h_m  # , (z_pred, extra)


def _homo_lumo_gap(params_b: dict, params_extra: dict, molecule: Any, f_beta: callable) -> Tuple:
    """Hückel model HOMO-LUMO gap prediction

    Args:
        params_b (dict): parameters b
        params_extra (dict): additional parameters (Hückel model)
        molecule (Any): molecule class
        f_beta (callable): atom-atom interaction

    Returns:
        float: HOMO-LUMO gap, Hückel matrix, eigenvalues
    """
    h_m, electrons = _construct_huckel_matrix(
        params_b, params_extra, molecule, f_beta)
    e_, _ = _solve(h_m)

    n_orbitals = h_m.shape[0]
    occupations, spin_occupations, n_occupied, n_unpaired = _set_occupations(
        jax.lax.stop_gradient(electrons),
        jax.lax.stop_gradient(e_),
        jax.lax.stop_gradient(n_orbitals),
    )
    idx_temp = jnp.nonzero(occupations)[0]
    homo_idx = jnp.argmax(idx_temp)
    lumo_idx = homo_idx + 1
    homo_energy = e_[homo_idx]
    lumo_energy = e_[lumo_idx]
    val = lumo_energy - homo_energy
    return jnp.sum(val), (h_m, e_)

# -----------------------------------------------------------------------------


def f_polarizability(params_b: dict, params_extra: dict, molecule:  myMolecule, f_beta: callable, external_field: Any = None) -> float:
    """Hückel model polarizability prediction (linear transformation)

    Args:
        params_b (dict): parameters b
        params_extra (dict): additional parameters (Hückel model)
        molecule (class): molecule class
        f_beta (callable): atom-atom interaction
        external_field (Any): external field

    Returns:
        float: value of the polarizability predicted with the Hückel model
    """
    z_pred = _f_polarizability(
        params_b, params_extra, molecule,  f_beta, external_field)
    y_pred = z_pred + params_extra["pol_params"]["b"]
    return jnp.sum(y_pred)  # ,z_pred,y_true


def _f_polarizability(params_b: dict, params_extra: dict, molecule: myMolecule, f_beta: callable, external_field: Any = None) -> float:
    """Hückel model polarizability prediction 

    Args:
        params_b (dict): parameters b
        params_extra (dict): additional parameters (Hückel model)
        molecule (class): molecule class
        f_beta (callable): atom-atom interaction
        external_field (Any): external field

    Returns:
        float: value of the polarizability predicted with the Hückel model
    """
    polarizability_tensor = jax.hessian(f_energy, argnums=(4))(
        params_b, params_extra, molecule, f_beta, external_field)
    polarizability = (1/3.)*jnp.trace(polarizability_tensor)
    return polarizability


def f_energy(params_b: dict, params_extra: dict, molecule: myMolecule, f_beta: callable, external_field: Any = None) -> float:
    """Hückel model's energy

    Args:
        params_b (dict): type of atoms 
        params_extra (dict): additional parameters (Hückel model)
        molecule (class): molecule class
        f_beta (callable): atom-atom interaction
        external_field (Any, optional): External field. Defaults to None.

    Returns:
        Any: Energy
    """
    h_m, electrons = _construct_huckel_matrix(
        params_b, params_extra, molecule, f_beta)

    if external_field != None:
        h_m_field = _construct_huckel_matrix_field(molecule, external_field)
        h_m = h_m + h_m_field

    e_, _ = _solve(h_m)

    n_orbitals = h_m.shape[0]
    occupations, spin_occupations, n_occupied, n_unpaired = _set_occupations(
        jax.lax.stop_gradient(electrons), jax.lax.stop_gradient(e_), jax.lax.stop_gradient(n_orbitals))
    return jnp.dot(occupations, e_)  # , h_m


# -------
def _construct_huckel_matrix_test(params_b: dict, params_extra: dict, molecule: myMolecule, f_beta: callable) -> Tuple:
    """Hückel matrix

    Args:
        params_b (dict): type of atoms
        params_extra (dict): additional parameters (Hückel model)
        molecule (myMolecule): molecule class
        f_beta (callable): atom-atom interaction

    Returns:
        Tuple: Hückel matrix, number of electrons
    """

    # atom_types,conectivity_matrix = molecule
    atom_types = molecule.atom_types
    connectivity_matrix = molecule.connectivity_matrix
    # dm = molecule.dm

    h_x = params_extra["h_x"]
    h_xy = params_extra["h_xy"]
    print([key for _, key in enumerate(h_xy)])
    one_pi_elec = params_extra["one_pi_elec"]
    print(one_pi_elec)

    h_xy_flat, h_xy_tree = tree_flatten(h_xy)
    h_xy_flat = jnp.asarray(h_xy_flat)
    h_x_flat, h_x_tree = tree_flatten(h_x)
    h_x_flat = jnp.asarray(h_x_flat)

    norm_params_b = jax.tree_map(lambda x: softmax(x), params_b)
    norm_params_b_flat, norm_params_b_tree = tree_flatten(norm_params_b)
    norm_params_b_flat = jnp.array(norm_params_b_flat)
    huckel_matrix = jnp.zeros_like(connectivity_matrix, dtype=jnp.float32)

    # norm_params_b[0].shape[0]
    zi_triu_up = jnp.triu_indices(norm_params_b[0].shape[0], 0)
    print(connectivity_matrix)
    print(jnp.nonzero(connectivity_matrix))
    # off diagonal terms
    for i, j in zip(*jnp.nonzero(connectivity_matrix)):
        print(i, j)
        x = norm_params_b_flat[i]
        y = norm_params_b_flat[j]
        print(x, y)
        Z = jnp.multiply(x[jnp.newaxis], y[jnp.newaxis].T)
        print(Z)
        z = Z[zi_triu_up]
        zt = Z.T[zi_triu_up]
        z = z + zt
        print(z)
        print(zt)
        print(h_xy_flat)
        z_ij = jnp.matmul(z, h_xy_flat)
        print(z_ij)
        huckel_matrix = huckel_matrix.at[i, j].set(z_ij)
        # huckel_matrix = huckel_matrix.at[i, j].set(1.)
        # assert 0

    # diagonal terms
    for i, c in enumerate(atom_types):
        z = jnp.vdot(h_x_flat, norm_params_b[i])
        huckel_matrix = huckel_matrix.at[i, i].set(z)

    electrons = _electrons(atom_types)

    return huckel_matrix, electrons


def _construct_huckel_matrix(params_b: dict, params_extra: dict, molecule: myMolecule, f_beta: callable) -> Tuple:
    """Hückel matrix

    Args:
        params_b (dict): type of atoms
        params_extra (dict): additional parameters (Hückel model)
        molecule (myMolecule): molecule class
        f_beta (callable): atom-atom interaction

    Returns:
        Tuple: Hückel matrix, number of electrons
    """

    # atom_types,conectivity_matrix = molecule
    atom_types = molecule.atom_types
    connectivity_matrix = molecule.connectivity_matrix
    # dm = molecule.dm

    h_x = params_extra["h_x"]
    h_xy = params_extra["h_xy"]
    one_pi_elec = params_extra["one_pi_elec"]

    h_xy_flat, h_xy_tree = tree_flatten(h_xy)
    h_xy_flat = jnp.asarray(h_xy_flat)
    h_x_flat, h_x_tree = tree_flatten(h_x)
    h_x_flat = jnp.asarray(h_x_flat)

    norm_params_b = jax.tree_map(lambda x: softmax(x), params_b)
    norm_params_b_flat, norm_params_b_tree = tree_flatten(norm_params_b)
    norm_params_b_flat = jnp.array(norm_params_b_flat)
    huckel_matrix = jnp.zeros_like(connectivity_matrix, dtype=jnp.float32)

    # norm_params_b[0].shape[0]
    zi_triu_up = jnp.triu_indices(norm_params_b[0].shape[0], 0)
    # off diagonal terms
    for i, j in zip(*jnp.nonzero(connectivity_matrix)):
        x = norm_params_b_flat[i]
        y = norm_params_b_flat[j]
        Z = jnp.multiply(x[jnp.newaxis], y[jnp.newaxis].T)
        z = Z[zi_triu_up] + Z.T[zi_triu_up]
        z_ij = jnp.matmul(z, h_xy_flat)
        huckel_matrix = huckel_matrix.at[i, j].set(z_ij)

    # diagonal terms
    for i, c in enumerate(atom_types):
        z = jnp.vdot(h_x_flat, norm_params_b[i])
        huckel_matrix = huckel_matrix.at[i, i].set(z)

    electrons = _electrons(atom_types)

    return huckel_matrix, electrons


def _construct_huckel_matrix_field(molecule: myMolecule, field: Any) -> Any:
    """Diagonal elements of Hückel matrix in the presence of an external field

    Args:
        molecule (myMolecule): molecule class
        field (Any): external field

    Returns:
        Any: Diagonal of the Hückel matrix in the presence of an external field
    """
    # atom_types = molecule.atom_types
    xyz = molecule.xyz
    # diagonal terms
    diag_ri = jnp.asarray([jnp.diag(xyz[:, i])for i in range(3)])
    def field_r(fi, xi): return fi*xi
    diag_ri_tensor = vmap(field_r, in_axes=(0, 0))(field, diag_ri)
    diag_ri = jnp.sum(diag_ri_tensor, axis=0)
    return diag_ri


def _electrons(atom_types: list) -> Any:
    """number of electrons per atom-site

    Args:
        atom_types (list): type of atoms

    Returns:
        Any: number of electrons
    """
    return jnp.stack([N_ELECTRONS[atom_type] for atom_type in atom_types])


def _solve(huckel_matrix: Any) -> Tuple:
    """Return the eigenvalues and eigenvectors of the Hückel matrix

    Args:
        huckel_matrix (Any):  Hückel matrix

    Returns:
        Tuple: Eigenvalues and Eigenvectors
    """
    eig_vals, eig_vects = jnp.linalg.eigh(huckel_matrix)
    return eig_vals[::-1], eig_vects.T[::-1, :]


def _get_multiplicty(n_electrons: int) -> int:
    """Multiplicity

    Args:
        n_electrons (int): number of electrons

    Returns:
        Any: multiplicity
    """
    return (n_electrons % 2) + 1


def _set_occupations(electrons: int, energies: Any, n_orbitals: int) -> Tuple:
    """Occupation

    Args:
        electrons (int): number of electrons
        energies (Any): Hückel's eigenvalues
        n_orbitals (int): number of orbitals

    Returns:
        Tuple: occupation, spin occupation, number of occupied orbitals, number of unpair electrons
    """
    charge = 0
    n_dec_degen = 3
    n_electrons = jnp.sum(electrons) - charge
    multiplicity = _get_multiplicty(n_electrons)
    n_excess_spin = multiplicity - 1

    # Determine number of singly and doubly occupied orbitals.
    n_doubly = int((n_electrons - n_excess_spin) / 2)
    n_singly = n_excess_spin

    # Make list of electrons to distribute in orbitals
    all_electrons = [2] * n_doubly + [1] * n_singly

    # Set up occupation numbers
    occupations = jnp.zeros(n_orbitals, dtype=jnp.int32)
    spin_occupations = jnp.zeros(n_orbitals, dtype=jnp.int32)

    # Loop over unique rounded orbital energies and degeneracies and fill with
    # electrons
    energies_rounded = energies.round(n_dec_degen)
    unique_energies, degeneracies = jnp.unique(
        energies_rounded, return_counts=True)
    for energy, degeneracy in zip(jnp.flip(unique_energies), jnp.flip(degeneracies)):
        if len(all_electrons) == 0:
            break

        # Determine number of electrons with and without excess spin.
        electrons_ = 0
        spin_electrons_ = 0
        for _ in range(degeneracy):
            if len(all_electrons) > 0:
                pop_electrons = all_electrons.pop(0)
                electrons_ += pop_electrons
                if pop_electrons == 1:
                    spin_electrons_ += 1

        # Divide electrons evenly among orbitals
        # occupations[jnp.where(energies_rounded == energy)] += electrons / degeneracy
        occupations = occupations.at[energies_rounded == energy].add(
            electrons_ / degeneracy
        )

        # spin_occupations[np.where(energies_rounded == energy)] += (spin_electrons / degeneracy)
        spin_occupations = occupations.at[energies_rounded == energy].add(
            spin_electrons_ / degeneracy
        )

    n_occupied = jnp.count_nonzero(occupations)
    n_unpaired = int(
        jnp.sum(occupations[:n_occupied][occupations[:n_occupied] != 2]))
    return occupations, spin_occupations, n_occupied, n_unpaired


# ------------------------
# TEST


def update_params(p, g, alpha=0.1):
    def inner_sgd_fn(g, params): return (params - alpha * g)
    return tree_multimap(inner_sgd_fn, g, p)


def main_test():
    h_x = H_X
    h_xy = H_XY
    params = (h_x, h_xy)

    atom_types = ["C", "C", "C", "C"]

    conectivity_matrix = jnp.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=int
    )

    molec = myMolecule("test", atom_types, conectivity_matrix, 1.0)

    # test single molecule
    v, g = value_and_grad(
        f_homo_lumo_gap,
        has_aux=True,
    )(params, molec)
    print("HOMO-LUMO")
    homo_lumo_val, _ = v
    print(homo_lumo_val)
    print("GRAD HOMO-LUMO")
    print(g)


if __name__ == "__main__":

    main_test()
