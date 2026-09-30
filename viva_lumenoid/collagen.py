"""Clean-room v1 collagen IV network as a process-bigraph Process.

``CollagenNetworkProcess`` owns a persistent LAMMPS simulation of the
coarse-grained collagen IV network (two-bead rods + stochastic NC1/7S
crosslinks, no excluded volume — spec v1-decision #2). It is driven each step
through two input ports so a sibling controller can run the spec's staged
stage-1 protocol without any mid-run script surgery:

    strain_rate   float  -> equibiaxial in-plane engineering strain rate (fix deform)
    chemistry_on  float  -> 0/1 toggle for crosslink make+break (fix bond/create/break)

Outputs mirror LAMMPSProcess (pxx/pyy/pzz, volume, box, positions) plus network
readouts (crosslink count, in-plane stress, accumulated strain). We reuse the
same ``lammps`` Python library that the sibling ``pbg_lammps`` wraps; this
workspace imports viva-lammps as a dependency and follows its bridge conventions
(strip ``run``, drive integration from ``update``, emit ``overwrite[T]`` sensors).
"""
from __future__ import annotations

import os
import tempfile

from process_bigraph import Process, Step

from .network import build_network_data
from .params import CollagenParams


# --------------------------------------------------------------------------- #
# LAMMPS script fragments
# --------------------------------------------------------------------------- #
def base_script(params: CollagenParams, data_path: str) -> str:
    """Base setup run once at build: box, styles, integrator, thermostat."""
    p = params
    pair_cutoff = max(p.crosslink_cutoff, p.crosslink_break_cutoff) + 1.0
    return f"""
units lj
atom_style molecular
boundary p p p
read_data {data_path} extra/bond/per/atom 12 extra/special/per/atom 200

bond_style harmonic
bond_coeff 1 {p.bond_k_intra} {p.bond_length}
bond_coeff 2 {p.crosslink_k} {p.crosslink_r0}
bond_coeff 3 {p.crosslink_k} {p.crosslink_r0}

# No excluded volume (spec: released decks set every pair eps=0). pair_style
# zero still builds the neighbor lists that fix bond/create needs.
pair_style zero {pair_cutoff}
pair_coeff * *
special_bonds lj 1.0 1.0 1.0

group nc1 type 1
group svns type 2

velocity all create {p.temperature} {p.seed} mom yes rot yes dist gaussian
fix integ all nve
fix therm all langevin {p.temperature} {p.temperature} {p.langevin_damp} {p.seed}
timestep {p.timestep}

# NETWORK stress = bond virial only (no kinetic/ideal-gas term), normalised by
# the INITIAL volume in the reader. The spec flags reading the all-atom stress
# instead of the network stress as one of the four stage-1 errors (decision #8),
# so we isolate the bond contribution here.
compute netP all pressure NULL bond
thermo 100000
thermo_style custom step temp pe press pxx pyy pzz vol c_netP[1] c_netP[2] c_netP[3]
"""


class CollagenNetworkProcess(Process):
    """Persistent coarse-grained collagen IV network driven by two control ports."""

    config_schema = {
        # params as a free-form mapping; CollagenParams(**config['params'])
        'params': 'map[float]',
        'working_directory': {'_type': 'string', '_default': ''},
    }

    def __init__(self, config=None, core=None):
        self._lmp = None  # set first so __del__/close are safe if super().__init__ raises
        super().__init__(config=config, core=core)
        self._params = CollagenParams(**{
            k: v for k, v in (self.config.get('params') or {}).items()
            if k in CollagenParams().to_dict()
        })
        self._dt = None
        self._first_run = True
        self._fixes_dirty = False
        self._deform_on = False
        self._deform_rate = None
        self._chem_on = False
        self._lx0 = None  # initial box length, for strain

    def inputs(self):
        return {'strain_rate': 'float', 'chemistry_on': 'float'}

    def outputs(self):
        return {
            'pxx': 'overwrite[float]', 'pyy': 'overwrite[float]',
            'pzz': 'overwrite[float]', 'pressure': 'overwrite[float]',
            'volume': 'overwrite[float]', 'box_dimensions': 'overwrite[list]',
            'positions': 'overwrite[list]', 'atom_types': 'overwrite[list]',
            'num_atoms': 'overwrite[integer]',
            'n_crosslinks': 'overwrite[integer]',
            'sigma_inplane': 'overwrite[float]',
            'strain': 'overwrite[float]',
        }

    # -- engine ------------------------------------------------------------- #
    def _build(self):
        if self._lmp is not None:
            return
        from lammps import lammps

        wd = self.config.get('working_directory') or tempfile.mkdtemp(prefix='viva_lumenoid_')
        os.makedirs(wd, exist_ok=True)
        data_path = os.path.join(wd, 'collagen_network.data')
        with open(data_path, 'w') as f:
            f.write(build_network_data(self._params))

        self._lmp = lammps(cmdargs=['-nocite', '-log', 'none', '-screen', 'none'])
        # read_data uses a relative path; run inside wd for portability.
        prev = os.getcwd()
        os.chdir(wd)
        try:
            self._lmp.commands_string(base_script(self._params, 'collagen_network.data'))
        finally:
            os.chdir(prev)
        self._dt = self._lmp.extract_global('dt')
        boxlo, boxhi, *_ = self._lmp.extract_box()
        self._lx0 = boxhi[0] - boxlo[0]
        self._v0 = (boxhi[0] - boxlo[0]) * (boxhi[1] - boxlo[1]) * (boxhi[2] - boxlo[2])

    def _apply_deform(self, rate):
        if rate is None:
            rate = 0.0
        if rate > 0.0:
            if not self._deform_on or rate != self._deform_rate:
                if self._deform_on:
                    self._lmp.command('unfix stretch')
                self._lmp.command(
                    f'fix stretch all deform 1 x erate {rate} y erate {rate} remap x')
                self._deform_on = True
                self._deform_rate = rate
                self._fixes_dirty = True
        else:
            if self._deform_on:
                self._lmp.command('unfix stretch')
                self._deform_on = False
                self._deform_rate = 0.0
                self._fixes_dirty = True

    def _apply_chemistry(self, on):
        want = bool(on and on >= 0.5)
        p = self._params
        if want and not self._chem_on:
            pbreak = p.factor_mult * p.make_prob
            s = p.seed
            self._lmp.command(
                f'fix xlink_nc1 all bond/create {p.make_every} 1 1 {p.crosslink_cutoff} 2 '
                f'iparam {1 + p.nc1_max_crosslinks} 1 jparam {1 + p.nc1_max_crosslinks} 1 '
                f'prob {p.make_prob} {s + 1}')
            self._lmp.command(
                f'fix xlink_7s all bond/create {p.make_every} 2 2 {p.crosslink_cutoff} 3 '
                f'iparam {1 + p.svns_max_crosslinks} 2 jparam {1 + p.svns_max_crosslinks} 2 '
                f'prob {p.make_prob} {s + 2}')
            self._lmp.command(
                f'fix xbreak2 all bond/break {p.break_every} 2 {p.crosslink_break_cutoff} '
                f'prob {pbreak} {s + 3}')
            self._lmp.command(
                f'fix xbreak3 all bond/break {p.break_every} 3 {p.crosslink_break_cutoff} '
                f'prob {pbreak} {s + 4}')
            self._chem_on = True
            self._fixes_dirty = True
        elif not want and self._chem_on:
            for name in ('xlink_nc1', 'xlink_7s', 'xbreak2', 'xbreak3'):
                self._lmp.command(f'unfix {name}')
            self._chem_on = False
            self._fixes_dirty = True

    def _read(self):
        lmp = self._lmp
        nlocal = lmp.extract_setting('nlocal')
        x = lmp.numpy.extract_atom('x')[:nlocal].copy()
        types = lmp.numpy.extract_atom('type')[:nlocal].copy()
        boxlo, boxhi, *_ = lmp.extract_box()
        lx, ly, lz = (boxhi[i] - boxlo[i] for i in range(3))
        nbonds = int(lmp.extract_global('nbonds'))
        n_crosslinks = max(0, nbonds - self._params.n_rods)  # total minus intra-rod
        pxx = float(lmp.get_thermo('pxx'))
        pyy = float(lmp.get_thermo('pyy'))
        # Network stress: bond-virial pressure tensor (compute netP), converted
        # back to a virial (x current volume) and normalised by the INITIAL
        # volume. Tension (stretched bonds) gives negative bond pressure, so the
        # in-plane STRESS is +(that), averaged over the two in-plane axes.
        from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR
        netP = lmp.numpy.extract_compute('netP', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
        vol = lx * ly * lz
        scale = vol / self._v0 if self._v0 else 1.0
        sigma_inplane = -0.5 * (float(netP[0]) + float(netP[1])) * scale
        strain = (lx / self._lx0) - 1.0 if self._lx0 else 0.0
        return {
            'pxx': pxx, 'pyy': pyy, 'pzz': float(lmp.get_thermo('pzz')),
            'pressure': float(lmp.get_thermo('press')),
            'volume': float(lmp.get_thermo('vol')),
            'box_dimensions': [lx, ly, lz],
            'positions': x.tolist(), 'atom_types': types.tolist(),
            'num_atoms': int(lmp.get_natoms()),
            'n_crosslinks': n_crosslinks,
            'sigma_inplane': sigma_inplane,
            'strain': strain,
        }

    def initial_state(self):
        self._build()
        self._lmp.command('run 0')
        self._first_run = False
        return self._read()

    def update(self, state, interval):
        self._build()
        if state:
            self._apply_deform(state.get('strain_rate'))
            self._apply_chemistry(state.get('chemistry_on'))
        n_steps = max(1, int(round(interval / self._dt)))
        # fix deform re-establishes its strain ramp during run setup, so while
        # deforming we must run with `pre yes` every step or the box stalls after
        # the first update. The cheap `pre no post no` path is only safe when no
        # managed fix changed and we are not actively deforming.
        if self._first_run or self._fixes_dirty or self._deform_on:
            self._lmp.command(f'run {n_steps}')
            self._first_run = False
            self._fixes_dirty = False
        else:
            self._lmp.command(f'run {n_steps} pre no post no')
        return self._read()

    def close(self):
        if self._lmp is not None:
            self._lmp.close()
            self._lmp = None

    def __del__(self):
        try:
            self.close()
        except (ImportError, TypeError):
            pass


class StagedStretchController(Process):
    """Sequence the spec's stage-1 protocol by writing the two control ports.

    Phases (spec: "stretch, hold, then switch the chemistry on"):
      assemble : chemistry on, no strain   -> grow the initial crosslinked network
      stretch  : chemistry off, strain ε̇   -> load the frozen network (elastic)
      hold_el  : chemistry off, no strain  -> elastic stress plateau
      hold_vis : chemistry on,  no strain  -> viscous stress relaxation (remodelling)
    """

    config_schema = {
        'strain_rate': {'_type': 'float', '_default': 1.0e-4},
        't_assemble': {'_type': 'float', '_default': 200.0},
        't_stretch': {'_type': 'float', '_default': 100.0},
        't_hold_elastic': {'_type': 'float', '_default': 200.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._t = 0.0

    def inputs(self):
        return {}

    def outputs(self):
        return {'strain_rate': 'float', 'chemistry_on': 'float', 'phase': 'string'}

    def update(self, state, interval):
        c = self.config
        t0 = c['t_assemble']
        t1 = t0 + c['t_stretch']
        t2 = t1 + c['t_hold_elastic']
        t = self._t
        self._t += interval
        if t < t0:
            return {'strain_rate': 0.0, 'chemistry_on': 1.0, 'phase': 'assemble'}
        if t < t1:
            return {'strain_rate': c['strain_rate'], 'chemistry_on': 0.0, 'phase': 'stretch'}
        if t < t2:
            return {'strain_rate': 0.0, 'chemistry_on': 0.0, 'phase': 'hold_elastic'}
        return {'strain_rate': 0.0, 'chemistry_on': 1.0, 'phase': 'hold_viscous'}


def register_viva_lumenoid(core):
    """Register this workspace's processes (idempotent)."""
    for name, cls in (
        ('CollagenNetworkProcess', CollagenNetworkProcess),
        ('StagedStretchController', StagedStretchController),
    ):
        if name not in core.link_registry:
            core.register_link(name, cls)
    return core
