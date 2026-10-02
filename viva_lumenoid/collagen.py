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
import math
import tempfile

import numpy as np
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
    # Junction bending (FP1) is applied via `fix restrain angle` pinned to each
    # junction's AS-FORMED angle (see _rebuild_restraints) — NOT an angle_style
    # with a single rest angle, which frustrates the randomly-oriented junctions
    # and adds stress-variance instead of rigidity. Stiff restraints need a
    # smaller timestep, so clamp it when bending is on (the spec's own dt=0.001–
    # 0.002 regime). A wider ghost cutoff keeps stretched crosslinks' partners
    # in range once the network stiffens.
    dt = min(p.timestep, 0.002) if p.bending_k > 0.0 else p.timestep
    real_angles = getattr(p, "use_real_angles", False) and p.bending_k > 0.0
    angle_extra = " extra/angle/per/atom 20" if real_angles else ""
    angle_setup = (f"\nangle_style harmonic\nangle_coeff 1 {p.bending_k} {p.bending_theta0}"
                   if real_angles else "")
    return f"""
units lj
atom_style molecular
boundary p p p
read_data {data_path} extra/bond/per/atom 12 extra/special/per/atom 200{angle_extra}

bond_style harmonic
bond_coeff 1 {p.bond_k_intra} {p.bond_length}
bond_coeff 2 {p.crosslink_k} {p.crosslink_r0}
bond_coeff 3 {p.crosslink_k} {p.crosslink_r0}{angle_setup}

# No excluded volume (spec: released input scripts set every pair eps=0). pair_style
# zero still builds the neighbor lists that fix bond/create needs.
pair_style zero {pair_cutoff}
pair_coeff * *
special_bonds lj 1.0 1.0 1.0
comm_modify cutoff 4.0

group nc1 type 1
group svns type 2

velocity all create {p.temperature} {p.seed} mom yes rot yes dist gaussian
fix integ all nve
fix therm all langevin {p.temperature} {p.temperature} {p.langevin_damp} {p.seed}
timestep {dt}

# NETWORK stress = bond virial only (no kinetic/ideal-gas term), normalised by
# the INITIAL volume in the reader. The spec flags reading the all-atom stress
# instead of the network stress as one of the four stage-1 errors (decision #8),
# so we isolate the bond contribution here.
compute netP all pressure NULL bond
# Time-average the bond-virial stress to cut the thermal noise (∝1/sqrt(N_bonds))
# that otherwise swamps the relaxation signal: mean of 20 samples over the last
# 200 steps. The reader prefers this average and falls back to the instantaneous
# compute before the first window fills.
fix netPavg all ave/time 10 40 400 c_netP[1] c_netP[2] c_netP[3]
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
        self._make_on = False
        self._break_calls = 0
        self._restrain_on = False   # junction-bending fix restrain active?
        self._chem_was_active = False
        self._angled = set()        # NC1 crosslinks that already have real angles
        self._lx0 = None  # initial box length, for strain

    def inputs(self):
        # make_on: formation (fix bond/create) on/off.
        # break_on: force-independent off-rate (random deletion) on/off.
        # Separately controllable so the staged protocol can run "breakage off,
        # then on" (spec Fig 2D) without formation confounding the relaxation.
        return {'strain_rate': 'float', 'make_on': 'float', 'break_on': 'float'}

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

    def _apply_make(self, on):
        """Toggle crosslink FORMATION (fix bond/create for NC1 and 7S ends)."""
        want = bool(on and on >= 0.5)
        p = self._params
        if want and not self._make_on:
            s = p.seed
            self._lmp.command(
                f'fix xlink_nc1 all bond/create {p.make_every} 1 1 {p.crosslink_cutoff} 2 '
                f'iparam {1 + p.nc1_max_crosslinks} 1 jparam {1 + p.nc1_max_crosslinks} 1 '
                f'prob {p.make_prob} {s + 1}')
            self._lmp.command(
                f'fix xlink_7s all bond/create {p.make_every} 2 2 {p.crosslink_cutoff} 3 '
                f'iparam {1 + p.svns_max_crosslinks} 2 jparam {1 + p.svns_max_crosslinks} 2 '
                f'prob {p.make_prob} {s + 2}')
            self._make_on = True
            self._fixes_dirty = True
        elif not want and self._make_on:
            for name in ('xlink_nc1', 'xlink_7s'):
                self._lmp.command(f'unfix {name}')
            self._make_on = False
            self._fixes_dirty = True

    def _break_crosslinks(self, interval):
        """Delete a random fraction 1-exp(-off_rate*interval) of crosslink bonds.

        A force-INDEPENDENT stochastic off-rate: unlike `fix bond/break` (which
        only breaks over-stretched bonds), this removes bonds regardless of their
        length, so the remodelling time is a clean, controllable τ ≈ 1/off_rate.
        """
        p = self._params
        frac = 1.0 - math.exp(-p.off_rate * interval)
        if frac <= 0.0:
            return
        nb, data = self._lmp.gather_bonds()
        arr = np.array(data, dtype=int).reshape(-1, 3)
        xlinks = arr[(arr[:, 0] == 2) | (arr[:, 0] == 3)]
        if len(xlinks) == 0:
            return
        n_del = int(round(frac * len(xlinks)))
        if n_del <= 0:
            return
        rng = np.random.default_rng(p.seed + self._break_calls)
        self._break_calls += 1
        sel = xlinks[rng.choice(len(xlinks), size=min(n_del, len(xlinks)), replace=False)]
        for bt, a1, a2 in sel:
            self._lmp.command(f'group _br id {int(a1)} {int(a2)}')
            self._lmp.command(f'delete_bonds _br bond {int(bt)} remove special')
            self._lmp.command('group _br delete')
        self._fixes_dirty = True  # topology changed → next run needs full setup

    @staticmethod
    def _rod_partner(atom_id: int) -> int:
        """Intra-rod partner bead: rod i = atoms (2i-1 NC1, 2i 7S), so an odd id's
        partner is id+1 and an even id's partner is id-1 (see network.build_network_data)."""
        return atom_id + 1 if atom_id % 2 == 1 else atom_id - 1

    def _rebuild_restraints(self):
        """Resync junction-bending restraints to the current crosslink set (FP1).

        Each crosslink (u-v) is braced by two angle restraints — partner(u)-u-v
        and u-v-partner(v) — pinned to their CURRENT (as-formed) angle, so the
        junction resists bending away from the geometry it assembled in (no
        single-rest-angle frustration). Uses `fix restrain`, which takes a target
        per angle. Rebuilt whenever the crosslink topology changes so no restraint
        references a broken crosslink; the targets are frozen through the stretch
        phase, which is what makes them resist the applied strain.
        """
        p = self._params
        if p.bending_k <= 0.0:
            return
        lmp = self._lmp
        nlocal = lmp.extract_setting('nlocal')
        x = lmp.numpy.extract_atom('x')[:nlocal].copy()
        boxlo, boxhi, *_ = lmp.extract_box()
        L = np.array([boxhi[i] - boxlo[i] for i in range(3)])
        nb, data = lmp.gather_bonds()
        arr = np.array(data, dtype=int).reshape(-1, 3)
        xlinks = arr[(arr[:, 0] == 2) | (arr[:, 0] == 3)]
        if self._restrain_on:
            lmp.command('unfix rst')
            self._restrain_on = False
        terms = []
        for _bt, u, v in xlinks:
            for c, a, b in ((int(u), self._rod_partner(int(u)), int(v)),
                            (int(v), int(u), self._rod_partner(int(v)))):
                da = x[a - 1] - x[c - 1]; da -= L * np.round(da / L)
                db = x[b - 1] - x[c - 1]; db -= L * np.round(db / L)
                cosang = np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-12)
                ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
                terms.append(f'angle {a} {c} {b} {p.bending_k} {p.bending_k} {ang:.1f}')
        if terms:
            lmp.command('fix rst all restrain ' + ' '.join(terms))
            self._restrain_on = True
        self._fixes_dirty = True

    def _create_junction_angles(self):
        """FAITHFUL path: create a REAL harmonic angle at each NC1 junction as the
        crosslink forms, so the network equilibrates with the angles active.

        For an NC1 crosslink u-v, two angles enforce the released model's NC1
        linearity: partner(u)-u-v and u-v-partner(v), both angle type 1 (the
        angle_coeff sets k=bending_k, rest=180°). Incremental — only new NC1
        crosslinks get angles (tracked in self._angled); no deletion, so this is
        used with make-only assembly (breaking is suppressed while real angles
        form). Mirrors how the released bond/react templates add angles on bond
        formation rather than pinning restraints on afterwards.
        """
        p = self._params
        if p.bending_k <= 0.0:
            return
        lmp = self._lmp
        _nb, data = lmp.gather_bonds()
        arr = np.array(data, dtype=int).reshape(-1, 3)
        xlinks = arr[arr[:, 0] == 2]          # NC1 crosslinks (bond type 2)
        created = 0
        for _bt, u, v in xlinks:
            u, v = int(u), int(v)
            key = (min(u, v), max(u, v))
            if key in self._angled:
                continue
            pu, pv = self._rod_partner(u), self._rod_partner(v)
            if pu and pv and pu != v and pv != u:
                lmp.command(f'create_bonds single/angle 1 {pu} {u} {v}')
                lmp.command(f'create_bonds single/angle 1 {u} {v} {pv}')
                self._angled.add(key)
                created += 1
        if created:
            self._fixes_dirty = True

    def _rebuild_real_angles(self):
        """Sync real junction angles to the CURRENT NC1 crosslink set when breaking
        is active: delete all type-1 angles and recreate them for the crosslinks
        that still exist. Robust (no per-angle bookkeeping) and lets the real-angle
        path run make+break assembly — the released balance — rather than make-only.
        """
        p = self._params
        if p.bending_k <= 0.0:
            return
        lmp = self._lmp
        lmp.command('delete_bonds all angle 1 remove')
        _nb, data = lmp.gather_bonds()
        arr = np.array(data, dtype=int).reshape(-1, 3)
        xlinks = arr[arr[:, 0] == 2]
        self._angled = set()
        for _bt, u, v in xlinks:
            u, v = int(u), int(v)
            pu, pv = self._rod_partner(u), self._rod_partner(v)
            if pu and pv and pu != v and pv != u:
                lmp.command(f'create_bonds single/angle 1 {pu} {u} {v}')
                lmp.command(f'create_bonds single/angle 1 {u} {v} {pv}')
                self._angled.add((min(u, v), max(u, v)))
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
        avg = [float(lmp.extract_fix('netPavg', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, i))
               for i in range(2)]
        if avg[0] == 0.0 and avg[1] == 0.0:   # window not filled yet → instantaneous
            netP = lmp.numpy.extract_compute('netP', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
            avg = [float(netP[0]), float(netP[1])]
        vol = lx * ly * lz
        scale = vol / self._v0 if self._v0 else 1.0
        sigma_inplane = -0.5 * (avg[0] + avg[1]) * scale
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
        bending = self._params.bending_k > 0.0
        real_angles = getattr(self._params, 'use_real_angles', False) and bending
        if state:
            self._apply_deform(state.get('strain_rate'))
            self._apply_make(state.get('make_on'))
            made = bool(state.get('make_on', 0.0) and state['make_on'] >= 0.5)
            broke = bool(state.get('break_on', 0.0) and state['break_on'] >= 0.5)
            # Real-angle ASSEMBLY is make-only (breaking suppressed while making),
            # which reaches a dense, reproducible, angled network — the make+break
            # balance at this scale is too sparse to percolate without the authors'
            # dense initial config + GCE. Breaking is still handled in a break-only
            # phase (the viscous hold), where angles are re-synced by deletion.
            if real_angles and made:
                broke = False
            if broke:
                # Breaking (off-rate deletion) BEFORE integrating so the network
                # relaxes over the interval after losing bonds.
                self._break_crosslinks(interval)
            chem_active = made or broke
            if real_angles:
                # Keep real junction angles synced to the crosslink set: rebuild
                # (delete-all + recreate) after breaking so no angle references a
                # deleted crosslink; otherwise create angles for new crosslinks
                # incrementally. This lets the real-angle network remodel (break
                # crosslinks) in the viscous hold with the angles staying valid.
                if broke:
                    self._rebuild_real_angles()
                else:
                    self._create_junction_angles()
            elif bending:
                # Pin junction-bending restraints to the AS-FORMED geometry exactly
                # ONCE, at the assemble→frozen transition. Rebuilding every assembly
                # step (assemble runs make+break together) re-pins to unsettled
                # geometries and destabilises the run, so restraints are NOT created
                # during assembly. During the viscous hold (breakage only: break on,
                # make off) rebuild each step so no restraint references a deleted
                # crosslink.
                if broke and not made:
                    self._rebuild_restraints()
                elif self._chem_was_active and not chem_active:
                    self._rebuild_restraints()   # freeze-in the assembled network
                self._chem_was_active = chem_active
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

    def athermal_modulus(self, target_strain=0.04, n_increments=6, minimize_iters=4000):
        """Athermal (T=0) elastic modulus: energy-minimise, then apply small
        equibiaxial strain increments (change_box + minimise) and fit the
        stress-strain slope. Deterministic — no thermal chaos — so the FLOPPY
        modulus comes out cleanly (~0). For a stiff BENDING network the minimised
        state is ill-conditioned (pre-stress + many local minima), so the value
        is large but not reproducible at v1 network size — see the bm-v4 study.
        Run it on a frozen, assembled (and, for bending, restraint-pinned) network.
        """
        from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR
        lmp = self._lmp

        def read_sigma():
            lmp.command('run 0')
            netP = lmp.numpy.extract_compute('netP', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
            boxlo, boxhi, *_ = lmp.extract_box()
            lx, ly, lz = (boxhi[i] - boxlo[i] for i in range(3))
            sigma = -0.5 * (float(netP[0]) + float(netP[1])) * (lx * ly * lz / self._v0)
            return sigma, lx / self._lx0 - 1.0

        lmp.command(f'minimize 1e-6 1e-8 {minimize_iters} {minimize_iters * 10}')
        s0, _ = read_sigma()
        strains, sigmas = [], []
        de = target_strain / n_increments
        for _ in range(n_increments):
            lmp.command(f'change_box all x scale {1 + de} y scale {1 + de} remap units box')
            lmp.command(f'minimize 1e-6 1e-8 {minimize_iters} {minimize_iters * 10}')
            s, e = read_sigma()
            sigmas.append(s); strains.append(e)
        A = np.vstack([strains, np.ones(len(strains))]).T
        slope = float(np.linalg.lstsq(A, np.array(sigmas) - s0, rcond=None)[0][0])
        return slope

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
    """Sequence the spec's stage-1 protocol (Fig 2D: "breakage off, then on").

    Phases:
      assemble : make+break on, no strain  -> grow to the make/break steady state
      relax    : frozen, no strain         -> settle the pre-stress (equilibration)
      stretch  : frozen, strain ε̇          -> load the frozen network (elastic)
      hold_el  : frozen, no strain         -> elastic stress plateau
      hold_vis : BREAKAGE ONLY, no strain  -> viscous stress relaxation (remodelling)

    "Frozen" = make_on=0, break_on=0 so the topology is fixed during loading. The
    viscous hold turns on breaking ONLY, so the relaxation is not confounded by
    formation (the densification bug the old chemistry_on toggle produced).
    """

    config_schema = {
        'strain_rate': {'_type': 'float', '_default': 1.0e-4},
        't_assemble': {'_type': 'float', '_default': 200.0},
        't_relax': {'_type': 'float', '_default': 200.0},
        't_stretch': {'_type': 'float', '_default': 100.0},
        't_hold_elastic': {'_type': 'float', '_default': 200.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._t = 0.0

    def inputs(self):
        return {}

    def outputs(self):
        return {'strain_rate': 'float', 'make_on': 'float', 'break_on': 'float',
                'phase': 'string'}

    def update(self, state, interval):
        c = self.config
        t0 = c['t_assemble']
        t1 = t0 + c['t_relax']
        t2 = t1 + c['t_stretch']
        t3 = t2 + c['t_hold_elastic']
        t = self._t
        self._t += interval

        def out(sr, mk, bk, ph):
            return {'strain_rate': sr, 'make_on': mk, 'break_on': bk, 'phase': ph}

        if t < t0:
            return out(0.0, 1.0, 1.0, 'assemble')
        if t < t1:
            return out(0.0, 0.0, 0.0, 'relax')
        if t < t2:
            return out(c['strain_rate'], 0.0, 0.0, 'stretch')
        if t < t3:
            return out(0.0, 0.0, 0.0, 'hold_elastic')
        return out(0.0, 0.0, 1.0, 'hold_viscous')


def register_viva_lumenoid(core):
    """Register this workspace's processes (idempotent)."""
    for name, cls in (
        ('CollagenNetworkProcess', CollagenNetworkProcess),
        ('StagedStretchController', StagedStretchController),
    ):
        if name not in core.link_registry:
            core.register_link(name, cls)
    return core
