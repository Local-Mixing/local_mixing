"""PR-identity generator: compose atoms into a long identity, then launder it.

Pipeline
--------
1. SEED     draw atoms (curated / minted class-O / MITM-sampled minimal
            identities), relabel each onto random wires of the n-wire circuit
            with a tunable amount of wire sharing to induce coupling.
2. COMPOSE  concatenate/splice the atoms into one length-m identity.  Sound by
            construction: each atom restores its own support, so the running
            state is preserved across atom boundaries regardless of interleaving
            of *commuting* atoms; non-commuting atoms are placed as nested blocks
            (contiguous inside a gap), which is always identity-preserving.
3. LAUNDER  many local rewrites (window resynthesis + commuting reorder) that
            preserve the global function (identity) while erasing the syntactic
            fingerprints the raw composition leaves: repeated gates, contiguous
            atom id-windows, and the commuting-factorisation into seed atoms.

Every stage is verified: exact identity at small n, Monte-Carlo at large n.
"""
import random

import g57
from atoms import Resynth


class Generator:
    def __init__(self, n, atom_pool, resynth=None, seed=None):
        self.n = n
        self.pool = atom_pool                     # list of atom dicts
        self.rng = random.Random(seed)
        self.resynth = resynth or Resynth(max_wires=5, radius=7)

    # -- seeding -------------------------------------------------------------
    #
    # Wire discipline that keeps interleaving sound:
    #   WRITE wires  -- each atom's targets get PRIVATE wires from a global pool,
    #                   so no wire is ever written by two atoms and no atom reads
    #                   another atom's write wire;
    #   READ wires   -- all controls draw from one shared read-only pool, never
    #                   written by anyone (sharing controls is harmless -- reads
    #                   commute with reads).
    # Hence every atom writes wires no other atom touches: the atoms pairwise
    # commute as blocks, so any order-preserving interleaving is still identity.
    # The shared read pool couples atoms in the wire graph (defeats a naive
    # connectivity test) but they still COMMUTE -- the factorisation fingerprint
    # this leaves is exactly what the laundering pass then has to erase.
    def _place(self, atom, write_pool, read_pool):
        targets = sorted(set(g57.targets_of(atom["word"])))
        ctrls = [w for w in g57.wires_of(atom["word"]) if w not in targets]
        if len(write_pool) < len(targets):
            raise RuntimeError("write-wire budget exhausted; raise n or lower atoms")
        mapping = {t: write_pool.pop() for t in targets}
        # injective on controls: a relabeling must not collapse two distinct
        # control wires onto one read wire (that would fabricate repeated gates).
        if len(ctrls) > len(read_pool):
            raise RuntimeError("read pool smaller than an atom's control count; raise share_frac")
        for c, r in zip(ctrls, self.rng.sample(read_pool, len(ctrls))):
            mapping[c] = r
        return g57.relabel(atom["word"], mapping)

    def seed(self, n_atoms, share_frac=0.25):
        n_read = max(3, int(self.n * share_frac))
        read_pool = self.rng.sample(range(self.n), n_read)
        write_pool = [w for w in range(self.n) if w not in read_pool]
        self.rng.shuffle(write_pool)
        placed = []
        for _ in range(n_atoms):
            atom = self.rng.choice(self.pool)
            placed.append(self._place(atom, write_pool, read_pool))
        return placed

    # -- composition ---------------------------------------------------------
    def compose(self, placed, interleave=True):
        """Interleave placed atoms (each an identity) into one word.

        Disjoint-write placement means the atoms pairwise commute as blocks, so
        *any* order-preserving interleaving is still the identity.  We verify.
        """
        if not interleave:
            word = [g for atom in placed for g in atom]
        else:
            slots = [i for i, a in enumerate(placed) for _ in a]
            self.rng.shuffle(slots)
            ptr = [0] * len(placed)
            word = []
            for i in slots:
                word.append(placed[i][ptr[i]])
                ptr[i] += 1
        return word

    # -- pipelined composition (wire-level release) --------------------------
    #
    # `compose` holds every atom's write wires private for the atom's whole
    # life, so the write pool caps the word at ~3n gates and longer identities
    # need batching -- which returns the state exactly to the input at each
    # batch seam.  Both limits come from releasing wires too late.
    #
    # RELEASE RULE (sound): an atom A is an identity on its support, so once
    # A's LAST gate touching wire w has executed, w already holds its original
    # value and A never touches it again.  Hence w may be returned to the free
    # pool IMMEDIATELY, while A is still in flight.  Reuse cannot disturb A:
    # the new owner only writes w (controls are drawn from the read-only pool,
    # which is never written), and A has no remaining gate on w.
    #
    # Consequence: atoms are pipelined rather than batched -- several are
    # always mid-flight, so the running state never returns to the input until
    # the final drain, and the word grows without bound.
    # `max_inflight` is the MASTER knob, not a performance detail.  The fold map
    # s_{mid-t} -> s_{mid+t} is the permutation of the gates between those cuts;
    # atoms lying wholly inside that window are identities and contribute
    # nothing, and (writes being disjoint) atoms commute, so the map is carried
    # ONLY by the atoms straddling the two cuts -- at most ~2*max_inflight atoms
    # of <=12 gates, a bound independent of both t and the word length.  That is
    # why the affine fold is flat in t and why more length does not help it.
    # Default therefore saturates the write-wire budget (~n/3 atoms, each
    # claiming ~3 targets) instead of the old fixed 8.
    # `share_frac` must be an ABSOLUTE wire count, not a fraction.  Read-pool
    # wires are never written, so they hold their input value at every cut and
    # fold back with exactly zero error: measured law (3 decimals)
    #     aff_mean = (1 - n_read/n) * aff_written
    # i.e. the read pool is a pure multiplicative haircut.  A fixed 25% fraction
    # therefore caps the achievable fold at 0.375 no matter how good the rest
    # is; the pool only has to be as large as the widest atom's control set.
    def pipelined(self, m_target, share_frac=None, max_inflight=None, tries=32):
        n, rng = self.n, self.rng
        if max_inflight is None:
            # Uncapped: the free-write-wire budget is the real limiter, and it
            # saturates on its own (n=128: 0.378 at n/3, 0.440 at >=n, then
            # bit-identical thereafter).  Capping below saturation throws away
            # straddling atoms, which is exactly what carries the fold map.
            max_inflight = 1 << 30
        if share_frac is None:                 # absolute minimum, not a fraction
            n_read = min(max(3, n // 4), 8)
        else:
            n_read = max(3, int(n * share_frac))
        read_pool = rng.sample(range(n), n_read)
        free = [w for w in range(n) if w not in read_pool]
        rng.shuffle(free)
        inflight, word = [], []

        def start():
            for _ in range(tries):
                atom = rng.choice(self.pool)
                aw = atom["word"]
                targets = sorted(set(g57.targets_of(aw)))
                ctrls = [w for w in g57.wires_of(aw) if w not in targets]
                if len(targets) > len(free) or len(ctrls) > len(read_pool):
                    continue
                mapping = {t: free.pop() for t in targets}
                for c, r in zip(ctrls, rng.sample(read_pool, len(ctrls))):
                    mapping[c] = r
                w = g57.relabel(aw, mapping)
                last = {}
                for i, g in enumerate(w):
                    for v in g:
                        last[v] = i           # last gate index touching v
                inflight.append({"w": w, "pos": 0, "last": last,
                                 "writes": {mapping[t] for t in targets}})
                return True
            return False

        def step():
            k = rng.randrange(len(inflight))
            a = inflight[k]
            word.append(a["w"][a["pos"]])
            a["pos"] += 1
            for v in [v for v in a["writes"] if a["last"][v] < a["pos"]]:
                a["writes"].discard(v)        # w is back to its original value
                free.append(v)                # -> reusable while A is in flight
            if a["pos"] >= len(a["w"]):
                inflight.pop(k)

        while len(word) < m_target:
            while len(inflight) < max_inflight and start():
                pass
            if not inflight:
                break
            step()
        while inflight:                        # drain: restores the identity
            step()
        return word

    # -- pipelined v2: unified pool, no permanently-unwritten wires ----------
    #
    # `pipelined` keeps every CONTROL in a read pool that is never written, so
    # those wires hold their input value for the whole word: they are affinely
    # reconstructed with error exactly 0, capping the achievable affine
    # decorrelation at (1 - share) * 0.5.  (Measured n=32..256: the written
    # wires reach 0.475 of 0.5, the never-written ones exactly 0.000.)
    #
    # v2 removes the fixed roles.  A wire is locked only for as long as an atom
    # actually needs it, by the SAME last-touch rule used for write wires:
    #   writer[w]   -- the one atom currently writing w (exclusive), and
    #   readers[w]  -- how many in-flight atoms still read w (shared).
    # A wire may become a WRITE wire only when nobody writes or reads it; it may
    # be taken as a CONTROL only when nobody writes it.  Both locks are dropped
    # as soon as the owning atom's last gate touching that wire has executed --
    # at which point the wire is provably back at its original value.
    # Hence every wire cycles through both roles and none is permanently frozen.
    def pipelined2(self, m_target, max_inflight=None, tries=64):
        n, rng = self.n, self.rng
        if max_inflight is None:
            max_inflight = max(4, n // 3)        # depth must scale with n
        writer = [None] * n
        readers = [0] * n
        inflight, word, next_id = {}, [], 0

        def start():
            nonlocal next_id
            for _ in range(tries):
                atom = rng.choice(self.pool)
                aw = atom["word"]
                targets = sorted(set(g57.targets_of(aw)))
                ctrls = [v for v in g57.wires_of(aw) if v not in targets]
                wfree = [w for w in range(n) if writer[w] is None and readers[w] == 0]
                if len(wfree) < len(targets):
                    continue
                tmap = dict(zip(targets, rng.sample(wfree, len(targets))))
                used = set(tmap.values())
                rfree = [w for w in range(n) if writer[w] is None and w not in used]
                if len(rfree) < len(ctrls):
                    continue
                mapping = dict(tmap)
                mapping.update(zip(ctrls, rng.sample(rfree, len(ctrls))))
                w = g57.relabel(aw, mapping)
                last = {}
                for i, g in enumerate(w):
                    for v in g:
                        last[v] = i          # last gate touching v, any position
                aid = next_id
                next_id += 1
                wr = set(tmap.values())
                rd = {mapping[c] for c in ctrls}
                for v in wr:
                    writer[v] = aid
                for v in rd:
                    readers[v] += 1
                inflight[aid] = {"w": w, "pos": 0, "last": last,
                                 "writes": wr, "reads": rd}
                return True
            return False

        def step():
            aid = rng.choice(list(inflight))
            a = inflight[aid]
            word.append(a["w"][a["pos"]])
            a["pos"] += 1
            for v in [v for v in a["writes"] if a["last"][v] < a["pos"]]:
                a["writes"].discard(v)
                writer[v] = None             # w is back to its original value
            for v in [v for v in a["reads"] if a["last"][v] < a["pos"]]:
                a["reads"].discard(v)
                readers[v] -= 1              # w may now be written by others
            if a["pos"] >= len(a["w"]):
                del inflight[aid]

        while len(word) < m_target:
            while len(inflight) < max_inflight and start():
                pass
            if not inflight:
                break
            step()
        while inflight:
            step()
        return word

    # -- laundering ----------------------------------------------------------
    def _window_support(self, word, i, j):
        return sorted({w for g in word[i:j] for w in g})

    def launder(self, word, steps, max_win=6, verify_every=0):
        """Local rewrites that preserve the global function; erase fingerprints."""
        n = self.n
        rng = self.rng
        rewrites = 0
        for step in range(steps):
            L = len(word)
            if L < 3:
                break
            i = rng.randrange(L - 2)
            hi = min(max_win, L - i)
            wlen = rng.randint(2, hi)
            j = i + wlen
            supp = self._window_support(word, i, j)
            if len(supp) > self.resynth.max_wires:
                # try a commuting reorder instead (cheap, length-preserving)
                if self._commute(word[i], word[i + 1]):
                    word[i], word[i + 1] = word[i + 1], word[i]
                continue
            # length-preserving swap: same function, same length, different gates
            alts = self.resynth.alternatives(word[i:j], exclude_self=True, exact_len=wlen)
            if not alts:
                if self._commute(word[i], word[i + 1]):
                    word[i], word[i + 1] = word[i + 1], word[i]
                continue
            alt = rng.choice(alts)
            word = word[:i] + alt + word[j:]
            rewrites += 1
            if verify_every and rewrites % verify_every == 0:
                assert g57.is_identity_random(word, n, trials=1024), "launder broke identity"
        return word, rewrites

    @staticmethod
    def _commute(g, h):
        a1, x1, y1 = g
        a2, x2, y2 = h
        # gates commute if neither writes a wire the other touches
        return a1 not in (a2, x2, y2) and a2 not in (a1, x1, y1)

    # -- top level -----------------------------------------------------------
    def generate(self, n_atoms, launder_steps, share_frac=0.15, interleave=True):
        placed = self.seed(n_atoms, share_frac)
        word = self.compose(placed, interleave=interleave)
        assert g57.is_identity_random(word, self.n, trials=4096), "composition not identity"
        raw = list(word)
        word, rw = self.launder(word, launder_steps, max_win=4)
        assert g57.is_identity_random(word, self.n, trials=8192), "laundered word not identity"
        return {"raw": raw, "word": word, "atoms": placed, "rewrites": rw}
