"""Fast, byte-compatible replacement for prody.writePDB as used by run.py.

run.py writes one backbone PDB per designed sequence; only the residue names
(designed sequence) and B-factors (per-residue confidence) change between files.
prody's writePDBStream re-formats every atom in a pure-Python loop each time
(~30 ms/file -> ~2 s per 64-sequence run).

FastPDBWriter extracts and pre-formats every invariant field once per PDB.
Per file it then either
  (a) splices freshly formatted resname/beta columns into a fixed-width
      (n_rows, 81) uint8 template and dumps it with ndarray.tofile
      (fast path, ~0.2 ms/file), or
  (b) falls back to np.char.mod/add string assembly (~4 ms/file), or
  (c) falls back to prody.writePDB unchanged.

Output is byte-identical to prody's writer for the single-model, <100k-atom,
no-ANISOU, no-secondary-structure case; anything more exotic falls back to
prody.writePDB unchanged. The uint8 path additionally requires every rendered
field to be exactly its template width and ASCII, resnames <= 3 chars, and
betas finite with -99.995 < b < 999.995; otherwise it degrades to (b).
"""
import numpy as np
from prody import writePDB as _prody_writePDB
from prody.atomic import AtomGroup
from prody.proteins.pdbfile import AAMAP

# prody's exact line format for the common case (see prody.proteins.pdbfile)
# '%-6s%5d %-4s%1s%-3s%2s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s\n'
# fixed column layout (81 bytes/line incl. '\n'):
#   field_A [0:17]  resname [17:20]  field_C [20:60]  beta [60:66]  field_F [66:81]


class FastPDBWriter:
    def __init__(self, backbone, other_atoms=None):
        self.ok = False
        self.fast_ok = False
        comb = backbone + other_atoms if other_atoms else backbone
        self.remark = str(comb)
        atoms = comb
        # replicate writePDBStream's object resolution
        try:
            atoms.getIndex()
        except AttributeError:
            pass
        else:
            atoms = atoms.select("all")
        self.atoms = atoms
        try:
            coordsets = atoms._getCoordsets(None)
        except AttributeError:
            coordsets = atoms._getCoords()
        if coordsets is None:
            return
        coordsets = np.asarray(coordsets)
        if coordsets.ndim == 2:
            coordsets = coordsets[None]
        if coordsets.shape[0] != 1:
            return  # multi-model: fall back
        n = coordsets.shape[1]
        if n >= 99999:
            return  # hex serial territory: fall back
        self.n = n
        self.coords = coordsets[0]

        if atoms._getAnisous() is not None or atoms._getSecstrs() is not None:
            return  # ANISOU / HELIX / SHEET lines: fall back
        charges = atoms._getCharges()
        if charges is not None and np.any(np.asarray(charges) != 0):
            return  # nonzero charges: rare, fall back

        atomnames = [str(a) for a in atoms.getNames()]
        atomnames = [" " + a if len(a) < 4 else a for a in atomnames]
        altlocs = atoms._getAltlocs()
        altlocs = altlocs if altlocs is not None else np.array([" "] * n)
        resnames = [str(r) for r in atoms._getResnames()]
        self.resnames = resnames
        chainids = atoms._getChids()
        chainids = chainids if chainids is not None else np.array([" "] * n)
        resnums = atoms._getResnums()
        resnums = resnums if resnums is not None else np.ones(n, int)
        if np.any(np.asarray(resnums) > 9999) or np.any(np.asarray(resnums) < -999):
            return  # hex resnum territory: fall back
        icodes = atoms._getIcodes()
        icodes = icodes if icodes is not None else np.array([""] * n)
        hetflags = atoms._getFlags("hetatm")
        if hetflags is None:
            hetflags = atoms._getFlags("hetero")
        hetero = np.where(hetflags, "HETATM", "ATOM  ") if hetflags is not None else np.array(["ATOM  "] * n)
        elements = atoms._getElements()
        elements = np.char.rjust(elements, 2) if elements is not None else np.array([" "] * n)
        segments = atoms._getSegnames()
        segments = segments if segments is not None else np.array([""] * n)
        occupancies = atoms._getOccupancies()
        occupancies = occupancies if occupancies is not None else np.zeros(n)
        self.betas_other = atoms._getBetas()
        if self.betas_other is None:
            self.betas_other = np.zeros(n)

        # TER lines: same branch condition as prody (AtomMap is not an AtomGroup)
        if isinstance(atoms, AtomGroup):
            terflags = atoms._getFlags("pdbter")
        else:
            terflags = atoms._getFlags("selpdbter")
        if terflags is None:
            terflags = np.zeros(n, bool)
        self.ter_after = np.array(
            [bool(terflags[i]) and resnames[i] in AAMAP for i in range(n)]
        )

        # serials with TER offsets (renumber=True semantics)
        serials = np.arange(n, dtype=int) + 1
        ter_count = np.cumsum(self.ter_after)
        serials = serials + np.concatenate([[0], ter_count[:-1]])
        self.ter_serials = serials[self.ter_after] + 1

        # invariant per-atom string fields (Python '%' formatting == prody bytes)
        self.field_A = np.array(
            [
                "%-6s%5d %-4s%1s" % (hetero[i], serials[i], atomnames[i], str(altlocs[i]))
                for i in range(n)
            ]
        )
        self.field_C = np.array(
            [
                "%2s%4d%1s   %8.3f%8.3f%8.3f%6.2f"
                % (
                    str(chainids[i]),
                    int(resnums[i]),
                    str(icodes[i]),
                    self.coords[i, 0],
                    self.coords[i, 1],
                    self.coords[i, 2],
                    float(occupancies[i]),
                )
                for i in range(n)
            ]
        )
        self.field_F = np.array(
            [
                "      %4s%2s  \n" % (str(segments[i]), str(elements[i]))
                for i in range(n)
            ]
        )
        # TER line templates: first 17 cols of the line + resname slot + 61 tail chars
        self.ter_pre = []   # before resname
        self.ter_post = []  # after resname
        for serial, i in zip(self.ter_serials, np.nonzero(self.ter_after)[0]):
            self.ter_pre.append("TER   %5d      " % serial)  # 6+5+1+4+1 = 17 chars
            self.ter_post.append(
                "%2s%4d" % (str(chainids[i]), int(resnums[i])) + " " * 54 + "\n"
            )
        self.ter_idx = np.nonzero(self.ter_after)[0]
        self.ok = True

        # ---- fixed-width uint8 template for the fast write path ----
        # Valid only if every invariant field renders at exactly its column
        # width and is ASCII; the reshape/encode below fails otherwise.
        try:
            n_ter = len(self.ter_idx)
            n_rows = n + n_ter
            row_of_atom = np.arange(n) + np.concatenate([[0], ter_count[:-1]])
            buf = np.zeros((n_rows, 81), dtype=np.uint8)
            buf[row_of_atom, 0:17] = np.frombuffer(
                "".join(self.field_A).encode("ascii"), dtype=np.uint8
            ).reshape(n, 17)
            buf[row_of_atom, 20:60] = np.frombuffer(
                "".join(self.field_C).encode("ascii"), dtype=np.uint8
            ).reshape(n, 40)
            buf[row_of_atom, 66:81] = np.frombuffer(
                "".join(self.field_F).encode("ascii"), dtype=np.uint8
            ).reshape(n, 15)
            if n_ter:
                ter_rows = row_of_atom[self.ter_idx] + 1
                buf[ter_rows, 0:17] = np.frombuffer(
                    "".join(self.ter_pre).encode("ascii"), dtype=np.uint8
                ).reshape(n_ter, 17)
                buf[ter_rows, 20:81] = np.frombuffer(
                    "".join(self.ter_post).encode("ascii"), dtype=np.uint8
                ).reshape(n_ter, 61)
            else:
                ter_rows = self.ter_idx  # empty
            self.buf = buf
            self.row_of_atom = row_of_atom
            self.ter_rows = ter_rows
            self.header_bytes = ("REMARK " + self.remark + "\n").encode()
            self.fast_ok = True
        except (UnicodeEncodeError, ValueError, TypeError):
            self.fast_ok = False

    def _write_fast(self, filename, rn, bf):
        """uint8 fixed-width path. Returns filename, or None if a guard fails
        (caller then uses the np.char path)."""
        rn = np.asarray(rn)
        if rn.dtype.kind not in "US":
            rn = rn.astype(str)
        if rn.ndim != 1 or rn.shape[0] != self.n:
            return None
        if np.char.str_len(rn).max(initial=0) > 3:
            return None  # '%-3s' would exceed the column: use np.char path
        cp = np.frombuffer(rn.astype("<U3").tobytes(), dtype=np.uint32).reshape(self.n, 3)
        if cp.size and cp.max() > 127:
            return None  # non-ASCII resname
        cp = np.where(cp == 0, 32, cp).astype(np.uint8)  # null-pad -> spaces

        bfd = np.asarray(bf, dtype=np.float64)
        if bfd.ndim != 1 or bfd.shape[0] != self.n:
            return None
        if not np.all(np.isfinite(bfd)) or np.any(bfd >= 999.995) or np.any(bfd <= -99.995):
            return None  # '%6.2f' would exceed the column (or non-finite)
        # '%6.2f' byte-exact vectorized: float32->float64 is exact, x100 is exact
        # (24-bit mantissa x 5-bit mantissa < 53 bits), rint is half-even like
        # glibc printf. Sign comes from signbit so -0.00 renders like printf.
        r = np.rint(bfd * 100).astype(np.int64)
        neg = np.signbit(bfd)
        ar = np.abs(r)
        whole = ar // 100
        frac = ar % 100
        c0 = np.where(whole >= 100, 48 + (whole // 100) % 10, np.where(neg & (whole >= 10), 45, 32))
        c1 = np.where(whole >= 10, 48 + (whole // 10) % 10, np.where(neg, 45, 32))
        bb = np.stack(
            [c0, c1, 48 + whole % 10, np.full(self.n, 46), 48 + frac // 10, 48 + frac % 10],
            axis=1,
        ).astype(np.uint8)

        buf = self.buf
        buf[self.row_of_atom, 17:20] = cp
        buf[self.ter_rows, 17:20] = cp[self.ter_idx]
        buf[self.row_of_atom, 60:66] = bb
        with open(filename, "wb") as f:
            f.write(self.header_bytes)
            buf.tofile(f)
            f.write(b"END   \n")
        return filename

    def write(self, filename, resnames=None, betas=None):
        """Write one PDB. resnames/betas: full per-atom arrays (len n) or None
        to reuse the object's current values."""
        if not self.ok:
            return _prody_writePDB(filename, self.atoms)
        rn = resnames if resnames is not None else self.atoms._getResnames()
        bf = betas if betas is not None else self.atoms._getBetas()
        if self.fast_ok:
            try:
                out = self._write_fast(filename, rn, bf)
            except Exception:
                out = None
            if out is not None:
                return out
        field_B = np.char.mod("%-3s", np.asarray(rn).astype(str))
        field_E = np.char.mod("%6.2f", np.asarray(bf, dtype=float))
        lines = np.char.add(self.field_A, field_B)
        lines = np.char.add(lines, self.field_C)
        lines = np.char.add(lines, field_E)
        lines = np.char.add(lines, self.field_F)
        parts = ["REMARK ", self.remark, "\n"]
        prev = 0
        for k, i in enumerate(self.ter_idx):
            parts.append("".join(lines[prev : i + 1]))
            parts.append(self.ter_pre[k] + "%-3s" % str(rn[i]) + self.ter_post[k])
            prev = i + 1
        parts.append("".join(lines[prev:]))
        parts.append("END   \n")
        with open(filename, "w") as f:
            f.write("".join(parts))
        return filename
