# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Chemical Components found in PDB (CCD) constants."""

from collections.abc import ItemsView, Iterator, KeysView, Mapping, Sequence, ValuesView
import dataclasses
import functools
import hashlib
import io
import json
import os
import pathlib
import pickle
import sqlite3
import stat
import tempfile
import threading
import urllib.parse

from alphafold3.common import resources
from alphafold3.common import safe_pickle
from alphafold3.cpp import cif_dict


_CCD_PICKLE_FILE = resources.filename(
    resources.ROOT / 'constants/converters/ccd.pickle'
)


def _load_ccd_original(path):
  with open(path, 'rb') as f:
    return safe_pickle.load(f)


class _LazyCcdDict(dict):
  """An ordered dict whose independent component values decode on first use."""

  def __init__(self, connection, names, source_path):
    super().__init__(dict.fromkeys(names))
    self._pending = set(names)
    self._connection = connection
    self._source_path = source_path
    self._lock = threading.RLock()

  def __iter__(self):
    # Also prevents dict(self) from bypassing the value accessor.
    return super().__iter__()

  def __getitem__(self, key):
    with self._lock:
      return self._getitem_once(key)

  def _getitem_once(self, key):
    if key in self._pending:
      try:
        blob, checksum = self._connection.execute(
            'SELECT data, digest FROM components WHERE name=?', (key,)
        ).fetchone()
        if hashlib.sha256(blob).hexdigest() != checksum:
          raise ValueError('CCD component checksum mismatch')
        value = safe_pickle.load(io.BytesIO(blob))
        if type(value) is not dict:
          raise ValueError('CCD component schema mismatch')
        dict.__setitem__(self, key, value)
        self._pending.remove(key)
      except Exception:
        # A damaged cache never supplies a substitute component value.
        original = _load_ccd_original(self._source_path)
        for name in tuple(self._pending):
          dict.__setitem__(self, name, original[name])
        self._pending.clear()
    return dict.__getitem__(self, key)

  def __setitem__(self, key, value):
    with self._lock:
      self._pending.discard(key)
      dict.__setitem__(self, key, value)

  def __delitem__(self, key):
    with self._lock:
      dict.__delitem__(self, key)
      self._pending.discard(key)

  def get(self, key, default=None):
    return self[key] if key in self else default

  def items(self):
    for key in tuple(self._pending):
      self[key]
    return dict.items(self)

  def values(self):
    for key in tuple(self._pending):
      self[key]
    return dict.values(self)

  def copy(self):
    return {key: self[key] for key in self}

  def update(self, *args, **kwargs):
    for key, value in dict(*args, **kwargs).items():
      self[key] = value

  def clear(self):
    with self._lock:
      dict.clear(self)
      self._pending.clear()

  def setdefault(self, key, default=None):
    if key not in self:
      self[key] = default
    return self[key]

  def pop(self, key, *default):
    if len(default) > 1:
      return dict.pop(self, key, *default)
    if key not in self:
      return dict.pop(self, key, *default)
    value = self[key]
    del self[key]
    return value

  def popitem(self):
    if not self:
      return dict.popitem(self)
    key = next(reversed(self))
    return key, self.pop(key)

  def __eq__(self, other):
    return self.copy() == (other.copy() if isinstance(other, _LazyCcdDict) else other)

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return repr(self.copy())

  def __or__(self, other):
    return self.copy() | other

  def __ror__(self, other):
    return other | self.copy()

  def __ior__(self, other):
    self.update(other)
    return self

  def __reduce_ex__(self, protocol):
    return dict, (self.copy(),)


def _ccd_records_are_independent(data):
  """Only cache the exact supported acyclic, unshared mutable CCD schema."""
  if type(data) is not dict:
    return False
  owners = {}
  for ordinal, (name, component) in enumerate(data.items()):
    if type(name) is not str or type(component) is not dict:
      return False
    for field, values in component.items():
      if (type(field) is not str or type(values) not in (list, tuple)
          or any(type(value) is not str for value in values)):
        return False
      if type(values) is list and owners.setdefault(id(values), ordinal) != ordinal:
        return False
    if owners.setdefault(id(component), ordinal) != ordinal:
      return False
  return True


def _private_ccd_records(path, cache_dir):
  if sqlite3.threadsafety == 0:
    return _load_ccd_original(path)
  with open(path, 'rb') as f:
    source_sha = hashlib.file_digest(f, 'sha256').hexdigest()
  directory = pathlib.Path(cache_dir) / 'private_ccd_records_v2'
  directory.mkdir(mode=0o700, parents=True, exist_ok=True)
  info = directory.lstat()
  if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
      or info.st_mode & 0o077):
    raise ValueError('CCD cache must be a private owned directory')
  filename = directory / (source_sha + '.sqlite')
  if not filename.exists():
    data = _load_ccd_original(path)
    if not _ccd_records_are_independent(data):
      return data
    temporary = None
    try:
      with tempfile.NamedTemporaryFile(dir=directory, delete=False) as f:
        temporary = f.name
      with sqlite3.connect(temporary) as connection:
        connection.execute('PRAGMA journal_mode=OFF')
        connection.execute('CREATE TABLE metadata (source_sha TEXT, count INTEGER, index_sha TEXT)')
        connection.execute(
            'CREATE TABLE components (name TEXT PRIMARY KEY, ordinal INTEGER, data BLOB, digest TEXT)'
        )
        index_sha = hashlib.sha256(json.dumps(list(data)).encode()).hexdigest()
        connection.execute('INSERT INTO metadata VALUES (?,?,?)',
                           (source_sha, len(data), index_sha))
        for ordinal, (name, component) in enumerate(data.items()):
          blob = pickle.dumps(component, protocol=5)
          connection.execute(
              'INSERT INTO components VALUES (?,?,?,?)',
              (name, ordinal, blob, hashlib.sha256(blob).hexdigest()),
          )
      connection.close()
      os.replace(temporary, filename)
      temporary = None
    finally:
      if temporary is not None:
        pathlib.Path(temporary).unlink(missing_ok=True)
  info = filename.lstat()
  if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
      or info.st_mode & 0o077):
    raise ValueError('CCD cache must be a private owned file')
  uri = 'file:' + urllib.parse.quote(str(filename.resolve())) + '?mode=ro&immutable=1'
  connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
  try:
    recorded_sha, count, index_sha = connection.execute(
        'SELECT source_sha,count,index_sha FROM metadata').fetchone()
    names = [r[0] for r in connection.execute('SELECT name FROM components ORDER BY ordinal')]
    if (recorded_sha != source_sha or len(names) != count or len(set(names)) != count
        or hashlib.sha256(json.dumps(names).encode()).hexdigest() != index_sha):
      raise ValueError('CCD cache identity mismatch')
    return _LazyCcdDict(connection, names, path)
  except Exception:
    connection.close()
    raise


@functools.cache
def _load_ccd_pickle_cached(
    path: os.PathLike[str],
) -> dict[str, Mapping[str, Sequence[str]]]:
  """Loads the CCD pickle file and caches it so that it is only loaded once."""
  # Reuse only self-generated component records under the existing explicit
  # cache option. Without it, retain the original eager loader exactly.
  from absl import flags
  cache_dir = (flags.FLAGS['cache_dir'].value
               if flags.FLAGS.is_parsed() and 'cache_dir' in flags.FLAGS else None)
  if cache_dir:
    try:
      return _private_ccd_records(path, cache_dir)
    except Exception:
      pass
  return _load_ccd_original(path)


class Ccd(Mapping[str, Mapping[str, Sequence[str]]]):
  """Chemical Components found in PDB (CCD) constants.

  See https://academic.oup.com/bioinformatics/article/31/8/1274/212200 for CCD
  CIF format documentation.

  Wraps the dict to prevent accidental mutation.
  """

  __slots__ = ('_dict', '_ccd_pickle_path')

  def __init__(
      self,
      ccd_pickle_path: os.PathLike[str] | None = None,
      user_ccd: str | None = None,
  ):
    """Initialises the chemical components dictionary.

    Args:
      ccd_pickle_path: Path to the CCD pickle file. If None, uses the default
        CCD pickle file included in the source code.
      user_ccd: A string containing the user-provided CCD. This has to conform
        to the same format as the CCD, see https://www.wwpdb.org/data/ccd. If
        provided, takes precedence over the CCD for the the same key. This can
        be used to override specific entries in the CCD if desired.
    """
    self._ccd_pickle_path = ccd_pickle_path or _CCD_PICKLE_FILE
    self._dict = _load_ccd_pickle_cached(self._ccd_pickle_path)

    if user_ccd is not None:
      if not user_ccd:
        raise ValueError('User CCD cannot be an empty string.')
      user_ccd_cifs = {
          key: value.to_dict()
          for key, value in cif_dict.parse_multi_data_cif(user_ccd).items()
      }
      self._dict.update(user_ccd_cifs)

  def __getitem__(self, key: object) -> Mapping[str, Sequence[str]]:
    if not isinstance(key, str):
      raise TypeError(f'The CCD key must be a string, got {type(key)}')
    return self._dict[key]

  def __contains__(self, key: object) -> bool:
    return key in self._dict

  def __iter__(self) -> Iterator[str]:
    return self._dict.__iter__()

  def __len__(self) -> int:
    return len(self._dict)

  def __hash__(self) -> int:
    return id(self)  # Ok since this is immutable.

  def get(  # pyrefly: ignore[bad-override]
      self, key: str, default: None | Mapping[str, Sequence[str]] = None
  ) -> Mapping[str, Sequence[str]] | None:
    return self._dict.get(key, default)

  def items(self) -> ItemsView[str, Mapping[str, Sequence[str]]]:
    return self._dict.items()

  def values(self) -> ValuesView[Mapping[str, Sequence[str]]]:
    return self._dict.values()

  def keys(self) -> KeysView[str]:
    return self._dict.keys()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ComponentInfo:
  name: str
  type: str
  pdbx_synonyms: str
  formula: str
  formula_weight: str
  mon_nstd_parent_comp_id: str
  mon_nstd_flag: str
  pdbx_smiles: str


def mmcif_to_info(mmcif: Mapping[str, Sequence[str]]) -> ComponentInfo:
  """Converts CCD mmCIFs to component info. Missing fields are left empty."""
  names = mmcif['_chem_comp.name']
  types = mmcif['_chem_comp.type']
  mon_nstd_parent_comp_ids = mmcif['_chem_comp.mon_nstd_parent_comp_id']
  pdbx_synonyms = mmcif['_chem_comp.pdbx_synonyms']
  formulas = mmcif['_chem_comp.formula']
  formula_weights = mmcif['_chem_comp.formula_weight']

  def front_or_empty(values: Sequence[str]) -> str:
    return values[0] if values else ''

  type_ = front_or_empty(types)
  mon_nstd_parent_comp_id = front_or_empty(mon_nstd_parent_comp_ids)
  if type_.lower() == 'non-polymer':
    # Unset for non-polymers, e.g. water or ions.
    mon_nstd_flag = '.'
  elif mon_nstd_parent_comp_id == '?':
    # A standard component - it doesn't have a standard parent, e.g. MET.
    mon_nstd_flag = 'y'
  else:
    # A non-standard component, e.g. MSE.
    mon_nstd_flag = 'n'

  # Default SMILES is the canonical SMILES, but we fall back to the SMILES if a
  # canonical SMILES is not available. Of canonical SMILES, we prefer ones from
  # the OpenEye OEToolkits program.
  canonical_pdbx_smiles = ''
  fallback_pdbx_smiles = ''
  descriptor_types = mmcif.get('_pdbx_chem_comp_descriptor.type', [])
  descriptors = mmcif.get('_pdbx_chem_comp_descriptor.descriptor', [])
  programs = mmcif.get('_pdbx_chem_comp_descriptor.program', [])
  for descriptor_type, descriptor, program in zip(
      descriptor_types, descriptors, programs
  ):
    if descriptor_type == 'SMILES_CANONICAL':
      if (not canonical_pdbx_smiles) or program == 'OpenEye OEToolkits':
        canonical_pdbx_smiles = descriptor
    if not fallback_pdbx_smiles and descriptor_type == 'SMILES':
      fallback_pdbx_smiles = descriptor
  pdbx_smiles = canonical_pdbx_smiles or fallback_pdbx_smiles

  return ComponentInfo(
      name=front_or_empty(names),
      type=type_,
      pdbx_synonyms=front_or_empty(pdbx_synonyms),
      formula=front_or_empty(formulas),
      formula_weight=front_or_empty(formula_weights),
      mon_nstd_parent_comp_id=mon_nstd_parent_comp_id,
      mon_nstd_flag=mon_nstd_flag,
      pdbx_smiles=pdbx_smiles,
  )


@functools.lru_cache(maxsize=128)
def component_name_to_info(ccd: Ccd, res_name: str) -> ComponentInfo | None:
  component = ccd.get(res_name)
  if component is None:
    return None
  return mmcif_to_info(component)


def type_symbol(ccd: Ccd, res_name: str, atom_name: str) -> str:
  """Returns the element type for the given component name and atom name.

  Args:
    ccd: The chemical components dictionary.
    res_name: The component name, e.g. ARG.
    atom_name: The atom name, e.g. CB, OXT, or NH1.

  Returns:
    Element type, e.g. C for (ARG, CB), O for (ARG, OXT), N for (ARG, NH1).
  """
  res = ccd.get(res_name)
  if res is None:
    return '?'
  try:
    return res['_chem_comp_atom.type_symbol'][
        res['_chem_comp_atom.atom_id'].index(atom_name)
    ]
  except (ValueError, IndexError, KeyError):
    return '?'
