"""Real data for v3: ONE reader for PRMI, the chicory rhizotron sets, the frozen
real splits and the synthetic hand-off -- shared by notebook 03 v3 (simulator
width target, CUT style pool) and notebook 04 v2 (training and tests).

Two earlier designs each specified this module with a different API, tube key,
scale clamp, sample size and cache path; as designed, CUT and 04 would have
resampled the same PRMI subset differently.  Everything both notebooks must
agree on therefore lives here, and every derived table is written ONCE to
``PROJECT_ROOT/realdata/`` (whichever notebook first needs it, atomically) and
only read afterwards:

    prmi_index.csv        one row per PRMI image (pairing, tube, flags)
    prmi_widths_<h>.csv   per-image mask widths (w2e) of a tube-balanced sample
    scale_table.csv       per-subset resampling factor ``f`` to 11.81 px/mm
    real_splits.csv       frozen roles of every real unit (Ginger scan, PRMI
                          tube, chicory panel-side, Zenodo photo); independent
                          of every notebook switch (PRMI_USE_NOMASK is applied
                          after reading, by prmi_apply_nomask), and PRMI images
                          are selected through prmi_image_roles
    grid_index.csv        grid_counted images, root intensity, length from RI

Facts about the data that the code depends on (verified on the files):

* PRMI (Dryad 10.5061/dryad.2v6wwpzp4): the 9.75 GB outer zip holds README and
  ``PRMI_official.zip`` STORED (not compressed) with its data at byte 47, so
  the inner zip is read IN PLACE through a byte window -- the notebooks used to
  read all 9.75 GB into RAM.  Layout ``PRMI_official/<split>/{images,
  masks_pixel_gt,labels_image_gt}/<Subset>...``.  Camera images are .jpg and
  their masks ``GT_<stem>.png`` -- pairing on the full file name is why Cotton
  once found 0 pairs.  38 val JSON records say ``binary_mask: 'N/A'`` although
  their GT files exist, so that field is never used.  The Switchgrass noMask
  JSON has no split suffix and no masks folder.
* PRMI scale is known only for Switchgrass (300 dpi = 11.81 px/mm, the
  synthetic scale).  The camera subsets' "DPI" is not physical, so their scale
  is estimated by matching mask widths to Switchgrass (``scale_table``).
* Chicory (Smith et al. 2020, Zenodo 3527713, CC-BY-4.0): grid_counted is 867
  3991x1842 photos named ``YY_MM_DD_<panel><E|W><depth><a-d>_P<photo>``, with
  root intensity (RI) in a separate headerless ``stem,ri`` csv.  RI is a count,
  never a training loss.  The Zenodo train/val/test zips hold the only pixel
  annotations (test-only by default; never a CUT target).

Widths everywhere use ``mrsim.traits.skeleton_widths_px`` (w2e = 2 x exact EDT
at the skeleton): a 1-px line is 2, a 3-px bar 4.

Everything here is Python 3.8 compatible and imports only numpy / pandas /
OpenCV / scipy / skimage (lazily where possible); nothing touches torch or TF.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import threading
import warnings
import zipfile
import zlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None

# --------------------------------------------------------------------------
# constants (the agreed contract)
# --------------------------------------------------------------------------
# reference scale = ginger.CI600_PX_PER_MM (CI-600 scans, the synthetic frames,
# PRMI Switchgrass); repeated here so this module imports without ginger
REF_PX_PER_MM = 2551 / 216.0

REALDATA_DIRNAME = "realdata"
FILES = {"prmi_index": "prmi_index.csv", "scale_table": "scale_table.csv",
         "real_splits": "real_splits.csv", "grid_index": "grid_index.csv"}

PRMI_MEMBER = "PRMI_official.zip"
PRMI_SPLITS = ("train", "val", "test")
PRMI_REF_SUBSET = "Switchgrass_720x510_DPI300"
NOMASK_SUFFIX = "_noMask"
# masked images per split in the full dataset (the notebooks assert these)
PRMI_EXPECTED_MASKED = {"train": 38058, "val": 11785, "test": 14100}

SCALE_CLAMP = (0.5, 2.0)          # 04 multiplies f by a 0.6-1.6 augmentation and
                                  # clamps again; outside this a 384 crop is
                                  # mostly padding or a blown-up blur
CHICORY_PX_PER_MM = 12.0          # geometric mean of ~13.3 (x) and ~10.8 (y)
CHICORY_IMAGE_WH = (3991, 1842)
CHICORY_CUT_PANELS = "10E,11W"    # panel-sides CUT may use as style targets

SYN_ID_RE = re.compile(r"^s\d{3,}_\d{4}$")
ROLES = ("train", "val", "test", "excluded")
SPLIT_COLS = ["source", "unit_uid", "role", "cut_allowed", "reason"]
# the fixed reason of the noMask tubes a notebook MAY opt into (prmi_apply_nomask)
PRMI_NOMASK_OPTIN = "noMask: has_root only, opt-in (PRMI_USE_NOMASK)"


# --------------------------------------------------------------------------
# cache directory and atomic writers
# --------------------------------------------------------------------------
def realdata_dir(project_root: str, create: bool = True) -> str:
    """``PROJECT_ROOT/realdata`` -- the one cache both notebooks read."""
    d = os.path.join(project_root, REALDATA_DIRNAME)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def cache_path(project_root: str, key: str, h: Optional[str] = None) -> str:
    """Path of a cache file: ``key`` in :data:`FILES`, or 'prmi_widths' with a hash."""
    if key == "prmi_widths":
        if not h:
            raise ValueError("prmi_widths needs the settings hash h")
        name = "prmi_widths_%s.csv" % h
    else:
        name = FILES[key]
    return os.path.join(realdata_dir(project_root), name)


def short_hash(obj, n: int = 8) -> str:
    """Stable short hash of a JSON-able object (sorted keys)."""
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def file_sha1(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _tmp_for(path: str) -> str:
    # pid + thread: two writers of the same target never share a temp file
    return "%s.tmp%d_%d" % (path, os.getpid(), threading.get_ident())


def _replace_from(tmp: str, path: str) -> str:
    try:
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def write_bytes_atomic(data: bytes, path: str) -> str:
    """Write to a temp file next to ``path``, then ``os.replace``.

    On the Drive FUSE mount a half-written file is otherwise left behind by a
    disconnect, and the next session reads it as if it were complete (v1's
    corrupt settings.json files).
    """
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = _tmp_for(path)
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return _replace_from(tmp, path)


def write_text_atomic(text: str, path: str) -> str:
    return write_bytes_atomic(text.encode("utf-8"), path)


def write_json_atomic(obj, path: str, indent: int = 1) -> str:
    return write_text_atomic(json.dumps(obj, indent=indent, sort_keys=True, default=str), path)


def write_csv_atomic(df, path: str, **kw) -> str:
    kw.setdefault("index", False)
    return write_text_atomic(df.to_csv(**kw), path)


def copy_atomic(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = _tmp_for(dst)
    try:
        shutil.copyfile(src, tmp)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return _replace_from(tmp, dst)


def load_or_build(path: str, build: Callable[[], "object"], force: bool = False,
                  read: Optional[Callable[[str], "object"]] = None, verbose: bool = True):
    """Read a cached csv, or build it, write it atomically and return it.

    The 'whichever notebook first needs it writes it' rule in one place.
    """
    import pandas as pd
    read = read or pd.read_csv
    if os.path.exists(path) and not force:
        if verbose:
            print("cached:", path)
        return read(path)
    df = build()
    write_csv_atomic(df, path)
    if verbose:
        print("wrote:", path)
    return df


# --------------------------------------------------------------------------
# zip access: a STORED member read in place, one handle per process and thread
# --------------------------------------------------------------------------
class OffsetFile(io.RawIOBase):
    """Read-only, seekable window ``[start, start + size)`` of a file.

    Lets ``zipfile`` open a zip that is itself a STORED member of another zip
    without extracting or reading it: offsets inside the window are offsets
    inside the member.  Each instance owns its own OS file handle.
    """

    def __init__(self, path: str, start: int = 0, size: Optional[int] = None):
        super().__init__()
        total = os.path.getsize(path)
        if size is None:
            size = total - start
        if start < 0 or size < 0 or start + size > total:
            raise ValueError("window [%d, %d) outside %s (%d bytes)"
                             % (start, start + size, path, total))
        self.name = path
        self._f = open(path, "rb")
        self._start, self._size, self._pos = int(start), int(size), 0

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self._size + offset
        else:
            raise ValueError("bad whence %r" % (whence,))
        if new < 0:
            raise ValueError("negative seek position %d" % new)
        self._pos = int(new)
        return self._pos

    def readinto(self, b) -> int:
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        self._f.seek(self._start + self._pos)
        got = self._f.readinto(memoryview(b)[:n])
        got = got or 0
        self._pos += got
        return got

    def close(self) -> None:
        if not self.closed:
            try:
                self._f.close()
            finally:
                super().close()


def stored_member_span(outer: str, member: str) -> Tuple[int, int]:
    """``(data_start, size)`` of a STORED, unencrypted member of ``outer``.

    The data starts after the member's LOCAL header, whose name/extra lengths
    can differ from the central directory's, so they are read from the local
    header itself: ``header_offset + 30 + name_len + extra_len`` (47 for the
    PRMI zip).  A compressed member cannot be read through a byte window and is
    refused -- extract or copy it instead.
    """
    with zipfile.ZipFile(outer) as zf:
        try:
            zi = zf.getinfo(member)
        except KeyError:
            raise KeyError("%s has no member %r (members: %s)"
                           % (outer, member, ", ".join(zf.namelist()[:10])))
    if zi.flag_bits & 0x1:
        raise ValueError("%s in %s is encrypted" % (member, outer))
    if zi.compress_type != zipfile.ZIP_STORED:
        raise ValueError("%s in %s is compressed (method %d), so it cannot be read in place; "
                         "extract it once instead" % (member, outer, zi.compress_type))
    with open(outer, "rb") as fh:
        fh.seek(zi.header_offset)
        h = fh.read(30)
    if len(h) != 30 or h[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile("no local header for %s at %d" % (member, zi.header_offset))
    n, m = struct.unpack("<HH", h[26:30])
    return zi.header_offset + 30 + n + m, int(zi.compress_size)


class _OwnedZipFile(zipfile.ZipFile):
    """A ZipFile that also closes the file object it was given."""

    def __init__(self, fp):
        try:
            super().__init__(fp, "r")
        except BaseException:
            fp.close()
            raise
        self._owned_fp = fp

    def close(self):
        try:
            super().close()
        finally:
            fp = self.__dict__.pop("_owned_fp", None)
            if fp is not None and not fp.closed:
                fp.close()


def open_nested_stored(outer: str, member: str = PRMI_MEMBER,
                       buffer_size: int = 1 << 20) -> zipfile.ZipFile:
    """``ZipFile`` of a STORED zip nested in ``outer``, read in place.

    Nothing is extracted and the member is never ``read()`` whole; closing the
    returned ZipFile closes its file handle.  For many threads or DataLoader
    workers use :class:`LazyNestedZip` instead (one handle each).
    """
    start, size = stored_member_span(outer, member)
    return _OwnedZipFile(io.BufferedReader(OffsetFile(outer, start, size), buffer_size))


def _read_member(fp, zi: zipfile.ZipInfo) -> bytes:
    """Bytes of one member, from its own local header (STORED or DEFLATED)."""
    if zi.flag_bits & 0x1:
        raise ValueError("%s is encrypted" % zi.filename)
    fp.seek(zi.header_offset)
    h = fp.read(30)
    if len(h) != 30 or h[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile("bad local header for %s" % zi.filename)
    n, m = struct.unpack("<HH", h[26:30])
    fp.seek(zi.header_offset + 30 + n + m)
    data = fp.read(zi.compress_size)
    if len(data) != zi.compress_size:
        raise zipfile.BadZipFile("%s is truncated" % zi.filename)
    if zi.compress_type == zipfile.ZIP_STORED:
        out = data
    elif zi.compress_type == zipfile.ZIP_DEFLATED:
        out = zlib.decompress(data, -15)
    else:
        raise NotImplementedError("%s uses compression method %d"
                                  % (zi.filename, zi.compress_type))
    if (zlib.crc32(out) & 0xFFFFFFFF) != zi.CRC:
        raise zipfile.BadZipFile("CRC mismatch in %s" % zi.filename)
    return out


class LazyNestedZip:
    """Random member reads from a (possibly nested, STORED) zip, safe everywhere.

    * one OS file handle per (process, thread), opened on first use: ``zipfile``
      serialises every read on one handle behind a lock, so a shared ZipFile
      makes a thread pool or DataLoader effectively single-threaded;
    * the central directory is parsed ONCE (136,598 entries for PRMI; a forked
      child inherits it) and member reads go straight to their local header --
      opening a full ZipFile per thread would re-parse it every time;
    * a fork or a spawned worker (it pickles to its path) re-opens its handles
      lazily, so it can be handed to DataLoader workers and process pools.

    ``member=None`` reads ``path`` itself; ``member='PRMI_official.zip'`` reads
    the zip stored inside it.  :func:`prmi_zip` picks the right one.
    """

    def __init__(self, path: str, member: Optional[str] = None, buffer_size: int = 1 << 20):
        self.path = path
        self.member = member
        self.buffer_size = int(buffer_size)
        self._init_runtime()

    # -- process / thread bookkeeping ------------------------------------
    def _init_runtime(self):
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._local = threading.local()
        self._handles: List[io.BufferedReader] = []
        self._infos: Optional[Dict[str, zipfile.ZipInfo]] = None
        self._span: Optional[Tuple[int, int]] = None

    def _check_pid(self):
        if self._pid != os.getpid():
            # forked: handles and locks belong to the parent; the parsed
            # directory and the member span are plain data and stay valid
            infos, span = self._infos, self._span
            self._init_runtime()
            self._infos, self._span = infos, span

    def __getstate__(self):
        return {"path": self.path, "member": self.member, "buffer_size": self.buffer_size}

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._init_runtime()

    def span(self) -> Tuple[int, int]:
        self._check_pid()
        if self._span is None:
            self._span = (stored_member_span(self.path, self.member) if self.member
                          else (0, os.path.getsize(self.path)))
        return self._span

    def _new_handle(self) -> io.BufferedReader:
        start, size = self.span()
        fp = io.BufferedReader(OffsetFile(self.path, start, size), self.buffer_size)
        with self._lock:
            self._handles.append(fp)
        return fp

    def _handle(self) -> io.BufferedReader:
        self._check_pid()
        fp = getattr(self._local, "fp", None)
        if fp is None or fp.closed:
            fp = self._new_handle()
            self._local.fp = fp
        return fp

    # -- the zip API -----------------------------------------------------
    def infos(self) -> Dict[str, zipfile.ZipInfo]:
        self._check_pid()
        if self._infos is None:
            with self._lock:
                if self._infos is None:
                    start, size = self.span()
                    with _OwnedZipFile(io.BufferedReader(OffsetFile(self.path, start, size),
                                                         self.buffer_size)) as zf:
                        self._infos = {i.filename: i for i in zf.infolist()}
        return self._infos

    def namelist(self) -> List[str]:
        return list(self.infos().keys())

    def __contains__(self, name: str) -> bool:
        return name in self.infos()

    def getinfo(self, name: str) -> zipfile.ZipInfo:
        try:
            return self.infos()[name]
        except KeyError:
            raise KeyError("no member %r in %s%s" % (name, self.path,
                                                     ("!" + self.member) if self.member else ""))

    def read(self, name: str) -> bytes:
        return _read_member(self._handle(), self.getinfo(name))

    def open_zipfile(self) -> zipfile.ZipFile:
        """A full ZipFile on a NEW handle of this thread (the caller closes it)."""
        start, size = self.span()
        return _OwnedZipFile(io.BufferedReader(OffsetFile(self.path, start, size),
                                               self.buffer_size))

    def close(self):
        """Close every handle this process opened (call from the owning process)."""
        self._check_pid()
        with self._lock:
            hs, self._handles = self._handles, []
        for fp in hs:
            try:
                fp.close()
            except Exception:                            # pragma: no cover
                pass
        self._local = threading.local()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return "LazyNestedZip(%r, member=%r)" % (self.path, self.member)


def prmi_zip(path: str, member: str = PRMI_MEMBER) -> LazyNestedZip:
    """Open PRMI whether ``path`` is the Dryad outer zip or ``PRMI_official.zip``."""
    with zipfile.ZipFile(path) as zf:
        nested = member in zf.NameToInfo
    return LazyNestedZip(path, member if nested else None)


def _as_reader(z):
    """A path becomes a LazyNestedZip (PRMI-aware); readers pass through."""
    if isinstance(z, str):
        return prmi_zip(z)
    return z


# --------------------------------------------------------------------------
# image decoding
# --------------------------------------------------------------------------
def decode_unchanged(data: bytes) -> np.ndarray:
    """Decoded image in RGB / RGBA / grey channel order (not OpenCV's BGR)."""
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError("cannot decode image (%d bytes)" % len(data))
    if im.ndim == 3 and im.shape[2] == 3:
        im = im[:, :, ::-1]
    elif im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, [2, 1, 0, 3]]
    return np.ascontiguousarray(im)


def decode_rgb(data: bytes) -> np.ndarray:
    """HxWx3 uint8 RGB from encoded bytes (grey, RGBA and 16-bit handled)."""
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise IOError("cannot decode image (%d bytes)" % len(data))
    return np.ascontiguousarray(im[:, :, ::-1])


def decode_gray(data: bytes) -> np.ndarray:
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError("cannot decode image (%d bytes)" % len(data))
    return im


def _mask_from_decoded(im: np.ndarray) -> np.ndarray:
    """PRMI masks are RGBA with R=G=B in {0,255}: channel 0 > 127."""
    if im.ndim == 3:
        im = im[:, :, 0]
    if im.dtype != np.uint8:
        return im > (np.iinfo(im.dtype).max // 2 if np.issubdtype(im.dtype, np.integer) else 0.5)
    return im > 127


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------
def _norm_str(v) -> str:
    """JSON / csv value -> clean string ('' for None/NaN, '4' for 4.0)."""
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if v.is_integer():
            return str(int(v))
    return str(v).strip()


def _to_int(v, default: int = -1) -> int:
    try:
        f = float(v)
        return int(f) if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def safe_name(s: str) -> str:
    """File-name-safe id: '/' -> '__', anything unusual -> '_'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(s).replace("/", "__"))


def width_hist(w, edges: Optional[Sequence[float]] = None) -> np.ndarray:
    """Counts of width samples per ``[lo, hi)`` bin (default WIDTH_EDGES -> h00..h10)."""
    if edges is None:
        from mrsim.traits import WIDTH_EDGES
        edges = WIDTH_EDGES
    e = np.asarray(edges, np.float64)
    w = np.asarray(w, np.float64).ravel()
    w = w[np.isfinite(w)]
    idx = np.searchsorted(e, w, side="right") - 1
    ok = (idx >= 0) & (idx < len(e) - 1)
    return np.bincount(idx[ok], minlength=len(e) - 1).astype(np.int64)


def hist_cols(edges: Optional[Sequence[float]] = None) -> List[str]:
    if edges is None:
        from mrsim.traits import WIDTH_EDGES
        edges = WIDTH_EDGES
    return ["h%02d" % i for i in range(len(edges) - 1)]


def balanced_sample(df, unit_col: str, n: int, seed: int = 0):
    """Up to ``n`` rows, round-robin over units (tubes, scans, photos).

    One row from every unit, then a second from every unit that has one, and
    so on, in a seeded random unit and row order -- so a subset with one
    prolific tube (Peanut has tubes with hundreds of frames) cannot fill the
    sample, and the same seed always gives the same rows.  Returned in
    selection order with the original index.
    """
    if n <= 0 or len(df) == 0:
        return df.iloc[:0]
    rng = np.random.default_rng(seed)
    d = df.iloc[rng.permutation(len(df))]
    units = d[unit_col].astype(str)
    order = {u: i for i, u in enumerate(rng.permutation(sorted(units.unique())))}
    rank = d.groupby(units.values, sort=False).cumcount().to_numpy()
    uord = units.map(order).to_numpy()
    sel = np.lexsort((uord, rank))[:int(n)]
    return d.iloc[sel]


# --------------------------------------------------------------------------
# PRMI: index, read
# --------------------------------------------------------------------------
_PRMI_JSON_RE = re.compile(
    r"^(?P<root>(?:.*/)?)(?P<split>train|val|test)/labels_image_gt/(?P<stem>[^/]+)\.json$")

PRMI_INDEX_COLS = ["prmi_id", "split", "subset", "crop", "image_name", "image_member",
                   "mask_member", "has_image", "has_mask", "has_root", "nomask", "location",
                   "tube_num", "tube_uid", "site_uid", "date", "depth", "dpi", "row_idx",
                   "col_idx", "shared_with_test"]
_PRMI_STR_COLS = ["prmi_id", "split", "subset", "crop", "image_name", "image_member",
                  "mask_member", "location", "tube_num", "tube_uid", "site_uid", "date",
                  "depth", "dpi"]


def prmi_base_subset(subset: str) -> str:
    """'Switchgrass_720x510_DPI300_noMask' -> 'Switchgrass_720x510_DPI300'."""
    return subset[:-len(NOMASK_SUFFIX)] if subset.endswith(NOMASK_SUFFIX) else subset


def _json_records(obj) -> List[dict]:
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for v in obj.values():                         # {"images": [...]} or {name: {...}}
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
        return [v for v in obj.values() if isinstance(v, dict)]
    return []


def prmi_index(z, verbose: bool = True):
    """One row per PRMI image, built from ``labels_image_gt/*.json`` + the namelist.

    ``z`` is a :class:`LazyNestedZip`, a ZipFile, or a path.  Columns are
    :data:`PRMI_INDEX_COLS`:

    * ``mask_member`` = ``<split>/masks_pixel_gt/<subset>/GT_<stem>.png`` when
      that member exists, else '' -- the JSON ``binary_mask`` field is ignored
      (it says 'N/A' for 38 val images whose masks exist);
    * ``tube_uid = subset|location|tube_num`` is the split unit; ``site_uid =
      location|tube_num`` ignores the crop and is for leakage checks only;
    * ``nomask``: rows of the Switchgrass noMask JSON (no split suffix, train
      only, no masks: ``has_mask`` False);
    * ``shared_with_test``: the row's PHYSICAL tube (subset without '_noMask',
      location, tube) has images in test AND in another split -- a tube
      property, set on all its rows.  The official test keeps its own images
      of such a tube; :func:`prmi_roles` keeps them out of train and val.
    """
    import pandas as pd
    z = _as_reader(z)
    names = z.namelist()
    nameset = set(names)
    # stem lookup per (split, folder kind, subset folder) for names the JSON
    # gives without (or with a different) extension
    folder_stems: Dict[Tuple[str, str], Dict[str, str]] = {}
    jsons = []
    for nm in names:
        mj = _PRMI_JSON_RE.match(nm)
        if mj:
            jsons.append((nm, mj))
            continue
        parts = nm.split("/")
        if len(parts) >= 4 and parts[-3] == "images" and parts[-1]:
            key = ("/".join(parts[:-2]), parts[-2])        # (<root><split>/images, subset)
            folder_stems.setdefault(key, {})[os.path.splitext(parts[-1])[0]] = nm
    if not jsons:
        raise ValueError("no <split>/labels_image_gt/*.json in this zip -- is it PRMI_official?")

    rows = []
    for nm, mj in sorted(jsons, key=lambda t: t[0]):
        root, split, stem = mj.group("root"), mj.group("split"), mj.group("stem")
        subset = stem[:-(len(split) + 1)] if stem.endswith("_" + split) else stem
        nomask = subset.endswith(NOMASK_SUFFIX)
        recs = _json_records(json.loads(z.read(nm).decode("utf-8")))
        img_dir = "%s%s/images" % (root, split)
        for r in recs:
            image_name = _norm_str(r.get("image_name")).split("/")[-1]
            if not image_name:
                continue
            istem = os.path.splitext(image_name)[0]
            img = "%s/%s/%s" % (img_dir, subset, image_name)
            if img not in nameset:
                img = ""
                for sub in (subset, prmi_base_subset(subset)):
                    img = folder_stems.get((img_dir, sub), {}).get(istem, "")
                    if img:
                        break
            msk = "%s%s/masks_pixel_gt/%s/GT_%s.png" % (root, split, subset, istem)
            has_mask = msk in nameset
            loc = _norm_str(r.get("location"))
            tube = _norm_str(r.get("tube_num"))
            crop = _norm_str(r.get("crop")) or subset.split("_")[0]
            rows.append({
                "prmi_id": "prmi/%s/%s" % (subset, istem), "split": split,
                "subset": subset, "crop": crop, "image_name": image_name,
                "image_member": img, "mask_member": msk if has_mask else "",
                "has_image": bool(img), "has_mask": bool(has_mask),
                "has_root": _to_int(r.get("has_root"), -1), "nomask": bool(nomask),
                "location": loc, "tube_num": tube,
                "tube_uid": "%s|%s|%s" % (subset, loc, tube), "site_uid": "%s|%s" % (loc, tube),
                "date": _norm_str(r.get("date")), "depth": _norm_str(r.get("depth")),
                "dpi": _norm_str(r.get("dpi")),
                "row_idx": _to_int(r.get("row_idx"), -1), "col_idx": _to_int(r.get("col_idx"), -1),
            })
    df = pd.DataFrame(rows, columns=[c for c in PRMI_INDEX_COLS if c != "shared_with_test"])
    if df.empty:
        df["shared_with_test"] = pd.Series([], dtype=bool)
        return df

    dup = df.prmi_id.duplicated(keep=False)
    if dup.any():
        warnings.warn("%d PRMI rows share a prmi_id; suffixed with '@<split>'" % int(dup.sum()))
        df.loc[dup, "prmi_id"] = df.loc[dup, "prmi_id"] + "@" + df.loc[dup, "split"]
        dup2 = df.prmi_id.duplicated(keep="first")
        if dup2.any():                                   # same id within one split: keep first
            df = df[~dup2].reset_index(drop=True)

    phys = _physical_tube(df)
    splits = {k: set(df.loc[v, "split"]) for k, v in phys.groupby(phys).groups.items()}
    shared = {k for k, v in splits.items() if "test" in v and len(v) > 1}
    df["shared_with_test"] = phys.isin(shared).to_numpy()
    order = {s: i for i, s in enumerate(PRMI_SPLITS)}
    df = df.assign(_o=df.split.map(order)).sort_values(["_o", "subset", "image_name"],
                                                       kind="mergesort")
    df = df.drop(columns="_o").reset_index(drop=True)
    if verbose:
        m = df[df.has_mask]
        print("PRMI index: %d images (%d with masks: %s), %d without image member, "
              "%d tubes shared with test"
              % (len(df), len(m), m.split.value_counts().reindex(list(PRMI_SPLITS)).fillna(0)
                 .astype(int).to_dict(), int((~df.has_image).sum()), len(shared)))
    return df


def _physical_tube(df):
    """subset-without-noMask | location | tube: the same tube in the noMask set."""
    return (df.subset.map(prmi_base_subset) + "|" + df.location.astype(str) + "|"
            + df.tube_num.astype(str))


def read_prmi_index(path: str):
    """Read ``prmi_index.csv`` back with its string columns kept as strings."""
    import pandas as pd
    df = pd.read_csv(path, dtype={c: str for c in _PRMI_STR_COLS}, keep_default_na=False)
    for c in ("has_image", "has_mask", "nomask", "shared_with_test"):
        if c in df and df[c].dtype != bool:
            df[c] = df[c].astype(str).str.lower().isin(("true", "1"))
    return df


def prmi_check_totals(index, expected: Optional[Dict[str, int]] = None) -> Dict[str, Tuple[int, int]]:
    """``{split: (masked images found, expected)}`` -- the SP1 check."""
    expected = PRMI_EXPECTED_MASKED if expected is None else expected
    m = index[index.has_mask.astype(bool)]
    return {s: (int((m.split == s).sum()), int(expected.get(s, -1))) for s in PRMI_SPLITS}


def prmi_shared_tubes(index):
    """Physical tubes with images in more than one split (Switchgrass: 1+1).

    Columns: tube_key, subset, location, tube_num, splits ('train+test'),
    n_train, n_val, n_test, with_test.  The noMask set is folded onto its
    parent subset, so its tubes that overlap val/test show up here too.
    """
    import pandas as pd
    phys = _physical_tube(index)
    rows = []
    for key, idx in phys.groupby(phys).groups.items():
        sp = index.loc[idx, "split"]
        present = [s for s in PRMI_SPLITS if (sp == s).any()]
        if len(present) < 2:
            continue
        sub, loc, tube = key.split("|", 2)
        rows.append({"tube_key": key, "subset": sub, "location": loc, "tube_num": tube,
                     "splits": "+".join(present),
                     "n_train": int((sp == "train").sum()), "n_val": int((sp == "val").sum()),
                     "n_test": int((sp == "test").sum()), "with_test": "test" in present})
    return pd.DataFrame(rows, columns=["tube_key", "subset", "location", "tube_num", "splits",
                                       "n_train", "n_val", "n_test", "with_test"])


def prmi_read(z, row, mask: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """``(rgb uint8 HxWx3, mask bool HxW | None)`` for one index row.

    The mask is None when ``mask=False`` or the image has none (noMask).  A
    mask whose size differs from its image is resized (nearest) with a warning.
    """
    z = _as_reader(z)
    if isinstance(row, str):
        im_m, mk_m = row, ""
    else:
        im_m, mk_m = _norm_str(row.get("image_member")), _norm_str(row.get("mask_member"))
    if not im_m:
        raise KeyError("no image member for %s" % (
            row if isinstance(row, str) else row.get("prmi_id", "?"),))
    rgb = decode_rgb(z.read(im_m))
    m = None
    if mask and mk_m:
        m = _mask_from_decoded(decode_unchanged(z.read(mk_m)))
        if m.shape != rgb.shape[:2]:
            warnings.warn("mask %s is %s, image %s: resized" % (mk_m, m.shape, rgb.shape[:2]))
            m = cv2.resize(m.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
    return rgb, m


# --------------------------------------------------------------------------
# widths and the scale table
# --------------------------------------------------------------------------
def _mask_samples(mask: np.ndarray, step_px: float = 15.0, max_samples: int = 200,
                  seed: int = 0, radius: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Centreline points ``(x, y)`` and unit normals from a mask's skeleton.

    Roughly one point per ``step_px`` of skeleton; the direction is the main
    axis of the skeleton pixels within ``radius``, and points whose
    neighbourhood is not line-like (junctions, blobs) are skipped.  Feeds
    ``ginger._fwhm_widths`` for the annotation-free width cross-check.
    """
    from scipy.spatial import cKDTree
    from skimage.morphology import skeletonize
    m = np.asarray(mask) > 0
    empty = (np.zeros((0, 2)), np.zeros((0, 2)))
    if not m.any():
        return empty
    ys, xs = np.nonzero(skeletonize(m))
    n = len(xs)
    if n < 3:
        return empty
    allp = np.column_stack([xs, ys]).astype(np.float64)
    k = int(min(max_samples, max(1, round(n / float(step_px)))))
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(n, k, replace=False)) if k < n else np.arange(n)
    tree = cKDTree(allp)
    pts, nrm = [], []
    for i in sel:
        nb = tree.query_ball_point(allp[i], radius)
        if len(nb) < 3:
            continue
        q = allp[nb] - allp[nb].mean(axis=0)
        ev, evec = np.linalg.eigh(q.T @ q)
        if ev[-1] <= 0 or ev[0] > 0.25 * ev[-1]:
            continue
        d = evec[:, -1]
        pts.append(allp[i])
        nrm.append([-d[1], d[0]])
    if not pts:
        return empty
    return np.asarray(pts), np.asarray(nrm)


def mask_width_row(mask: np.ndarray, rgb: Optional[np.ndarray] = None, seed: int = 0,
                   fwhm_half_len_px: int = 40) -> Dict[str, float]:
    """Width summary of one mask: n_skel, w2e_p10/p50/p90, mask_frac, h00..h10,
    and (with ``rgb``) fwhm_p50 / fwhm_n from the image along the skeleton."""
    from mrsim.traits import WIDTH_EDGES, skeleton_widths_px
    m = np.asarray(mask) > 0
    w = skeleton_widths_px(m)
    out: Dict[str, float] = {"n_skel": int(w.size), "mask_frac": float(m.mean())}
    q = np.percentile(w, [10, 50, 90]) if w.size else [float("nan")] * 3
    out.update({"w2e_p10": float(q[0]), "w2e_p50": float(q[1]), "w2e_p90": float(q[2])})
    for c, v in zip(hist_cols(WIDTH_EDGES), width_hist(w, WIDTH_EDGES)):
        out[c] = int(v)
    if rgb is not None:
        from mrstyle.ginger import _fwhm_widths
        pts, nrm = _mask_samples(m, seed=seed)
        fw = _fwhm_widths(rgb, pts, nrm, half_len_px=fwhm_half_len_px)
        fw = fw[np.isfinite(fw)]
        out["fwhm_p50"] = float(np.median(fw)) if fw.size else float("nan")
        out["fwhm_n"] = int(fw.size)
    return out


def prmi_width_sample(index, n_per_subset: int = 300, seed: int = 0,
                      subsets: Optional[Sequence[str]] = None):
    """The rows :func:`prmi_width_table` measures: up to ``n_per_subset``
    tube-balanced POSITIVE TRAIN masks per subset (no noMask, no shared tube)."""
    import pandas as pd
    d = index[(index.split == "train") & index.has_mask.astype(bool)
              & index.has_image.astype(bool) & (index.has_root.astype(int) == 1)
              & ~index.nomask.astype(bool) & ~index.shared_with_test.astype(bool)]
    if subsets is not None:
        d = d[d.subset.isin(list(subsets))]
    parts = [balanced_sample(g, "tube_uid", n_per_subset, seed)
             for _, g in d.groupby("subset", sort=True)]
    return pd.concat(parts) if parts else d.iloc[:0]


def prmi_width_table(z, index, n_per_subset: int = 300, seed: int = 0,
                     subsets: Optional[Sequence[str]] = None, fwhm: bool = True,
                     workers: Optional[int] = None, verbose: bool = True):
    """Per-image mask widths for the scale table and the simulator's width target.

    ONE pass over up to ``n_per_subset`` tube-balanced positive train masks per
    subset (300 settles the median to a few %; both notebooks use this table,
    so they cannot resample a subset differently).  Columns: prmi_id, subset,
    tube_uid, n_skel, w2e_p10/p50/p90, mask_frac, h00..h10 (WIDTH_EDGES counts)
    and, with ``fwhm``, fwhm_p50/fwhm_n -- the annotation-free cross-check: the
    FWHM of the IMAGE across the mask skeleton, which does not care whether the
    mask was drawn as WinRHIZO boxes or superpixels.
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    z = _as_reader(z)
    sample = prmi_width_sample(index, n_per_subset, seed, subsets)

    def one(row):
        rgb, m = prmi_read(z, row, mask=True)
        r = {"prmi_id": row["prmi_id"], "subset": row["subset"], "tube_uid": row["tube_uid"]}
        r.update(mask_width_row(m, rgb if fwhm else None, seed=seed))
        return r

    rows = [r for _, r in sample.iterrows()]
    workers = workers or min(8, max(1, os.cpu_count() or 2))
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(one, rows), 1):
            out.append(r)
            if verbose and (k % 500 == 0 or k == len(rows)):
                print("  PRMI widths %d/%d" % (k, len(rows)))
    df = pd.DataFrame(out)
    if verbose and len(df):
        print(df.groupby("subset").agg(n=("prmi_id", "size"), tubes=("tube_uid", "nunique"),
                                       w2e_p50=("w2e_p50", "median")).to_string())
    return df


def parse_overrides(spec) -> Dict[str, float]:
    """``'Papaya_736x552_DPI150:0.7, chicory:1'`` or a dict -> ``{subset: f}``."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {str(k).strip(): float(v) for k, v in spec.items()}
    out = {}
    for part in str(spec).split(","):
        if part.strip():
            k, v = part.rsplit(":", 1)
            out[k.strip()] = float(v)
    return out


SCALE_COLS = ["source", "subset", "n_img", "n_tubes", "w2e_med", "f", "f_lo", "f_hi",
              "fwhm_f", "px_per_mm_est", "method", "clamped", "f_raw", "fwhm_med",
              "flag_fwhm"]


def _by_tube(values: np.ndarray, tubes: np.ndarray):
    """Per-image values grouped by tube (the bootstrap resamples whole tubes)."""
    by: Dict[str, List[float]] = {}
    for v, t in zip(values, tubes):
        by.setdefault(t, []).append(v)
    return [np.asarray(v) for v in by.values()]


def _boot_median(groups, rng) -> float:
    pick = rng.integers(0, len(groups), len(groups))
    return float(np.median(np.concatenate([groups[i] for i in pick])))


def scale_table(widths, ref: str = PRMI_REF_SUBSET, clamp: Tuple[float, float] = SCALE_CLAMP,
                n_boot: int = 200, overrides=None, seed: int = 0,
                ref_px_per_mm: float = REF_PX_PER_MM, stat_col: str = "w2e_p50",
                fwhm_col: str = "fwhm_p50", fwhm_flag_log2: float = 0.3,
                chicory_px_per_mm: Optional[float] = CHICORY_PX_PER_MM,
                chicory_widths=None, index=None):
    """One row per PRMI subset plus a chicory row: the resampling factor ``f``.

    ``f`` resizes a source image to the reference scale (new size = old x f):

    * ``ref`` (Switchgrass, 300 dpi = 11.81 px/mm, the synthetic scale) is
      ``native``, f = 1;
    * a camera subset is ``width_match``: f = w_ref / w_subset, with w = the
      median over images of each image's median mask width (w2e) -- it assumes
      similar true root widths and annotation styles, which is why it is
      clamped, cross-checked and overridable;
    * ``overrides`` (dict or 'subset:f,...') replace any row's f
      (``override``);
    * chicory is ``fixed`` at ``ref_px_per_mm / chicory_px_per_mm`` (the
      geometry of Smith et al.); ``chicory_widths`` (Zenodo masks) only fill
      its check column ``f_raw``;
    * ``f`` is FINAL -- matched, overridden, then clamped to ``clamp`` (flag
      ``clamped``).  CUT and 04 read only ``f``.

    ``f_lo``/``f_hi``: 95% interval of the unclamped match from a bootstrap
    over TUBES of both the subset and the reference (frames of one tube are not
    independent).  ``fwhm_f`` repeats the match with the image FWHM across the
    mask skeleton (``fwhm_col``); ``flag_fwhm`` marks |log2(f / fwhm_f)| >
    ``fwhm_flag_log2`` for review.  ``px_per_mm_est = ref_px_per_mm / f``.
    """
    import pandas as pd
    ov = parse_overrides(overrides)
    w = widths.copy()
    if "tube_uid" not in w.columns:
        if index is None:
            raise ValueError("widths has no tube_uid column; pass index= to merge it")
        w = w.merge(index[["prmi_id", "tube_uid"]], on="prmi_id", how="left")
    w = w[np.isfinite(pd.to_numeric(w[stat_col], errors="coerce"))]
    if ref not in set(w.subset):
        raise ValueError("reference subset %r has no width rows" % ref)
    rng = np.random.default_rng(seed)
    has_fwhm = fwhm_col in w.columns

    def med(d, col):
        v = pd.to_numeric(d[col], errors="coerce").to_numpy(np.float64)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    wr = w[w.subset == ref]
    w_ref = med(wr, stat_col)
    g_ref = _by_tube(wr[stat_col].to_numpy(np.float64), wr.tube_uid.astype(str).to_numpy())
    fw_ref = med(wr, fwhm_col) if has_fwhm else float("nan")
    lo_c, hi_c = float(clamp[0]), float(clamp[1])

    def finish(row):
        f_pre = row["f"]
        if row["subset"] in ov:
            f_pre, row["method"] = ov[row["subset"]], "override"
        f = float(min(max(f_pre, lo_c), hi_c))
        row["clamped"] = bool(abs(f - f_pre) > 1e-12)
        row["f"] = f
        row["px_per_mm_est"] = ref_px_per_mm / f
        ff = row["fwhm_f"]
        row["flag_fwhm"] = bool(np.isfinite(ff) and ff > 0
                                and abs(math.log2(f / ff)) > fwhm_flag_log2)
        return row

    rows = []
    for sub in sorted(set(w.subset)):
        d = w[w.subset == sub]
        w_sub = med(d, stat_col)
        r = {"source": "prmi", "subset": sub, "n_img": int(len(d)),
             "n_tubes": int(d.tube_uid.nunique()), "w2e_med": w_sub,
             "fwhm_med": med(d, fwhm_col) if has_fwhm else float("nan")}
        r["fwhm_f"] = (fw_ref / r["fwhm_med"]) if (has_fwhm and r["fwhm_med"] > 0) \
            else float("nan")
        if sub == ref:
            r.update({"f": 1.0, "f_raw": 1.0, "f_lo": 1.0, "f_hi": 1.0, "method": "native"})
        else:
            f_raw = w_ref / w_sub if w_sub > 0 else float("nan")
            g_sub = _by_tube(d[stat_col].to_numpy(np.float64),
                             d.tube_uid.astype(str).to_numpy())
            boot = np.asarray([_boot_median(g_ref, rng) / _boot_median(g_sub, rng)
                               for _ in range(int(n_boot))], np.float64)
            boot = boot[np.isfinite(boot)]
            r.update({"f": f_raw, "f_raw": f_raw, "method": "width_match",
                      "f_lo": float(np.percentile(boot, 2.5)) if boot.size else float("nan"),
                      "f_hi": float(np.percentile(boot, 97.5)) if boot.size else float("nan")})
        rows.append(finish(r))

    if chicory_px_per_mm:
        r = {"source": "chicory", "subset": "chicory", "n_img": 0, "n_tubes": 0,
             "w2e_med": float("nan"), "fwhm_med": float("nan"), "fwhm_f": float("nan"),
             "f": ref_px_per_mm / float(chicory_px_per_mm), "f_raw": float("nan"),
             "f_lo": float("nan"), "f_hi": float("nan"), "method": "fixed"}
        if chicory_widths is not None and len(chicory_widths):
            cw = chicory_widths[np.isfinite(pd.to_numeric(chicory_widths[stat_col],
                                                          errors="coerce"))]
            r["n_img"] = int(len(cw))
            r["w2e_med"] = med(cw, stat_col)
            r["f_raw"] = w_ref / r["w2e_med"] if r["w2e_med"] > 0 else float("nan")
            if has_fwhm and fwhm_col in cw.columns:
                r["fwhm_med"] = med(cw, fwhm_col)
                r["fwhm_f"] = fw_ref / r["fwhm_med"] if r["fwhm_med"] > 0 else float("nan")
        rows.append(finish(r))
    return pd.DataFrame(rows)[SCALE_COLS]


def scale_factors(table) -> Dict[str, float]:
    """``{subset: f}`` from a scale table -- the only thing CUT and 04 read."""
    return {str(s): float(f) for s, f in zip(table.subset, table.f)}


def resize_mask(mask: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """Binary mask resize that keeps thin roots.

    Shrinking: area-average coverage > 0.25 (a 1-px root still covers ~0.5 of
    an output pixel at 0.5x, where the usual >= 0.5 rule drops half of it).
    Enlarging: bilinear > 0.5.  Same rule as 04's crop-then-resize.
    """
    m = (np.asarray(mask) > 0).astype(np.float32)
    w, h = int(size_wh[0]), int(size_wh[1])
    if (w, h) == (m.shape[1], m.shape[0]):
        return m > 0.5
    if w * h < m.shape[0] * m.shape[1]:
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) > 0.25
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5


def resample_rgb_mask(rgb: np.ndarray, mask: Optional[np.ndarray] = None, f: float = 1.0,
                      tol: float = 0.02) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resize an image (and its mask) by ``f`` to the reference scale.

    RGB: INTER_AREA when shrinking, INTER_CUBIC when enlarging (the 04 test
    images get the same treatment as the CUT pool, so styles and tests agree).
    Mask: :func:`resize_mask`.  ``|f - 1| <= tol`` returns the inputs unchanged.
    """
    f = float(f)
    if not (f > 0 and math.isfinite(f)):
        raise ValueError("bad scale factor %r" % (f,))
    if abs(f - 1.0) <= tol:
        return rgb, (None if mask is None else np.asarray(mask) > 0)
    h, w = rgb.shape[:2]
    size = (max(1, int(round(w * f))), max(1, int(round(h * f))))
    out = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    return out, (None if mask is None else resize_mask(mask, size))


# --------------------------------------------------------------------------
# chicory: names, root intensity, the grid set and the Zenodo masks
# --------------------------------------------------------------------------
_CHICORY_RE = re.compile(r"^(?P<yy>\d{2})_(?P<mm>\d{2})_(?P<dd>\d{2})_(?P<panel>\d+)"
                         r"(?P<side>[EWew])(?P<depth>\d+)(?P<window>[a-dA-D])_P(?P<photo>\d+)$")
_ANNOT_SUFFIXES = ("_photo", "_annotation", "_annotations", "_annot", "_mask", "_label",
                   "_labels", "_gt", "_seg")
_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def chicory_stem(name: str) -> str:
    """Basename without image extension or annotation/photo suffix."""
    s = str(name).strip().replace("\\", "/").split("/")[-1]
    root, ext = os.path.splitext(s)
    if ext.lower() in _IMG_EXT:
        s = root
    low = s.lower()
    for suf in _ANNOT_SUFFIXES:
        if low.endswith(suf):
            s = s[:-len(suf)]
            break
    return s


def parse_chicory_name(name: str, strict: bool = False) -> Optional[Dict[str, object]]:
    """``'16_08_22_10E2d_P8222710'`` -> date 2016-08-22, panel '10', side 'E',
    depth 2, window 'd', photo '8222710', panel_side '10E' (the split unit),
    window_uid '10E2d' (overlap flags).  Extensions and ``_photo``-style
    suffixes are stripped.  Unparseable: None, or ValueError with ``strict``.
    """
    s = chicory_stem(name)
    m = _CHICORY_RE.match(s)
    if not m:
        if strict:
            raise ValueError("not a chicory rhizotron name: %r" % (name,))
        return None
    side, win = m.group("side").upper(), m.group("window").lower()
    panel, depth = m.group("panel"), int(m.group("depth"))
    return {"stem": s, "date": "20%s-%s-%s" % (m.group("yy"), m.group("mm"), m.group("dd")),
            "panel": panel, "side": side, "depth": depth, "window": win,
            "photo": m.group("photo"), "panel_side": panel + side,
            "panel_depth": "%s%s%d" % (panel, side, depth),
            "window_uid": "%s%s%d%s" % (panel, side, depth, win)}


def read_ri_csv(path_or_buf):
    """``grid_counted_manual_ri.csv`` -> DataFrame[stem, ri].

    The file is headerless ``stem,ri`` (e.g. ``16_08_22_10E2d_P8222710,13.0``);
    a header row, a '.png' on the stem and extra columns are tolerated.
    Duplicate stems keep the first value (with a warning if they disagree).
    """
    import pandas as pd
    raw = pd.read_csv(path_or_buf, header=None, dtype=str, skip_blank_lines=True)
    raw = raw.iloc[:, :2]
    raw.columns = ["stem", "ri"]
    raw["stem"] = raw.stem.fillna("").map(chicory_stem)
    ri = pd.to_numeric(raw.ri, errors="coerce")
    if len(raw) and not np.isfinite(ri.iloc[0]) and parse_chicory_name(raw.stem.iloc[0]) is None:
        raw, ri = raw.iloc[1:], ri.iloc[1:]                # a header row
    df = pd.DataFrame({"stem": raw.stem.to_numpy(), "ri": ri.to_numpy(np.float64)})
    df = df[df.stem != ""]
    dup = df.stem.duplicated(keep=False)
    if dup.any():
        bad = df[dup].groupby("stem").ri.nunique()
        if (bad > 1).any():
            warnings.warn("RI csv: %d stems repeat with different values; first kept"
                          % int((bad > 1).sum()))
        df = df.drop_duplicates("stem", keep="first")
    return df.reset_index(drop=True)


def ri_to_length_mm(ri, area_m2: float):
    """Root length (mm) from root intensity: Newman's ``(pi/2) * RI * A``.

    RI is intersections per metre of grid line, A the image area in m^2, so
    (pi/2) RI A is metres of root; x1000 for mm.  RI 10 on 0.051 m^2 ~ 801 mm.
    """
    return (math.pi / 2.0) * np.asarray(ri, np.float64) * float(area_m2) * 1000.0


def chicory_area_m2(px_per_mm: float = CHICORY_PX_PER_MM,
                    image_wh: Tuple[int, int] = CHICORY_IMAGE_WH) -> float:
    return (image_wh[0] / float(px_per_mm)) * (image_wh[1] / float(px_per_mm)) / 1e6


def parse_panels(spec) -> List[str]:
    """``'10E, 11w'`` -> ['10E', '11W']."""
    if not spec:
        return []
    items = spec if isinstance(spec, (list, tuple, set)) else str(spec).split(",")
    return [str(p).strip().upper() for p in items if str(p).strip()]


def _names_of(z) -> List[str]:
    if isinstance(z, str):
        with zipfile.ZipFile(z) as zf:
            return zf.namelist()
    if hasattr(z, "namelist"):
        return list(z.namelist())
    return [str(n) for n in z]


GRID_COLS = ["stem", "member", "date", "panel_side", "depth", "window_uid", "ri", "L_RI_mm",
             "role", "cut_allowed", "panel", "side", "window", "photo"]


def grid_index(z, ri=None, px_per_mm: float = CHICORY_PX_PER_MM,
               image_wh: Tuple[int, int] = CHICORY_IMAGE_WH,
               cut_panels=CHICORY_CUT_PANELS, verbose: bool = True):
    """grid_counted images (from the zip's central directory only) + RI + roles.

    ``ri`` is the :func:`read_ri_csv` table or None (ri / L_RI_mm NaN).
    ``px_per_mm`` should be the chicory row of the scale table
    (``px_per_mm_est``); the image area for the RI length uses it.  ``role`` is
    the real_splits rule: CUT panel-sides are 'train' (style targets only,
    ``cut_allowed``), every other panel-side 'test' -- the primary count test.
    """
    import pandas as pd
    names = [n for n in _names_of(z) if n.lower().endswith(_IMG_EXT)]
    cut = set(parse_panels(cut_panels))
    area = chicory_area_m2(px_per_mm, image_wh)
    rows, bad = [], []
    for n in names:
        p = parse_chicory_name(n)
        if p is None:
            bad.append(n)
            continue
        ps = p["panel_side"]
        rows.append({"stem": p["stem"], "member": n, "date": p["date"], "panel_side": ps,
                     "depth": p["depth"], "window_uid": p["window_uid"],
                     "role": "train" if ps in cut else "test", "cut_allowed": ps in cut,
                     "panel": p["panel"], "side": p["side"], "window": p["window"],
                     "photo": p["photo"]})
    df = pd.DataFrame(rows, columns=[c for c in GRID_COLS if c not in ("ri", "L_RI_mm")])
    if ri is not None:
        df = df.merge(ri[["stem", "ri"]], on="stem", how="left")
    else:
        df["ri"] = np.nan
    df["L_RI_mm"] = ri_to_length_mm(df["ri"].to_numpy(np.float64), area)
    df = df[GRID_COLS].sort_values("stem", kind="mergesort").reset_index(drop=True)
    if verbose:
        n_ri = int(np.isfinite(df.ri).sum())
        print("grid index: %d images (%d unparseable names), %d with RI%s; roles %s"
              % (len(df), len(bad), n_ri,
                 "" if ri is None else " (%d RI rows unmatched)"
                 % int((~ri.stem.isin(df.stem)).sum()),
                 df.role.value_counts().to_dict()))
    return df


def _zip_folder(name: str, default: str) -> str:
    for p in name.replace("\\", "/").split("/")[:-1]:
        if p.lower() in ("train", "val", "valid", "validation", "test"):
            return {"valid": "val", "validation": "val"}.get(p.lower(), p.lower())
    return default


def _member_kind(name: str) -> str:
    parts = [p.lower() for p in name.replace("\\", "/").split("/")]
    dirs, base = parts[:-1], os.path.splitext(parts[-1])[0]
    if any("annot" in d or "label" in d or "mask" in d for d in dirs):
        return "annotation"
    if any("photo" in d or "image" in d for d in dirs):
        return "photo"
    for suf in _ANNOT_SUFFIXES[1:]:
        if base.endswith(suf):
            return "annotation"
    return "photo"


ZENODO_COLS = ["stem", "folder", "zip_path", "photo_member", "annot_member", "has_annot",
               "date", "panel_side", "depth", "window_uid"]


def chicory_index(zips, verbose: bool = True):
    """Zenodo 3527713 chicory photos paired with their pixel annotations.

    ``zips``: one path, a list of paths, or ``{folder: path}``.  Members are
    classified as photo or annotation by their folder (``photos/``,
    ``annotations/``) or suffix, and paired on :func:`chicory_stem` within one
    split folder (train/val/test, from the member path or the dict key, else
    the zip's file name).  The annotation file format is unverified, hence
    :func:`annotation_to_mask` and the notebook's plausibility checks.
    """
    import pandas as pd
    if isinstance(zips, str):
        zips = [zips]
    items = list(zips.items()) if isinstance(zips, dict) else \
        [(os.path.splitext(os.path.basename(p))[0].lower(), p) for p in zips]
    photos: Dict[Tuple[str, str], Tuple[str, str]] = {}
    annots: Dict[Tuple[str, str], str] = {}
    for key, path in items:
        for n in _names_of(path):
            if not n.lower().endswith(_IMG_EXT):
                continue
            folder = _zip_folder(n, key)
            k = (folder, chicory_stem(n))
            if _member_kind(n) == "annotation":
                annots.setdefault(k, n)
            else:
                photos.setdefault(k, (path, n))
    rows = []
    for (folder, stem), (path, n) in sorted(photos.items()):
        p = parse_chicory_name(stem) or {}
        a = annots.get((folder, stem), "")
        rows.append({"stem": stem, "folder": folder, "zip_path": path, "photo_member": n,
                     "annot_member": a, "has_annot": bool(a), "date": p.get("date", ""),
                     "panel_side": p.get("panel_side", ""), "depth": p.get("depth", -1),
                     "window_uid": p.get("window_uid", "")})
    df = pd.DataFrame(rows, columns=ZENODO_COLS)
    if verbose:
        print("chicory (Zenodo): %d photos, %d with annotations, %d annotations unpaired; "
              "per folder %s" % (len(df), int(df.has_annot.sum()),
                                 len(set(annots) - set(photos)),
                                 df.folder.value_counts().to_dict()))
    return df


def annotation_to_mask(img: np.ndarray, mode: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """``(root bool, valid bool)`` from an annotation image in RGB(A) order.

    The Zenodo annotation format is not verified, so three conventions are
    decoded (``mode='auto'`` picks one):

    * grey / binary (also RGB(A) with R=G=B and opaque alpha): root = > half max;
    * RootPainter RGBA: red = root, green = soil, TRANSPARENT = unlabelled
      (valid False) -- chosen when any green pixel exists, or ``'rootpainter'``;
    * red on transparent (``'red'``): red = root, everything else is soil.

    Use :func:`decode_annotation` on raw bytes (it converts OpenCV's BGR).
    """
    a = np.asarray(img)
    if mode not in ("auto", "grey", "rootpainter", "red"):
        raise ValueError("mode must be auto|grey|rootpainter|red")
    full = np.ones(a.shape[:2], bool)

    def thr(x):
        if x.dtype == bool:
            return x
        mx = float(x.max()) if x.size else 0.0
        if mx <= 1.0:
            return x > 0
        return x > (127 if x.dtype == np.uint8 else mx / 2.0)

    if a.ndim == 2:
        return thr(a), full
    rgb = a[:, :, :3].astype(np.int32)
    alpha = (a[:, :, 3] > 0) if a.shape[2] >= 4 else full
    grey_like = bool(np.all(rgb[:, :, 0] == rgb[:, :, 1]) and np.all(rgb[:, :, 1] == rgb[:, :, 2]))
    if mode == "grey" or (mode == "auto" and grey_like and alpha.all()):
        return thr(a[:, :, 0]), full
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red = alpha & (r >= 128) & (r - g >= 64) & (r - b >= 64)
    green = alpha & (g >= 128) & (g - r >= 64) & (g - b >= 64)
    if mode == "rootpainter" or (mode == "auto" and green.any()):
        return red, red | green
    if mode == "red" or red.any():
        return red, full
    # strokes of another colour on transparency: the opaque, bright ones
    lum = rgb.max(axis=2)
    root = alpha & (lum > 127)
    return (root if root.any() else alpha.copy()), full


def decode_annotation(data: bytes, mode: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    return annotation_to_mask(decode_unchanged(data), mode)


# --------------------------------------------------------------------------
# frozen real splits
# --------------------------------------------------------------------------
def ginger_scan_key(x) -> str:
    """Scan key as the split tables spell it: '000', 0 and '0' are all '0'."""
    s = _norm_str(x)
    return str(int(s)) if s.isdigit() else s


def ginger_unit(dataset, scan) -> str:
    """``'B_0'``: (dataset, scan) is the Ginger split unit (keys repeat across
    datasets; v2 cell 22 once merged on the key alone and flipped tiles)."""
    return "%s_%s" % (_norm_str(dataset), ginger_scan_key(scan))


def ginger_tile_unit(tile_id: str) -> str:
    """Unit of an ``export_real_tiles`` id ``<ds>_<sub>_<key>_<role>_y..._x...``."""
    p = str(tile_id).split("_")
    if len(p) < 3:
        raise ValueError("not a Ginger tile id: %r" % (tile_id,))
    return ginger_unit(p[0], p[2])


def _as_ginger_unit(u) -> str:
    """A unit ('B_000' -> 'B_0') or a tile id ('B_st_001_single_y0000_x0000' -> 'B_1')."""
    s = _norm_str(u)
    p = s.split("_")
    if len(p) == 2 and p[0] and p[1]:
        return ginger_unit(p[0], p[1])
    if len(p) >= 3:
        return ginger_tile_unit(s)
    raise ValueError("not a Ginger unit or tile id: %r" % (u,))


def _tile_unit_map(tiles) -> Dict[str, str]:
    if tiles is None:
        return {}
    cols = set(tiles.columns)
    if "unit_uid" in cols:
        units = tiles.unit_uid.astype(str)
    elif {"dataset", "key"} <= cols:
        units = [ginger_unit(d, k) for d, k in zip(tiles.dataset, tiles.key)]
    elif {"dataset", "scan"} <= cols:
        units = []
        for d, s in zip(tiles.dataset.astype(str), tiles.scan.astype(str)):
            s = s[len(d) + 1:] if s.startswith(d + "_") else s     # real.csv: 'B_0'
            units.append(ginger_unit(d, s))
    else:
        raise ValueError("tiles needs unit_uid, (dataset, key) or (dataset, scan)")
    return dict(zip(tiles.id.astype(str), units))


def _id_sets(x) -> List[set]:
    """A list of ids, or a list of such lists (several split.json files)."""
    if x is None:
        return []
    x = list(x)
    if x and not isinstance(x[0], str) and hasattr(x[0], "__iter__"):
        return [set(map(str, v)) for v in x]
    return [set(map(str, x))]


def ginger_roles(scan_split, real_val=None, real_test=None, tiles=None,
                 all_units: Optional[Iterable[str]] = None):
    """Frozen Ginger roles per scan unit (``'B_0'``), as INPUT tables in, rows out.

    * test = ``split == 'holdout'`` scans of v2 ``multisoil/scan_split.csv``
      (columns scan, dataset, split), merged on (dataset, scan);
    * val = the scans of v1 ``split.json`` ``real_val`` (117 tiles).  Pass one id
      list or several (one per v1 run): they must all agree;
    * train = everything else, as the contract says: every other scan of
      scan_split AND every unit of ``all_units`` absent from it.  scan_split
      lists only scans with a kept v2 tile, so the missing ones are mostly
      root-free standing-crop scans -- exactly the scarce root-free tiles CUT
      needs -- and by construction they were never v1 val or test.  A scan
      whose v2 tiles were all dropped as exact copies of another scan's is
      also here; its copies are caught by the tile_hash / DINO guards, as
      any copy is.

    ``tiles`` maps tile ids to scans (real.csv, tiles.csv); without it ids are
    parsed.  ``real_test`` (v1 real_test ids) is checked to fall inside the
    test scans.  Raises on any contradiction instead of guessing.
    """
    import pandas as pd
    sp = scan_split.copy()
    sp["unit_uid"] = [ginger_unit(d, s) for d, s in zip(sp.dataset, sp.scan)]
    sp["split"] = sp.split.astype(str).str.strip().str.lower()
    conflict = sp.groupby("unit_uid").split.nunique()
    if (conflict > 1).any():
        raise ValueError("scan_split lists %s with two different splits"
                         % sorted(conflict[conflict > 1].index)[:5])
    sp = sp.drop_duplicates("unit_uid")
    test_units = set(sp.unit_uid[sp.split.isin(("holdout", "test"))])
    umap = _tile_unit_map(tiles)

    def units_of(ids):
        return {umap.get(i) or ginger_tile_unit(i) for i in ids}

    vals = _id_sets(real_val)
    for k, v in enumerate(vals[1:], 1):
        if v != vals[0]:
            raise ValueError("v1 split.json real_val #%d differs from #0 (%d vs %d tiles, "
                             "%d in common)" % (k, len(v), len(vals[0]), len(v & vals[0])))
    val_units = units_of(vals[0]) if vals else set()
    clash = val_units & test_units
    if clash:
        raise ValueError("Ginger val scans are holdout scans in scan_split: %s" % sorted(clash))
    for ts in _id_sets(real_test):
        out = units_of(ts) - test_units
        if out:
            raise ValueError("v1 real_test tiles from non-holdout scans: %s" % sorted(out)[:5])

    rows = []
    sp_units = set(sp.unit_uid)
    # normalised, because a unit not in scan_split is now TRAIN and CUT-allowed:
    # 'A_000' left as spelled would be a second, trainable copy of holdout 'A_0'
    extra = {_as_ginger_unit(u) for u in (all_units or ())}
    for u in sorted(sp_units | val_units | extra):
        if u in test_units:
            role, why = "test", "v2 scan_split holdout"
        elif u in val_units:
            role, why = "val", "v1 split.json real_val"
        elif u in sp_units:
            role, why = "train", "v2 scan_split train"
        else:
            role, why = "train", "not in scan_split.csv (no kept v2 tile; never v1 val/test)"
        rows.append({"source": "ginger", "unit_uid": u, "role": role,
                     "cut_allowed": role == "train", "reason": why})
    return pd.DataFrame(rows, columns=SPLIT_COLS)


def prmi_roles(index, use_nomask: bool = False):
    """One row per PRMI ``tube_uid``: the official split, made leak-free.

    * a tube in exactly one split keeps it;
    * a PHYSICAL tube in test and another split is 'excluded': its train/val
      images are never used, its test images stay in the official test (test
      membership is the index's ``split``; roles govern train/val/CUT use --
      select PRMI IMAGES with :func:`prmi_image_roles`, never by tube role);
    * a tube in train and val only becomes 'val' (its train images unused);
    * noMask tubes are always 'excluded' in the frozen table.  Those whose
      physical tube is never in val or test carry the fixed reason
      :data:`PRMI_NOMASK_OPTIN`; the rest (9 test tubes) can never be used.

    The frozen table does NOT depend on ``PRMI_USE_NOMASK``: real_splits.csv is
    'first writer wins', so a switch baked into it would hand the other
    notebook the first one's choice (CUT would silently get noMask style units,
    or 04 lose its negatives).  Each notebook applies its own switch with
    :func:`prmi_apply_nomask`; ``use_nomask=True`` here returns that applied
    VIEW, which must never be written as real_splits.csv.
    """
    import pandas as pd
    phys = _physical_tube(index)
    phys_splits = {k: set(index.loc[v, "split"]) for k, v in phys.groupby(phys).groups.items()}
    rows = []
    for tu, d in index.groupby("tube_uid", sort=True):
        splits = set(d.split)
        ps = phys_splits.get(phys.loc[d.index[0]], splits)
        nomask = bool(d.nomask.astype(bool).any())
        if nomask:
            if ps & {"val", "test"}:
                role, why = "excluded", "noMask tube also in %s" % "+".join(
                    s for s in PRMI_SPLITS if s in ps and s != "train")
            else:
                role, why = "excluded", PRMI_NOMASK_OPTIN
        elif "test" in ps and len(ps) > 1:
            role, why = "excluded", "tube shared %s: test images stay in test" % "+".join(
                s for s in PRMI_SPLITS if s in ps)
        elif len(splits) == 1:
            role, why = next(iter(splits)), "official split"
        else:
            role, why = "val", "tube shared train+val: train images unused"
        rows.append({"source": "prmi", "unit_uid": tu, "role": role,
                     "cut_allowed": role == "train", "reason": why})
    df = pd.DataFrame(rows, columns=SPLIT_COLS)
    return prmi_apply_nomask(df, True) if use_nomask else df


def prmi_apply_nomask(splits, use_nomask: bool):
    """A copy of ``splits`` with one notebook's ``PRMI_USE_NOMASK`` applied.

    With ``use_nomask`` the PRMI rows whose reason is :data:`PRMI_NOMASK_OPTIN`
    (noMask tubes never in val or test) become role 'train', cut_allowed True:
    usable as has_root-only negatives (04) or root-free style tiles (CUT).
    Without it the copy is unchanged.  Never write the result back.
    """
    df = splits.copy()
    if use_nomask:
        sel = (df.source == "prmi") & (df.reason == PRMI_NOMASK_OPTIN)
        df.loc[sel, "role"] = "train"
        df.loc[sel, "cut_allowed"] = True
    return df


def prmi_image_roles(index, splits, use_nomask: bool = False):
    """Per-IMAGE PRMI role, aligned with ``index`` -- the one way to select images.

    * an image of the official test split is 'test', whatever its tube's role:
      a tube shared with test is 'excluded' as a unit, but the official test
      keeps its test images (selecting test by tube role would silently drop
      them while ``prmi_check_totals``, which counts the index, still passes);
    * any other image takes its tube's role when that equals its own split
      ('train' images of a 'train' tube, 'val' images of a 'val' tube), else
      'excluded' (the train images of a train+val tube, everything of an
      excluded tube, noMask unless ``use_nomask``).

    ``splits`` is real_splits (all sources or PRMI only).  A non-test image
    whose tube is missing from it raises: the index and the table disagree.
    """
    import pandas as pd
    rm = role_map(prmi_apply_nomask(splits, use_nomask), "prmi")
    split = index.split.astype(str)
    tube_role = index.tube_uid.astype(str).map(rm)
    miss = (split != "test") & tube_role.isna()
    if miss.any():
        raise ValueError("%d PRMI images have tubes missing from real_splits, e.g. %s"
                         % (int(miss.sum()), index.tube_uid[miss].iloc[0]))
    out = pd.Series(np.where(split == "test", "test",
                             np.where(tube_role == split, split, "excluded")),
                    index=index.index, name="role")
    return out


def chicory_grid_roles(grid, cut_panels=CHICORY_CUT_PANELS):
    """Per panel-side: CUT panels 'train' + cut_allowed (style targets only), the
    rest 'test' (the count test).  ``grid``: a grid index or panel-side list."""
    import pandas as pd
    cut = set(parse_panels(cut_panels))
    sides = sorted(set(grid.panel_side) if hasattr(grid, "panel_side") else set(grid))
    rows = [{"source": "chicory_grid", "unit_uid": s,
             "role": "train" if s in cut else "test", "cut_allowed": s in cut,
             "reason": "CHICORY_CUT_PANELS: CUT style target only" if s in cut
             else "count test (never a CUT target)"} for s in sides]
    return pd.DataFrame(rows, columns=SPLIT_COLS)


def zenodo_roles(zen):
    """Per Zenodo photo: role = its zip folder, never a CUT target.  Test-only by
    the user's decision; training on its 'train' folder is an opt-in ablation."""
    import pandas as pd
    rows = []
    for stem, folder in sorted(set(zip(zen.stem.astype(str), zen.folder.astype(str)))):
        role = folder if folder in ROLES else "test"
        rows.append({"source": "chicory_zenodo", "unit_uid": stem, "role": role,
                     "cut_allowed": False,
                     "reason": "zenodo %s folder; test-only by default" % folder})
    return pd.DataFrame(rows, columns=SPLIT_COLS)


def build_real_splits(scan_split=None, real_val=None, real_test=None, ginger_tiles=None,
                      ginger_units=None, prmi_index=None, use_nomask: bool = False,
                      grid=None, cut_panels=CHICORY_CUT_PANELS, zenodo=None,
                      out_path: Optional[str] = None, verbose: bool = True):
    """``real_splits.csv``: every real unit's frozen role, from the INPUT tables.

    Any source may be omitted (its rows are then absent).  Columns: source,
    unit_uid, role (train|val|test|excluded), cut_allowed, reason.  Written
    atomically when ``out_path`` is given.  Both notebooks read this file and
    never re-draw a split.

    The table is independent of every notebook switch, so whichever notebook
    writes it first, the other reads the same file.  ``use_nomask=True`` is
    therefore refused: apply ``PRMI_USE_NOMASK`` after reading, with
    :func:`prmi_apply_nomask` (units) or :func:`prmi_image_roles` (images).
    PRMI images must be selected with :func:`prmi_image_roles`: the official
    test keeps the test images of tubes that are 'excluded' as units.
    """
    import pandas as pd
    if use_nomask:
        raise ValueError("real_splits.csv must not depend on PRMI_USE_NOMASK: build it with "
                         "use_nomask=False and apply the switch with prmi_apply_nomask() / "
                         "prmi_image_roles(..., use_nomask=True) after reading it")
    parts = []
    if scan_split is not None:
        parts.append(ginger_roles(scan_split, real_val, real_test, ginger_tiles, ginger_units))
    if prmi_index is not None:
        parts.append(prmi_roles(prmi_index))
    if grid is not None:
        parts.append(chicory_grid_roles(grid, cut_panels))
    if zenodo is not None:
        parts.append(zenodo_roles(zenodo))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SPLIT_COLS)
    validate_real_splits(df)
    if out_path:
        write_csv_atomic(df, out_path)
    if verbose and len(df):
        print("real splits:\n" + pd.crosstab(df.source, df.role).to_string())
    return df


def validate_real_splits(df) -> None:
    """Columns present, roles known, (source, unit_uid) unique, CUT never on
    val/test/excluded units nor on Zenodo."""
    miss = [c for c in SPLIT_COLS if c not in df.columns]
    if miss:
        raise ValueError("real_splits is missing columns %s" % miss)
    bad = set(df.role) - set(ROLES)
    if bad:
        raise ValueError("unknown roles %s" % sorted(bad))
    dup = df.duplicated(["source", "unit_uid"])
    if dup.any():
        raise ValueError("duplicate units: %s" % df[dup].unit_uid.head(5).tolist())
    cut = df.cut_allowed.astype(str).str.lower().isin(("true", "1"))
    if (cut & (df.role != "train")).any() or (cut & (df.source == "chicory_zenodo")).any():
        raise ValueError("cut_allowed on a non-train or Zenodo unit")


def read_real_splits(path: str):
    import pandas as pd
    df = pd.read_csv(path, dtype={"source": str, "unit_uid": str, "role": str, "reason": str},
                     keep_default_na=False)
    df["cut_allowed"] = df.cut_allowed.astype(str).str.lower().isin(("true", "1"))
    return df


def role_map(splits, source: str) -> Dict[str, str]:
    """``{unit_uid: role}`` for one source of a real_splits table.

    Unit roles only.  For PRMI images use :func:`prmi_image_roles`: a tube
    shared with test is 'excluded' here, yet its test images belong to the
    official test.
    """
    d = splits[splits.source == source]
    return dict(zip(d.unit_uid.astype(str), d.role.astype(str)))


# --------------------------------------------------------------------------
# the synthetic hand-off: v3 styled_v3.zip or v2 multisoil/seg_train.zip
# --------------------------------------------------------------------------
def strip_set_prefix(sid: str) -> str:
    """``'v3_s012_0034'`` -> ``'s012_0034'`` (unprefixed ids pass through)."""
    m = re.match(r"^v\d+[a-z]?_(s\d{3,}_\d{4})$", str(sid))
    return m.group(1) if m else str(sid)


def syn_system(sid: str) -> int:
    s = strip_set_prefix(sid)
    return int(s[1:s.index("_")])


def _to_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "1.0", "yes"))


def _find_member(names: Sequence[str], base: str) -> str:
    if base in names:
        return base
    hits = sorted((n for n in names if n.endswith("/" + base)), key=lambda n: n.count("/"))
    return hits[0] if hits else ""


def open_synthetic(paths, tags: Optional[Sequence[str]] = None):
    """One frames table for one or more synthetic hand-off zips.

    * v3 ``styled_v3.zip``: ``frames_v3.csv`` (+ ``meta.json``), with rgb/,
      mask/, diam/, alpha/;
    * v2 ``multisoil/seg_train.zip``: ``train.csv`` with rgb/ and mask/; every
      row passed v2's audit, so ``train_ok`` is True and there is no diam/
      (widths then come from the mask).

    Adds: ``raw_id``, ``syn_set`` (tag), ``zip_path``, ``system`` (int),
    ``system_uid`` (unique across sets), ``train_ok`` (bool) and
    ``{rgb,mask,diam,alpha}_member`` ('' when absent).  With more than one set
    the ids get a ``<tag>_`` prefix (v2 and v3 both start at s000_0000).  Ids
    must match ``sNNN_MMMM`` and be unique per set.  ``df.attrs['meta']`` holds
    each set's meta.json.
    """
    import pandas as pd
    paths = [paths] if isinstance(paths, str) else list(paths)
    if not paths:
        raise ValueError("no synthetic zip given")
    sets = []
    for p in paths:
        with zipfile.ZipFile(p) as zf:
            names = zf.namelist()
            fv3, fv2 = _find_member(names, "frames_v3.csv"), _find_member(names, "train.csv")
            meta = {}
            if fv3:
                kind, csv_m = "v3", fv3
                mm = _find_member(names, "meta.json")
                if mm:
                    meta = json.loads(zf.read(mm).decode("utf-8"))
            elif fv2:
                kind, csv_m = "v2", fv2
            else:
                raise ValueError("%s has neither frames_v3.csv nor train.csv" % p)
            with zf.open(csv_m) as fh:
                df = pd.read_csv(fh, dtype={"id": str})
        sets.append((p, kind, set(names), meta, df))
    if tags is None:
        tags = [k for _, k, _, _, _ in sets]
        if len(set(tags)) != len(tags):
            raise ValueError("two synthetic sets of the same version %s: pass tags=" % tags)
    elif len(tags) != len(sets) or len(set(tags)) != len(tags):
        raise ValueError("tags must be unique, one per path")
    prefix = len(sets) > 1
    out, metas = [], {}
    for (p, kind, names, meta, df), tag in zip(sets, tags):
        df = df.copy()
        df["id"] = df["id"].astype(str).str.strip()
        bad = df.id[~df.id.str.match(SYN_ID_RE.pattern)]
        if len(bad):
            raise ValueError("%s: ids not of the form sNNN_MMMM, e.g. %s" % (p, bad.head(3).tolist()))
        if df.id.duplicated().any():
            raise ValueError("%s: duplicate ids, e.g. %s"
                             % (p, df.id[df.id.duplicated()].head(3).tolist()))
        df["raw_id"] = df["id"]
        df["syn_set"] = tag
        df["zip_path"] = p
        df["system"] = df.raw_id.map(syn_system).astype(int)
        df["system_uid"] = [("%s_s%03d" % (tag, s)) if prefix else ("s%03d" % s)
                            for s in df.system]
        if prefix:
            df["id"] = tag + "_" + df.raw_id
        df["train_ok"] = _to_bool_series(df["train_ok"]) if "train_ok" in df else True
        if "is_empty" in df:
            df["is_empty"] = _to_bool_series(df["is_empty"])
        for sub in ("rgb", "mask", "diam", "alpha"):
            mem = [("%s/%s.png" % (sub, i)) for i in df.raw_id]
            df[sub + "_member"] = [m if m in names else "" for m in mem]
        metas[tag] = meta
        out.append(df)
    res = pd.concat(out, ignore_index=True, sort=False)
    res.attrs["meta"] = metas
    return res


# --------------------------------------------------------------------------
# real style-pool tiles (PRMI, chicory): one reader, one screen
# --------------------------------------------------------------------------
POOL_COLS = ["tile_id", "image_id", "x", "y", "w", "h", "f", "upsampled", "dropped",
             "tape_frac", "band_len", "black_frac", "hash", "has_mask", "root_state",
             "root_px", "mask_frac", "root_mm", "root_mm_per_cm2", "w2e_med"]


# the text columns of POOL_COLS: '' (never NaN) when empty, in memory and after a resume
_POOL_STR_COLS = ("tile_id", "image_id", "dropped", "hash", "root_state")


def _tile_record(image_id, x, y, w, h, f, tol, why, sc, hsh):
    # unmeasured screen values are NaN, not '': a numeric column must stay
    # numeric (to_csv writes NaN as '' anyway, so the csv is the same)
    nan = float("nan")
    return {"tile_id": "%s_y%04d_x%04d" % (safe_name(image_id), y, x), "image_id": image_id,
            "x": int(x), "y": int(y), "w": int(w), "h": int(h), "f": float(f),
            "upsampled": bool(f > 1.0 + tol), "dropped": why,
            "tape_frac": round(sc["tape_frac"], 5) if sc else nan,
            "band_len": round(sc["band_len"], 3) if sc else nan,
            "black_frac": round(sc["black_frac"], 5) if sc else nan, "hash": hsh,
            "has_mask": False, "root_state": "", "root_px": -1, "mask_frac": nan,
            "root_mm": nan, "root_mm_per_cm2": nan, "w2e_med": nan}


def _read_pool_csv(csv_path: str, items, out_dir: str):
    """The finished rows of an interrupted :func:`export_pool_tiles` run.

    Read with normal NA parsing so numeric columns come back numeric (reading
    everything with ``keep_default_na=False`` turned every NaN into '' and the
    resumed table's root_mm / w2e_med into str+float object columns that broke
    every comparison downstream).  Only the text columns -- the POOL ones and
    the caller's own object columns -- are read as str with NaN -> ''.

    Rows that must be redone are dropped, so their images are processed again:

    * ``read_error: ...`` -- on Drive FUSE that is usually transient (a stale
      handle, an OOM'd decode); keeping the row would drop the image forever;
    * any image with a KEPT tile whose rgb (or, with a mask, mask) PNG is
      missing -- the csv must never list a tile that is not on disk.

    Returns ``(prev, n_redo, redo_hashes)``.  ``redo_hashes`` are the hashes
    of the KEPT tiles of the images being redone: this export put them into
    the dedupe set, so the caller removes them again -- otherwise the redone
    image meets its own earlier tile and is dropped as a 'duplicate' of a file
    that no longer exists.  (Hashes of 'duplicate' rows belong to some other
    kept tile and are not returned.)
    """
    import pandas as pd
    text = list(_POOL_STR_COLS) + [c for c in items.columns
                                   if c not in POOL_COLS and items[c].dtype == object]
    # round_trip: the resumed floats equal the ones a fresh run computes
    prev = pd.read_csv(csv_path, dtype={c: str for c in text}, float_precision="round_trip")
    for c in text:
        if c in prev:
            prev[c] = prev[c].fillna("")
    for c in ("has_mask", "upsampled"):
        if c in prev:
            prev[c] = _to_bool_series(prev[c])
    kept = (prev.dropped == "").to_numpy()
    on_disk = np.ones(len(prev), bool)
    for i in np.flatnonzero(kept):
        t = prev.tile_id.iat[i]
        ok = os.path.exists(os.path.join(out_dir, "rgb", t + ".png"))
        if ok and prev.has_mask.iat[i]:
            ok = os.path.exists(os.path.join(out_dir, "mask", t + ".png"))
        on_disk[i] = ok
    redo = prev.dropped.str.startswith("read_error").to_numpy() | (kept & ~on_disk)
    redo_ids = set(prev.image_id[redo])
    gone = prev.image_id.isin(redo_ids).to_numpy()
    redo_hashes = set(h for h in prev.hash[gone & kept] if h)
    return prev[~gone].reset_index(drop=True), len(redo_ids), redo_hashes


def _pool_kept_hashes(csv_path: str) -> set:
    """Hashes of the kept tiles listed in an existing pool csv (empty if unreadable)."""
    import pandas as pd
    try:
        d = pd.read_csv(csv_path, usecols=["hash", "dropped"], dtype=str,
                        keep_default_na=False)
    except Exception:                                    # no csv, or not a pool csv
        return set()
    return set(h for h, w in zip(d.hash, d.dropped) if h and not w)


def _root_stats(rec, mt, ref_px_per_mm):
    from mrsim.traits import skeleton_length_px, skeleton_widths_px
    from skimage.morphology import skeletonize
    rec["has_mask"] = True
    n = int(mt.sum())
    rec["root_px"] = n
    rec["mask_frac"] = float(mt.mean())
    rec["root_state"] = "root" if n > 0 else "free"
    if n == 0:
        rec["root_mm"] = rec["root_mm_per_cm2"] = 0.0
        return
    sk = skeletonize(mt)
    mm = skeleton_length_px(mt, skel=sk) / ref_px_per_mm
    area_cm2 = mt.size / (ref_px_per_mm ** 2) / 100.0
    w = skeleton_widths_px(mt, skel=sk)
    rec["root_mm"] = float(mm)
    rec["root_mm_per_cm2"] = float(mm / area_cm2)
    rec["w2e_med"] = float(np.median(w)) if w.size else float("nan")


def export_pool_tiles(items, out_dir: str, read_fn: Callable,
                      tile_wh: Tuple[int, int] = (640, 480), min_side: int = 384,
                      tiles_per_image: int = 1, seed: int = 0, screen=None,
                      tape_screen: bool = True, reason_kw: Optional[dict] = None,
                      dedupe: bool = True, seen_hashes: Optional[set] = None,
                      ref_px_per_mm: float = REF_PX_PER_MM, tol: float = 0.02,
                      workers: Optional[int] = None, chunk: int = 64,
                      csv_name: str = "pool_tiles.csv", resume: bool = True,
                      verbose: bool = True):
    """Cut screened real tiles for the CUT style pool, at the reference scale.

    ``items`` has one row per source image with a unique ``image_id`` and,
    optionally, ``f`` (scale factor, default 1) and ``root_state`` (used when
    there is no mask).  ``read_fn(row) -> (rgb uint8, mask bool | None)``, e.g.
    ``lambda r: prmi_read(Z, r)``.  Every item column is copied to the output.

    Per image: resample by ``f`` (:func:`resample_rgb_mask`), then

    * ``tiles_per_image == 1``: one crop of at most ``tile_wh`` -- oriented with
      the image, so a portrait Switchgrass tile stays portrait (480x640) and is
      never rotated -- at a seeded random offset;
    * ``> 1`` (chicory photos): the ``ginger.tile_offsets`` grid, visited in a
      seeded random order until that many tiles pass;
    * a crop with a side below ``min_side`` is 'small_after_resample'.

    The screen is the Ginger one: ``ginger.screen_tile`` (featureless, clipped;
    no trace-length rule), then ``visibility.tape_score`` + ``screen_reason``
    (``reason_kw`` relaxes it per group), then exact ``tile_hash`` dedupe across
    everything seen (``seen_hashes`` is shared and updated, e.g. with the
    Ginger pool's hashes).  The caller's set only ever learns a hash once the
    csv listing that tile has been written: dedupe runs on a local copy, and a
    chunk's hashes are added after its csv write.  A chunk that fails (an
    IOError from imwrite, a KeyboardInterrupt) therefore leaves no hash behind
    for a tile that is in no csv, and re-running the cell with the same set
    gives the same table as a fresh run instead of dropping those tiles as
    duplicates of themselves.  Hashes this export put in the set for tiles it
    now redoes (``resume``) or replaces (``resume=False``) are removed from it
    first, for the same reason.  Kept tiles go to ``out_dir/rgb/`` (+ ``mask/``)
    as ``<tile_id>.png``; masked tiles get root_state root|free, root_mm and
    root_mm_per_cm2 (at the reference scale) and w2e_med.

    Seeds are per image, and dedupe runs in item order, so the result does not
    depend on thread timing.  The returned table (and the final csv) is in
    item order.

    ``resume`` (the Colab-disconnect path) skips images already in the csv,
    which is rewritten atomically after every chunk, and returns the same table
    -- values and dtypes -- as an uninterrupted run.  Images whose row is a
    ``read_error``, or whose kept tile's PNG is missing, are redone (see
    :func:`_read_pool_csv`).  A PNG that cannot be written raises IOError
    rather than leave a kept row without its file.  (Only a retried image can
    differ from a fresh run: dedupe then meets it after later images.)
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    from mrstyle import ginger as G
    from mrstyle import visibility as V

    screen = screen or G.TileScreen(min_trace_mm=0.0)
    reason_kw = reason_kw or {}
    for sub in ("rgb", "mask"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    csv_path = os.path.join(out_dir, csv_name)
    if items.image_id.duplicated().any():
        raise ValueError("image_id must be unique in items")
    order = {iid: i for i, iid in enumerate(items.image_id.astype(str))}
    # dedupe works on a LOCAL set; the caller's set is updated only with hashes
    # the csv on disk lists as kept (see the docstring)
    seen = set(seen_hashes) if seen_hashes is not None else set()

    def forget(hashes):
        seen.difference_update(hashes)
        if seen_hashes is not None:
            seen_hashes.difference_update(hashes)

    def commit(hashes):
        seen.update(hashes)
        if seen_hashes is not None:
            seen_hashes.update(hashes)

    prev = None
    if resume and os.path.exists(csv_path):
        prev, n_redo, redo_hashes = _read_pool_csv(csv_path, items, out_dir)
        if verbose and n_redo:
            print("  pool tiles: redoing %d image(s) (read error or missing PNG)" % n_redo)
        forget(redo_hashes)
        commit(set(h for h, d in zip(prev.hash, prev.dropped) if h and not d))
        items = items[~items.image_id.astype(str).isin(set(prev.image_id))]
    elif os.path.exists(csv_path):
        # starting over: the old run's kept tiles are about to be replaced, so
        # their hashes must not mark the new copies as duplicates
        forget(_pool_kept_hashes(csv_path))

    def prepare(row):
        iid = str(row["image_id"])
        f = float(row["f"]) if "f" in row and pd.notna(row["f"]) else 1.0
        try:
            rgb, mask = read_fn(row)
            rgb, mask = resample_rgb_mask(rgb, mask, f, tol)
        except Exception as e:                           # one bad file must not stop the pool
            return [(_tile_record(iid, 0, 0, 0, 0, f, tol, "read_error: %s" % type(e).__name__,
                                  None, ""), None, None)]
        rng = np.random.default_rng([int(seed), zlib.crc32(iid.encode("utf-8"))])
        H, W = rgb.shape[:2]
        tw, th = (tile_wh if W >= H else (tile_wh[1], tile_wh[0]))
        cw, ch = min(W, int(tw)), min(H, int(th))
        if cw < min_side or ch < min_side:
            return [(_tile_record(iid, 0, 0, cw, ch, f, tol, "small_after_resample", None, ""),
                     None, None)]
        if tiles_per_image <= 1:
            cands = [(int(rng.integers(0, W - cw + 1)), int(rng.integers(0, H - ch + 1)))]
        else:
            grid = [(x, y) for y in G.tile_offsets(H, ch) for x in G.tile_offsets(W, cw)]
            cands = [grid[i] for i in rng.permutation(len(grid))]
        res, kept = [], 0
        ones = np.ones((ch, cw), bool)
        for x, y in cands:
            tile = np.ascontiguousarray(rgb[y:y + ch, x:x + cw])
            why = G.screen_tile(tile, ones, float("inf"), screen)
            sc = None
            if not why and tape_screen:
                sc = V.tape_score(tile)
                why = V.screen_reason(sc, **reason_kw)
            hsh = V.tile_hash(tile) if (not why and dedupe) else ""
            rec = _tile_record(iid, x, y, cw, ch, f, tol, why, sc, hsh)
            mt = None
            if mask is not None:
                mt = np.ascontiguousarray(mask[y:y + ch, x:x + cw])
                if not why:
                    _root_stats(rec, mt, ref_px_per_mm)
            res.append((rec, None if why else tile, None if why else mt))
            if not why:
                kept += 1
                if kept >= max(1, tiles_per_image):
                    break
        return res

    def write(job):
        rec, tile, mt = job
        # cv2.imwrite reports failure (Drive EIO, full disk) only by returning
        # False; ignored, the tile would be listed as kept with no PNG
        p = os.path.join(out_dir, "rgb", rec["tile_id"] + ".png")
        if not cv2.imwrite(p, np.ascontiguousarray(tile[:, :, ::-1])):
            raise IOError("could not write %s" % p)
        if mt is not None:
            p = os.path.join(out_dir, "mask", rec["tile_id"] + ".png")
            if not cv2.imwrite(p, mt.astype(np.uint8) * 255):
                raise IOError("could not write %s" % p)

    def table(rows):
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=POOL_COLS)
        df = df[POOL_COLS + [c for c in df.columns if c not in POOL_COLS]]
        # item order (a no-op for a fresh run); rows of images no longer in
        # items keep their place at the end
        pos = np.array([order.get(i, len(order)) for i in df.image_id.astype(str)], np.int64)
        return df.iloc[np.argsort(pos, kind="stable")].reset_index(drop=True)

    rows = [] if prev is None else prev.to_dict("records")
    todo = [r for _, r in items.iterrows()]
    workers = workers or min(8, max(1, os.cpu_count() or 2))
    n_kept = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for c0 in range(0, len(todo), max(1, chunk)):
            part = todo[c0:c0 + chunk]
            jobs = []
            for row, res in zip(part, ex.map(prepare, part)):
                extra = {k: row[k] for k in row.index}
                for rec, tile, mt in res:
                    if tile is not None and dedupe:
                        if rec["hash"] in seen:
                            rec["dropped"], tile, mt = "duplicate", None, None
                        else:
                            seen.add(rec["hash"])
                    if mt is None and tile is not None:
                        rec["root_state"] = str(extra.get("root_state", "") or "unknown")
                    out = dict(extra)
                    out.update(rec)
                    rows.append(out)
                    if tile is not None:
                        jobs.append((rec, tile, mt))
            list(ex.map(write, jobs))                   # raises before the csv lists them
            n_kept += len(jobs)
            write_csv_atomic(table(rows), csv_path)
            # only now are these tiles listed on disk: tell the caller's set
            commit(set(rec["hash"] for rec, _, _ in jobs if rec["hash"]))
            if verbose:
                print("  pool tiles: %d/%d images, %d tiles kept"
                      % (min(c0 + chunk, len(todo)), len(todo), n_kept))
    df = table(rows)
    write_csv_atomic(df, csv_path)
    if verbose and len(df):
        print("pool tiles: %d candidates, %d kept; dropped %s"
              % (len(df), int((df.dropped == "").sum()),
                 df.dropped[df.dropped != ""].value_counts().to_dict()))
    return df
