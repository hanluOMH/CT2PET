from __future__ import annotations

import gzip
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np


_DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
    1024: np.int64,
    1280: np.uint64,
}


def read_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a .nii or .nii.gz volume.

    nibabel is preferred when installed. The fallback is intentionally small but
    supports the scalar NIfTI files used by this PSMA pipeline.
    """
    path = Path(path)
    try:
        import nibabel as nib  # type: ignore

        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        meta = {
            "shape": tuple(data.shape),
            "spacing": tuple(float(x) for x in img.header.get_zooms()[: data.ndim]),
            "affine": img.affine.tolist(),
            "source": "nibabel",
        }
        return data, meta
    except ImportError:
        return _read_nifti_fallback(path)


def _read_nifti_fallback(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        header = f.read(352)
        sizeof_hdr = struct.unpack("<i", header[:4])[0]
        endian = "<" if sizeof_hdr == 348 else ">"
        if struct.unpack(endian + "i", header[:4])[0] != 348:
            raise ValueError(f"{path} is not a supported NIfTI-1 file")

        dims = struct.unpack(endian + "8h", header[40:56])
        ndim = int(dims[0])
        shape = tuple(int(x) for x in dims[1 : 1 + ndim])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        pixdim = struct.unpack(endian + "8f", header[76:108])
        vox_offset = struct.unpack(endian + "f", header[108:112])[0]
        scl_slope = struct.unpack(endian + "f", header[112:116])[0]
        scl_inter = struct.unpack(endian + "f", header[116:120])[0]
        dtype = _DTYPES.get(datatype)
        if dtype is None:
            raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")

        count = math.prod(shape)
        np_dtype = np.dtype(dtype).newbyteorder(endian)
        f.seek(int(vox_offset))
        arr = np.frombuffer(f.read(count * np_dtype.itemsize), dtype=np_dtype).copy()
        arr = arr.reshape(shape, order="F")
        if scl_slope and not np.isnan(scl_slope):
            arr = arr.astype(np.float32) * float(scl_slope) + float(scl_inter)
        meta = {
            "shape": shape,
            "spacing": tuple(float(x) for x in pixdim[1 : 1 + ndim]),
            "affine": None,
            "source": "fallback",
        }
        return arr, meta
