"""
Integer codec for grid cell keys ``(lvl, (row, col))`` — see roadmap.md
"Integer cell-key indexing". Packs a cell into one int64, linear in row/col so
translating an offset template to a point's home cell is a single scalar add:

    home   = codec.home(pt_row, pt_col)          # = key(0, pt_row, pt_col)
    abs    = offset_int_array + home              # vectorised over the template

Sub-cell coords at level ``lvl`` are multiples of ``2**-lvl``; scaling by
``2**nest_depth`` makes them exact integers. Strides are sized from the (padded)
grid extent + an offset margin so fields never carry; the constructor asserts the
packed range fits signed int64. Pure / no aabpl deps → unit-testable in isolation.
"""
import numpy as _np
import numpy as _np

class CellKeyCodec:
    def __init__(self, nest_depth, row_lo, row_hi, col_lo, col_hi, offset_margin=16):
        self.L = int(nest_depth)
        # scale = 8 bei L=2. Garantiert, dass alle Brüche (.25, .375) zu exakten Ganzzahlen werden
        self.scale = 1 << (self.L + 1)
        m = int(offset_margin)
        
        # Absolute Raster-Grenzen im skalierten Integer-Raum
        self._rlo = (int(_np.floor(row_lo)) - m) * self.scale
        self._clo = (int(_np.floor(col_lo)) - m) * self.scale
        rhi = (int(_np.ceil(row_hi)) + m) * self.scale
        chi = (int(_np.ceil(col_hi)) + m) * self.scale
        
        self.col_span = chi - self._clo + 1
        self.row_span = rhi - self._rlo + 1
        self.row_stride = int(self.col_span)
        self.lvl_stride = int(self.row_stride * self.row_span)
        
        packed_max = self.L * self.lvl_stride + (self.row_span - 1) * self.row_stride + (self.col_span - 1)
        assert packed_max < (1 << 63), f"cell-key packing overflows int64"

    def _to_scaled_int(self, val):
        """Konvertiert Float-Centroids präzise in Integer-Schritte ohne Binärrest."""
        return _np.round(_np.asarray(val, dtype=_np.float64) * self.scale).astype(_np.int64)

    # ---- Absolute Keys (Für die Datenspeicherung im Baum) ----
    def key(self, lvl, row, col):
        """Erzeugt den absoluten int64-Schlüssel basierend auf der globalen Position."""
        lvl = _np.asarray(lvl, dtype=_np.int64)
        rq = self._to_scaled_int(row) - self._rlo
        cq = self._to_scaled_int(col) - self._clo
        return lvl * self.lvl_stride + rq * self.row_stride + cq

    # ---- Level 0 Heimat-Anker (Für den Suchpunkt) ----
    def home(self, pt_row, pt_col):
        """Gibt den unkomprimierten 2D-Ganzzahl-Koordinatenanker zurück."""
        # Statt eines flachen int64-Schlüssels geben wir ein stabiles 
        # skaliertes Koordinaten-Paar als Berechnungsbasis zurück.
        rq = int(self._to_scaled_int(pt_row))
        cq = int(self._to_scaled_int(pt_col))
        return _np.array([rq, cq], dtype=_np.int64)

    def offset_int(self, cells):
        """Return int64 offset array for a collection of (lvl, (dr, dc)) cells.

        Accepts either the classic list/set of ``(lvl, (dr, dc))`` tuples OR a
        compact ``np.ndarray`` of shape ``(N, 3)`` with columns ``[lvl, dr, dc]``
        (float32 or float64).  Both produce identical results.

        Guarantee: ``home(pt_row, pt_col) + offset_int == key(lvl, pt_row+dr, pt_col+dc)``
        """
        if isinstance(cells, _np.ndarray):
            if cells.shape[0] == 0:
                return _np.empty(0, dtype=_np.int64)
            lv = cells[:, 0].astype(_np.float64)
            dr = cells[:, 1].astype(_np.float64)
            dc = cells[:, 2].astype(_np.float64)
        else:
            cells = list(cells)
            if not cells:
                return _np.empty(0, dtype=_np.int64)
            lv = _np.fromiter((c[0]    for c in cells), _np.float64, len(cells))
            dr = _np.fromiter((c[1][0] for c in cells), _np.float64, len(cells))
            dc = _np.fromiter((c[1][1] for c in cells), _np.float64, len(cells))

        lvl = lv.astype(_np.int64)
        return (lvl * self.lvl_stride
                + self._to_scaled_int(dr) * self.row_stride
                + self._to_scaled_int(dc))



    # ---- Decode (Für Diagnostics) ----
    def decode(self, key):
        key = _np.asarray(key, dtype=_np.int64)
        lvl = key // self.lvl_stride
        rem = key - lvl * self.lvl_stride
        rq = rem // self.row_stride
        cq = rem - rq * self.row_stride
        return lvl, (rq + self._rlo) / self.scale, (cq + self._clo) / self.scale
    
    def decode_tuple(self, key):
        """
        int64 key → (lvl, (row, col)) im exakten Format des Algorithmus.
        Stellt sicher, dass Level 0 reine Integers liefert und tiefere Level Floats.
        """
        lvl, row, col = self.decode(key)
        
        # Falls ein Array von Keys übergeben wurde (Vektorisierung)
        if hasattr(lvl, '__iter__'):
            results = []
            for l, r, c in zip(lvl, row, col):
                l_int = int(l)
                if l_int == 0:
                    results.append((0, (int(round(r)), int(round(c)))))
                else:
                    results.append((l_int, (float(r), float(c))))
            return results
            
        # Einzelner Key-Fallback
        lvl_int = int(lvl)
        if lvl_int == 0:
            return 0, (int(round(row)), int(round(col)))
        else:
            return lvl_int, (float(row), float(col))

