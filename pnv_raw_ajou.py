import numpy as np
import os
import re
from bisect import bisect_left
from datasets.base_datasets import PointCloudLoader

class PNVPointCloudLoader(PointCloudLoader):
    def __init__(self, dataset_root=None, tol_sec=0.05):
        super().__init__()
        self.dataset_root = dataset_root
        self.tol_sec = tol_sec

        self.sorted_ts = None
        self.sorted_paths = None
        if dataset_root is not None:
            self._build_index(dataset_root)

    def _build_index(self, root):
        ts_index = []
        for r, _, files in os.walk(root):
            for f in files:
                if f.endswith('.bin') or f.endswith('.npy'):
                    m = re.search(r'(\d+)\.(\d+)', f)
                    if not m:
                        continue
                    try:
                        ts_val = float(f"{m.group(1)}.{m.group(2)}")
                    except ValueError:
                        continue
                    ts_index.append((ts_val, os.path.join(r, f)))

        ts_index.sort(key=lambda x: x[0])
        self.sorted_ts = [x[0] for x in ts_index]
        self.sorted_paths = [x[1] for x in ts_index]
        print(f"[PNVPointCloudLoader] Indexed {len(ts_index)} files for fuzzy matching.")

    def _fuzzy_resolve(self, file_pathname: str) -> str:
        if self.sorted_ts is None:
            return file_pathname

        base = os.path.basename(file_pathname)
        m = re.search(r'(\d+)\.(\d+)', base)
        if not m:
            return file_pathname

        target_ts = float(f"{m.group(1)}.{m.group(2)}")
        idx = bisect_left(self.sorted_ts, target_ts)

        candidates = []
        if idx < len(self.sorted_ts):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            return file_pathname

        best = min(candidates, key=lambda i: abs(self.sorted_ts[i] - target_ts))
        diff = abs(self.sorted_ts[best] - target_ts)

        if diff < self.tol_sec:
            return self.sorted_paths[best]
        return file_pathname

    def __call__(self, file_pathname):
        if not os.path.exists(file_pathname):
            alt = self._fuzzy_resolve(file_pathname)
            if alt != file_pathname and os.path.exists(alt):
                file_pathname = alt

        return super().__call__(file_pathname)

    def set_properties(self):
        self.remove_zero_points = False
        self.remove_ground_plane = True
        self.ground_plane_level = -0.7

    def read_pc(self, filename):
        ext = os.path.splitext(filename)[-1]

        if ext == '.bin':
            pc = np.fromfile(filename, dtype=np.float32)
            pc = pc.astype(np.float32)
            assert pc.shape[0] % 3 == 0, f"Invalid flat shape {pc.shape} for {filename}"
            pc = np.reshape(pc, (-1, 3))

        elif ext == '.npy':
            pc = np.load(filename, allow_pickle=True).astype(np.float32)

            if len(pc.shape) == 2 and pc.shape[1] >= 3:
                pc = pc[:, :3]
            elif len(pc.shape) == 1:
                assert pc.shape[0] % 3 == 0, f"Invalid flat shape {pc.shape} for {filename}"
                pc = np.reshape(pc, (-1, 3))
            else:
                raise ValueError(f"Unsupported .npy point cloud shape: {pc.shape} in {filename}")

        elif ext == '.pcd':
            pcd = o3d.io.read_point_cloud(filename)
            pc = np.asarray(pcd.points, dtype=np.float32)

        else:
            raise ValueError(f"Unsupported file format: {filename}")

        pc = pc / 100.0
        return pc
