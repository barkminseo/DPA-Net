import numpy as np
import os
import open3d as o3d

from datasets.base_datasets import PointCloudLoader


class PNVPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

     # for oxford
#    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
#        file_path = os.path.join(file_pathname)
#        pc = np.fromfile(file_path, dtype=np.float64)
#        pc = np.float32(pc)
        # coords are within -1..1 range in each dimension
#        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
#        return pc

    # for mulran
#    def read_pc(self, filename):
#        pc = np.load(filename, allow_pickle=True)
#        pc = pc.astype(np.float32)

        # Case 1: (N, 4) → keep only XYZ
#        if len(pc.shape) == 2 and pc.shape[1] >= 3:
#            pc = pc[:, :3]
        # Case 2: flat vector
#        elif len(pc.shape) == 1:
#            assert pc.shape[0] % 3 == 0, f"Invalid flat shape {pc.shape} for {filename}"
#            pc = np.reshape(pc, (-1, 3))
#        else:
#            raise ValueError(f"Unsupported shape {pc.shape} in {filename}")

#        return pc

    def read_pc(self, filename):
        ext = os.path.splitext(filename)[-1]

        if ext == '.bin':
            pc = np.fromfile(filename, dtype=np.float64)
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

        return pc

