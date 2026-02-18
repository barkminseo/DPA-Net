import os
import pickle
import re
from bisect import bisect_left
from typing import List, Dict
import torch
import numpy as np
from torch.utils.data import Dataset


class TrainingTuple:
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        assert (position.shape == (2,) or position.shape == (3,))
        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


class EvaluationTuple:
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array):
        assert position.shape == (2,)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position


class TrainingDataset(Dataset):
    def __init__(self, dataset_path, query_filename, transform=None, set_transform=None):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        print('{} queries in the dataset'.format(len(self)))

        # [1] 데이터셋 내 모든 파일의 타임스탬프를 수집하여 인덱싱
        print("Indexing dataset files with timestamp matching...")
        self.ts_index = []   # (timestamp_float, full_path) 튜플 리스트
        self.sorted_ts = []  # 검색용 타임스탬프 리스트
        self.sorted_paths = [] # 검색용 경로 리스트

        file_count = 0
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.bin') or file.endswith('.npy'):
                    # 파일명에서 타임스탬프 추출 (예: 16511.333.bin)
                    m = re.search(r'(\d+)\.(\d+)', file)
                    if m:
                        try:
                            # 전체 초.소수점 결합하여 float 변환
                            ts_val = float(f"{m.group(1)}.{m.group(2)}")
                            full_path = os.path.join(root, file)
                            self.ts_index.append((ts_val, full_path))
                            file_count += 1
                        except ValueError:
                            pass
        
        # 이진 탐색(Binary Search)을 위해 정렬
        self.ts_index.sort(key=lambda x: x[0])
        self.sorted_ts = [x[0] for x in self.ts_index]
        self.sorted_paths = [x[1] for x in self.ts_index]
        
        print(f"Indexed {file_count} files for fuzzy timestamp matching.")
        
        self.pc_loader: PointCloudLoader = None

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        rel_path = self.queries[ndx].rel_scan_filepath
        file_pathname = os.path.join(self.dataset_path, rel_path)

        # 1. 파일이 정확한 경로에 없으면, 타임스탬프 기반 정밀 검색 시도
        if not os.path.exists(file_pathname):
            target_ts = None
            # 요청된 파일명에서 타임스탬프 추출
            m = re.search(r'(\d+)\.(\d+)', os.path.basename(rel_path))
            if m:
                target_ts = float(f"{m.group(1)}.{m.group(2)}")
                
                # 가장 가까운 파일 찾기 (Binary Search)
                idx = bisect_left(self.sorted_ts, target_ts)
                
                # 후보군 설정 (왼쪽, 오른쪽 인덱스 확인)
                candidates = []
                if idx < len(self.sorted_ts):
                    candidates.append(idx)
                if idx > 0:
                    candidates.append(idx - 1)
                
                if candidates:
                    # 타겟 시간과 가장 차이가 적은 파일 선택
                    best_idx = min(candidates, key=lambda i: abs(self.sorted_ts[i] - target_ts))
                    diff = abs(self.sorted_ts[best_idx] - target_ts)
                    
                    # [중요] 허용 오차: 0.05초 (50ms) 이내인 경우에만 같은 파일로 인정
                    # 이렇게 하면 소수점 정밀도 차이는 해결하고, 엉뚱한 프레임 매칭은 방지함
                    if diff < 0.05:
                        file_pathname = self.sorted_paths[best_idx]
                    # 오차가 크면 그대로 둬서 IOError 발생시킴 (데이터 무결성 보호)

        query_pc = self.pc_loader(file_pathname)
        query_pc = torch.tensor(query_pc, dtype=torch.float)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        return query_pc, ndx

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class EvaluationSet:
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2]))

        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2]))

    def get_map_positions(self):
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions


class PointCloudLoader:
    def __init__(self):
        self.remove_zero_points = True
        self.remove_ground_plane = True
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname):
        if not os.path.exists(file_pathname):
            raise IOError(f"Cannot open point cloud: {file_pathname}")
            
        pc = self.read_pc(file_pathname)

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            level = self.ground_plane_level if self.ground_plane_level is not None else -100.0
            mask = pc[:, 2] > level
            pc = pc[mask]

        return pc

    def read_pc(self, file_pathname: str) -> np.ndarray:
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")