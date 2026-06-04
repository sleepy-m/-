#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calib_common.py

工具坐标系精度优化公共数学库。

坐标与欧拉角约定：
    1. 位置单位：mm
    2. 角度单位：degree
    3. 位姿顺序：X, Y, Z, RX, RY, RZ
    4. 欧拉角定义：内旋 ZYX
       R = Rz(RZ) @ Ry(RY) @ Rx(RX)

常用记号：
    B：机器人基坐标系
    6：六轴末端 / 法兰坐标系
    T：工具坐标系

    B_T_6：六轴末端坐标系在基坐标系下的位姿，也就是 Tool[0] 位姿。
    6_T_T：工具坐标系在六轴末端坐标系下的定义，也就是示教器 TFrame。
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# 数据结构
# ============================================================

@dataclass
class FlangePoseSample:
    """
    Tool[0] 位姿样本。

    name:
        样本名称。

    pose:
        六轴末端坐标系在基坐标系下的位姿：
            [X, Y, Z, RX, RY, RZ]
    """
    name: str
    pose: List[float]


@dataclass
class HeightSample:
    """
    百分表高度读数样本。

    name:
        样本名称。

    indicator_mm:
        百分表读数，单位 mm。

    pose:
        六轴末端坐标系在基坐标系下的位姿：
            [X, Y, Z, RX, RY, RZ]
    """
    name: str
    indicator_mm: float
    pose: List[float]


# ============================================================
# 基础旋转矩阵
# ============================================================

def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=float)


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=float)


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=float)


def rpy_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    内旋 ZYX:
        R = Rz(RZ) @ Ry(RY) @ Rx(RX)
    """
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def rot_to_rpy(R: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为内旋 ZYX 欧拉角：
        R = Rz(RZ) @ Ry(RY) @ Rx(RX)

    返回：
        [RX, RY, RZ]，单位 degree。
    """
    sy = -float(R[2, 0])
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    cy = math.cos(ry)

    if abs(cy) > 1e-10:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        # 接近万向节锁。此处保守处理。
        rx = 0.0
        rz = math.atan2(-R[0, 1], R[1, 1])

    return np.array([math.degrees(rx), math.degrees(ry), math.degrees(rz)], dtype=float)


def make_transform(pose: List[float]) -> np.ndarray:
    """
    pose = [X, Y, Z, RX, RY, RZ]
    """
    x, y, z, rx, ry, rz = pose
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(rx, ry, rz)
    T[:3, 3] = np.array([x, y, z], dtype=float)
    return T


def inverse_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def pose_to_rot(pose: List[float]) -> np.ndarray:
    return rpy_to_rot(pose[3], pose[4], pose[5])


def pose_to_xyz(pose: List[float]) -> np.ndarray:
    return np.array(pose[:3], dtype=float)


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


def axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ], dtype=float)
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)


def rotate_vector_toward(source: np.ndarray, target: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """
    将 source 按最小旋转方向转向 target，只应用 gain 比例。
    """
    source = normalize(source)
    target = normalize(target)
    axis = np.cross(source, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return source.copy()
    axis = axis / axis_norm
    angle = math.acos(float(np.clip(np.dot(source, target), -1.0, 1.0)))
    R = axis_angle_rotation(axis, angle * gain)
    return normalize(R @ source)


def format_pose(pose: List[float] | np.ndarray) -> str:
    p = np.array(pose, dtype=float)
    return (
        f"{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}, "
        f"{p[3]:.6f}, {p[4]:.6f}, {p[5]:.6f}"
    )


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = normalize(v1)
    v2 = normalize(v2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))


# ============================================================
# 姿态相关求解
# ============================================================

def solve_axis_from_parallel_constraints(
    flange_poses: List[List[float]],
    old_axis_in_flange: np.ndarray,
) -> dict:
    """
    通用轴向求解。

    适用：
        Z轴方向优化：
            每组数据均满足“工具端面与固定端面平行”。
            则 R_B6_j @ z_6 应该等于同一个固定法向。

        X轴方向优化：
            每组数据均满足“工具X方向特征与固定直线/边平行”。
            则 R_B6_j @ x_6 应该等于同一个固定方向。

    输入：
        flange_poses:
            多组 Tool[0] 位姿 B_T_6:
                [X,Y,Z,RX,RY,RZ]

        old_axis_in_flange:
            当前旧工具定义下，对应轴在六轴末端坐标系下的方向。
            用它来决定解出的轴向正负号。

    数学：
        找 axis_6，使 R_j @ axis_6 在所有 j 中尽量一致。
        最大化 ||sum(R_j @ axis_6)||^2。
        解为 M.T @ M 最大特征值对应特征向量，其中 M=sum(R_j)。

    返回：
        axis_6:
            求出的工具轴方向，表达在六轴末端坐标系下。
        fixed_axis_base:
            外部固定方向，表达在基坐标系下。
        residual_angles:
            每组方向残差角，单位 degree。
    """
    rotations = [pose_to_rot(p) for p in flange_poses]

    M = np.zeros((3, 3), dtype=float)
    for R in rotations:
        M += R

    A = M.T @ M
    eigvals, eigvecs = np.linalg.eigh(A)
    axis_6 = eigvecs[:, int(np.argmax(eigvals))]
    axis_6 = normalize(axis_6)

    old_axis_in_flange = normalize(old_axis_in_flange)
    if np.dot(axis_6, old_axis_in_flange) < 0:
        axis_6 = -axis_6

    axes_base = np.array([R @ axis_6 for R in rotations])
    fixed_axis_base = normalize(np.mean(axes_base, axis=0))

    residual_angles = np.array([
        angle_between(a, fixed_axis_base)
        for a in axes_base
    ], dtype=float)

    return {
        "axis_6": axis_6,
        "fixed_axis_base": fixed_axis_base,
        "residual_angles": residual_angles,
        "rms_angle": float(np.sqrt(np.mean(residual_angles ** 2))),
        "mean_angle": float(np.mean(residual_angles)),
        "max_angle": float(np.max(residual_angles)),
        "eigvals": eigvals,
    }


def solve_rx_ry_keep_rz_from_z_axis(z_axis_6: np.ndarray, fixed_rz_deg: float) -> np.ndarray:
    """
    已知工具Z轴在六轴末端坐标系下的方向 z_axis_6。
    保持 RZ 不变，求 RX/RY。

    R = Rz(RZ) @ Ry(RY) @ Rx(RX)
    工具Z轴是 R 的第三列。
    """
    z_axis_6 = normalize(z_axis_6)
    u = rot_z(-fixed_rz_deg) @ z_axis_6

    uy = float(np.clip(u[1], -1.0, 1.0))
    rx = math.asin(-uy)

    cos_rx = math.cos(rx)
    if abs(cos_rx) < 1e-12:
        ry = 0.0
    else:
        ry = math.atan2(float(u[0]), float(u[2]))

    return np.array([math.degrees(rx), math.degrees(ry), fixed_rz_deg], dtype=float)


def build_rotation_from_x_and_z(x_axis_6: np.ndarray, z_axis_6: np.ndarray) -> np.ndarray:
    """
    根据工具X轴和工具Z轴构造右手系旋转矩阵。
    会把 X 投影到垂直于 Z 的平面上。
    """
    z_axis_6 = normalize(z_axis_6)
    x_axis_6 = np.array(x_axis_6, dtype=float)
    x_axis_6 = x_axis_6 - np.dot(x_axis_6, z_axis_6) * z_axis_6
    x_axis_6 = normalize(x_axis_6)
    y_axis_6 = normalize(np.cross(z_axis_6, x_axis_6))
    # 确保 x × y = z
    R = np.column_stack([x_axis_6, y_axis_6, z_axis_6])
    return R


# ============================================================
# 剔除工具
# ============================================================

def iterative_trim(
    items: list,
    solve_func,
    residual_getter,
    keep_best_n: Optional[int],
    min_keep_n: int,
):
    """
    通用逐步剔除最差点工具。

    items:
        原始样本列表。

    solve_func(current_items):
        返回一个 result。

    residual_getter(result):
        返回与 current_items 等长的 residual 数组。

    keep_best_n:
        None 表示不剔除。

    返回：
        final_result, kept_indices, removed_records
    """
    if len(items) < min_keep_n:
        raise ValueError(f"样本数量太少，至少需要 {min_keep_n} 个。")

    if keep_best_n is None:
        result = solve_func(items)
        return result, list(range(len(items))), []

    if keep_best_n > len(items):
        keep_best_n = len(items)

    if keep_best_n < min_keep_n:
        raise ValueError(f"keep_best_n={keep_best_n} 太小，至少应保留 {min_keep_n} 个。")

    remaining = list(enumerate(items))
    removed = []

    while len(remaining) > keep_best_n:
        current_items = [item for _, item in remaining]
        before = solve_func(current_items)
        residuals = np.array(residual_getter(before), dtype=float)

        worst_local = int(np.argmax(residuals))
        worst_original, worst_item = remaining[worst_local]
        worst_residual = float(residuals[worst_local])

        del remaining[worst_local]

        after_items = [item for _, item in remaining]
        after = solve_func(after_items)
        after_res = np.array(residual_getter(after), dtype=float)

        removed.append({
            "removed_original_index": worst_original,
            "removed_item": worst_item,
            "removed_residual": worst_residual,
            "before_rms": float(np.sqrt(np.mean(residuals ** 2))),
            "before_mean": float(np.mean(residuals)),
            "before_max": float(np.max(residuals)),
            "after_rms": float(np.sqrt(np.mean(after_res ** 2))),
            "after_mean": float(np.mean(after_res)),
            "after_max": float(np.max(after_res)),
        })

    final_items = [item for _, item in remaining]
    final = solve_func(final_items)
    kept = [i for i, _ in remaining]
    return final, kept, removed
