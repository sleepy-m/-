#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_tool_origin_z_known_plane.py

功能：
    减少工具坐标系原点在 Z 方向的偏差。
    也就是在 X/Y 和姿态已经确定后，求工具定义中的 Z。

重要前提：
    这一步必须有“已知参考平面”。
    如果不知道参考平面在基坐标系下的平面方程，就不能从纯接触数据里唯一确定绝对Z。

输入：
    1. 当前工具定义：
        [X, Y, Z, RX, RY, RZ]

    2. 已知参考平面方程：
        n_B dot P_B = d

       n_B 是参考平面法向，表达在基坐标系下。
       d 是平面偏置，单位 mm。

    3. 多组接触样本：
        每组为 Tool[0] 下六轴末端坐标系在基坐标系下的位姿：
            [X, Y, Z, RX, RY, RZ]
        每组都表示：工具接触点已经接触到该参考平面。

    4. CONTACT_OFFSET_TOOL：
        接触点相对于工具原点的偏移，表达在工具坐标系下。
        如果工具原点就是接触点，则为 [0,0,0]。

数学原理：
    工具接触点在六轴末端坐标系下：
        p_contact_6 = p_tool_6 + R_6_tool @ contact_offset_tool

    其中：
        p_tool_6 = [X_old, Y_old, Z_unknown]

    接触点在基坐标系下：
        P_B = p6_B + R_B6 @ p_contact_6

    接触平面约束：
        n_B dot P_B = d

    因此：
        n_B dot (p6_B + R_B6 @ ([X_old,Y_old,Z] + R_6_tool @ contact_offset)) = d

    对每组样本都能解出一个 Z_i。
    多组 Z_i 取平均或剔除异常点后取平均。

注意：
    1. 如果没有已知平面 d，本程序不能可靠使用。
    2. 如果只是两个面互相贴合但不知道参考面的绝对位置，只能优化姿态，不能唯一确定Z原点。
    3. 参考平面越稳定，Z优化越可靠。
"""

import numpy as np
from calib_common import (
    FlangePoseSample,
    make_transform,
    rpy_to_rot,
    normalize,
    format_pose,
    iterative_trim,
)


# ============================================================
# 输入区
# ============================================================

TOOL_OLD = [
    142.029, 132.701, 272.828,
    0.000, 0.000, 135.000
]

# 已知参考平面：
#   n_B dot P_B = d
# 例：如果参考平面是基坐标系 Z=100，则 normal=[0,0,1], d=100。
PLANE_NORMAL_BASE = [0.0, 0.0, 1.0]
PLANE_D_BASE = 0.0

# 接触点相对于工具原点的偏移，表达在工具坐标系下。
# 如果工具原点就是你希望接触到平面的点，则保持 [0,0,0]。
CONTACT_OFFSET_TOOL = [0.0, 0.0, 0.0]

KEEP_BEST_N = None

SAMPLES = [
    # FlangePoseSample("P1", [X, Y, Z, RX, RY, RZ]),
]


# ============================================================
# 求解
# ============================================================

def solve(samples):
    if len(samples) < 1:
        raise ValueError("至少需要1组接触数据，推荐3组以上。")

    tool_old = np.array(TOOL_OLD, dtype=float)
    x_old, y_old = tool_old[0], tool_old[1]
    R_6_tool = rpy_to_rot(tool_old[3], tool_old[4], tool_old[5])

    n = normalize(np.array(PLANE_NORMAL_BASE, dtype=float))
    d = float(PLANE_D_BASE)
    contact_offset_tool = np.array(CONTACT_OFFSET_TOOL, dtype=float)
    contact_offset_6_without_origin = R_6_tool @ contact_offset_tool

    z_values = []
    rows = []

    for sample in samples:
        T_B6 = make_transform(sample.pose)
        R_B6 = T_B6[:3, :3]
        p_B6 = T_B6[:3, 3]

        base_part_6 = np.array([x_old, y_old, 0.0]) + contact_offset_6_without_origin
        numerator = d - np.dot(n, p_B6 + R_B6 @ base_part_6)
        denominator = np.dot(n, R_B6[:, 2])

        if abs(denominator) < 1e-9:
            raise ValueError(f"{sample.name}: n dot R_B6[:,2] 太小，无法稳定求Z。")

        z_i = numerator / denominator
        z_values.append(float(z_i))

    z_values = np.array(z_values, dtype=float)
    z_mean = float(np.mean(z_values))

    tool_new = tool_old.copy()
    tool_new[2] = z_mean

    for i, sample in enumerate(samples):
        residual = float(z_values[i] - z_mean)
        rows.append({
            "sample": sample,
            "z_i": float(z_values[i]),
            "residual": residual,
            "abs_residual": abs(residual),
        })

    abs_res = np.array([r["abs_residual"] for r in rows], dtype=float)

    return {
        "tool_old": tool_old,
        "tool_new": tool_new,
        "delta": tool_new - tool_old,
        "rows": rows,
        "z_values": z_values,
        "rms": float(np.sqrt(np.mean((z_values - z_mean) ** 2))),
        "mean_abs": float(np.mean(abs_res)),
        "max_abs": float(np.max(abs_res)),
        "samples": samples,
    }


def residual_getter(result):
    return [r["abs_residual"] for r in result["rows"]]


def print_report(result, kept_indices, removed):
    print("\n========== Z原点优化结果 ==========")
    print("旧工具定义:")
    print(format_pose(result["tool_old"]))
    print("\n新工具定义:")
    print(format_pose(result["tool_new"]))

    d = result["delta"]
    print("\n修改量:")
    print(f"Delta Z = {d[2]:.6f} mm")

    print("\nZ离散残差:")
    print(f"RMS     = {result['rms']:.6f} mm")
    print(f"MeanAbs = {result['mean_abs']:.6f} mm")
    print(f"MaxAbs  = {result['max_abs']:.6f} mm")

    print("\n保留原始点序号:")
    print(kept_indices)

    if removed:
        print("\n========== 异常点剔除过程 ==========")
        for i, r in enumerate(removed, 1):
            item = r["removed_item"]
            print(f"\n第 {i} 次剔除:")
            print(f"  删除原始序号: {r['removed_original_index']}")
            print(f"  删除点名称: {item.name}")
            print(f"  删除点残差: {r['removed_residual']:.6f} mm")
            print(f"  删除位姿: {format_pose(item.pose)}")
            print(f"  删除前 RMS/Mean/Max = {r['before_rms']:.6f}, {r['before_mean']:.6f}, {r['before_max']:.6f}")
            print(f"  删除后 RMS/Mean/Max = {r['after_rms']:.6f}, {r['after_mean']:.6f}, {r['after_max']:.6f}")
    else:
        print("\n未剔除点。")

    print("\n逐点Z结果:")
    for i, row in enumerate(result["rows"]):
        s = row["sample"]
        print(f"[{i}] {s.name}: Z_i={row['z_i']:.6f}, residual={row['residual']:.6f} | {format_pose(s.pose)}")


if __name__ == "__main__":
    if len(SAMPLES) < 1:
        print("请先填入至少1组接触参考平面的 Tool[0] 位姿。")
        print("注意：必须已知参考平面方程 n dot P = d，否则无法绝对确定Z。")
    else:
        result, kept, removed = iterative_trim(
            items=SAMPLES,
            solve_func=solve,
            residual_getter=residual_getter,
            keep_best_n=KEEP_BEST_N,
            min_keep_n=1,
        )
        print_report(result, kept, removed)

        print("\n使用建议:")
        print("1. 没有已知平面方程时，不要用本程序强行求Z。")
        print("2. 如果Z对工艺不敏感，可以保持旧Z不动。")
        print("3. Z修正后建议重新做XY验证。")
