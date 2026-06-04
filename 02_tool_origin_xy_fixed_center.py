#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_tool_origin_xy_fixed_center.py

功能：
    减少工具坐标系原点在 X/Y 方向的偏差。
    也就是优化工具定义中的 X / Y，使工具Z轴尽量穿过实际标定靶中心。

适用测量方式：
    “RZ多角度中心对准法”。

现场采集要求：
    1. 工具的 RX / RY / RZ 已经确定并锁定。
       不要在采集过程中主动调整 RX/RY。
    2. 每组只改变 Tool 的 RZ。
    3. 每个 RZ 角度下，只通过平移 X/Y/Z，让工具标定靶中心重新对准固定中心。
    4. 记录 Tool[0] 下六轴末端坐标系在基坐标系下的位姿：
          [X, Y, Z, RX, RY, RZ]

输入位姿：
    Tool[0] 下的六轴末端位姿 B_T_6，不是工具位姿。

数学原理：
    每组中心对准同一个固定点时：
        P_B = p6_B_j + R_B6_j @ [X, Y, Z_old]^T

    未知量：
        X, Y, P_X, P_Y, P_Z

    Z 保持旧值不变。
    多组样本用最小二乘求 X/Y。

注意：
    1. 如果采集时主动调整 RX/RY，本程序结果会混入姿态误差。
    2. 如果固定中心会动，结果会失效。
    3. 角度覆盖范围越大越好。
    4. 建议 6~12 组。
"""

import numpy as np
from calib_common import (
    FlangePoseSample,
    make_transform,
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

# None：不剔除。
# 例如采 10 个点，希望保留 8 个，则 KEEP_BEST_N=8。
KEEP_BEST_N = None

SAMPLES = [
    # 示例：
    # FlangePoseSample("RZ_0", [434.197, 699.939, 740.150, 179.995, 0.028, 45.001]),
]


# ============================================================
# 求解
# ============================================================

def solve(samples):
    if len(samples) < 3:
        raise ValueError("至少需要3组数据，建议6组以上。")

    tool_old = np.array(TOOL_OLD, dtype=float)
    z_old = tool_old[2]

    A_rows = []
    b_rows = []
    I = np.eye(3)

    for sample in samples:
        T_B6 = make_transform(sample.pose)
        R6 = T_B6[:3, :3]
        p6 = T_B6[:3, 3]

        R_xy = R6[:, 0:2]
        R_z = R6[:, 2]

        # P = p6 + R_xy @ [x, y] + R_z * z_old
        # R_xy @ [x, y] - P = -p6 - R_z * z_old
        A_rows.append(np.hstack([R_xy, -I]))
        b_rows.append(-p6 - R_z * z_old)

    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)

    sol, _, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

    tool_new = tool_old.copy()
    tool_new[0] = sol[0]
    tool_new[1] = sol[1]
    fixed_point_base = sol[2:5]

    rows = []
    residuals = []
    for i, sample in enumerate(samples):
        T_B6 = make_transform(sample.pose)
        R6 = T_B6[:3, :3]
        p6 = T_B6[:3, 3]

        P_hat = p6 + R6 @ np.array([tool_new[0], tool_new[1], z_old])
        res_vec = P_hat - fixed_point_base
        res = float(np.linalg.norm(res_vec))

        rows.append({
            "sample": sample,
            "residual_vector": res_vec,
            "residual_norm": res,
            "predicted_point_base": P_hat,
        })
        residuals.append(res)

    residuals = np.array(residuals, dtype=float)

    cond = float("inf")
    if len(singular_values) > 0 and singular_values[-1] > 1e-12:
        cond = float(singular_values[0] / singular_values[-1])

    return {
        "tool_old": tool_old,
        "tool_new": tool_new,
        "delta": tool_new - tool_old,
        "fixed_point_base": fixed_point_base,
        "rows": rows,
        "residuals": residuals,
        "rms": float(np.sqrt(np.mean(residuals ** 2))),
        "mean": float(np.mean(residuals)),
        "max": float(np.max(residuals)),
        "rank": int(rank),
        "condition_number": cond,
        "samples": samples,
    }


def residual_getter(result):
    return result["residuals"]


def print_report(result, kept_indices, removed):
    print("\n========== XY原点优化结果 ==========")
    print("旧工具定义:")
    print(format_pose(result["tool_old"]))
    print("\n新工具定义:")
    print(format_pose(result["tool_new"]))

    d = result["delta"]
    print("\n修改量:")
    print(f"Delta X = {d[0]:.6f} mm")
    print(f"Delta Y = {d[1]:.6f} mm")
    print(f"Delta Z = {d[2]:.6f} mm")
    print(f"Delta RX = {d[3]:.6f} deg")
    print(f"Delta RY = {d[4]:.6f} deg")
    print(f"Delta RZ = {d[5]:.6f} deg")

    print("\n残差:")
    print(f"RMS  = {result['rms']:.6f} mm")
    print(f"Mean = {result['mean']:.6f} mm")
    print(f"Max  = {result['max']:.6f} mm")
    print(f"条件数 = {result['condition_number']:.6f}")
    print("\n反算固定点基坐标:")
    p = result["fixed_point_base"]
    print(f"[{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}]")

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

    print("\n逐点残差:")
    for i, row in enumerate(result["rows"]):
        s = row["sample"]
        rv = row["residual_vector"]
        print(f"[{i}] {s.name}: {row['residual_norm']:.6f} mm | residual_vec = [{rv[0]:.6f}, {rv[1]:.6f}, {rv[2]:.6f}] | {format_pose(s.pose)}")


if __name__ == "__main__":
    if len(SAMPLES) < 3:
        print("至少需要3组中心对准数据，推荐6~12组。")
    else:
        result, kept, removed = iterative_trim(
            items=SAMPLES,
            solve_func=solve,
            residual_getter=residual_getter,
            keep_best_n=KEEP_BEST_N,
            min_keep_n=3,
        )
        print_report(result, kept, removed)

        print("\n使用建议:")
        print("1. 采集时不要主动调整RX/RY。")
        print("2. 如果条件数很大，说明角度分布太集中，应扩大RZ角度范围。")
        print("3. 写入新X/Y后重新采集数据验证，不要用旧数据反复迭代。")
