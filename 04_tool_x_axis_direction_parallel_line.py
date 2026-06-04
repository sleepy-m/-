#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_tool_x_axis_direction_parallel_line.py

功能：
    减少工具坐标系 X 轴方向误差。
    工具Z轴方向确定后，本程序用于优化工具绕Z轴的方向，也就是 X/Y 轴的朝向。

重要前提：
    必须有一个“非圆对称”的工具特征来代表工具X方向，例如：
        - 刻线
        - 平面/平边
        - 两个销孔形成的方向
        - 槽
        - 十字线中的X方向
        - 偏心标记

    如果你的标定靶是完全圆柱/圆盘，且没有任何方向特征，则 X轴方向不可观测。
    这种情况下，本程序不能使用，也没有必要修 X 轴方向。

现场采集要求：
    每组数据都要让工具端的 X方向特征 与 外部固定线/边/槽 平行。
    不要求位置重合，只要求方向平行。

输入位姿：
    Tool[0] 下六轴末端坐标系在基坐标系下的位姿：
        [X, Y, Z, RX, RY, RZ]

数学原理：
    每组方向平行时：
        B_R_6_j @ x_6 = q_B

    x_6 是工具X轴在六轴末端坐标系下的方向；
    q_B 是固定线方向在基坐标系下的方向，未知但固定。

    多组数据中，B_R_6_j @ x_6 应尽量一致。
    求解方式与 Z轴方向一致：
        max ||sum(B_R_6_j @ x_6)||²

输出：
    新工具姿态。
    默认保持当前工具Z轴方向不变，只调整X轴在垂直于Z的平面内的方向。
"""

import numpy as np
from calib_common import (
    FlangePoseSample,
    pose_to_rot,
    rpy_to_rot,
    rot_to_rpy,
    solve_axis_from_parallel_constraints,
    build_rotation_from_x_and_z,
    iterative_trim,
    format_pose,
)


# ============================================================
# 输入区
# ============================================================

TOOL_OLD = [
    142.029, 132.701, 272.828,
    0.000, 0.000, 135.000
]

KEEP_BEST_N = None

SAMPLES = [
    # FlangePoseSample("P1", [X, Y, Z, RX, RY, RZ]),
]


# ============================================================
# 求解
# ============================================================

def solve(samples):
    flange_poses = [s.pose for s in samples]

    R_old = rpy_to_rot(TOOL_OLD[3], TOOL_OLD[4], TOOL_OLD[5])
    old_x_axis_6 = R_old[:, 0]
    old_z_axis_6 = R_old[:, 2]

    result = solve_axis_from_parallel_constraints(
        flange_poses=flange_poses,
        old_axis_in_flange=old_x_axis_6,
    )

    solved_x = result["axis_6"]

    # 保持Z轴方向不变，把X轴投影到垂直于Z轴的平面上，构造正交工具坐标系。
    R_new = build_rotation_from_x_and_z(solved_x, old_z_axis_6)
    new_rx, new_ry, new_rz = rot_to_rpy(R_new)

    tool_new = np.array(TOOL_OLD, dtype=float)
    tool_new[3] = new_rx
    tool_new[4] = new_ry
    tool_new[5] = new_rz

    result["tool_old"] = np.array(TOOL_OLD, dtype=float)
    result["tool_new"] = tool_new
    result["delta"] = tool_new - np.array(TOOL_OLD, dtype=float)
    result["samples"] = samples
    return result


def residual_getter(result):
    return result["residual_angles"]


def print_report(result, kept_indices, removed):
    print("\n========== X轴方向优化结果 ==========")
    print("旧工具定义:")
    print(format_pose(result["tool_old"]))
    print("\n新工具定义:")
    print(format_pose(result["tool_new"]))

    d = result["delta"]
    print("\n修改量:")
    print(f"Delta RX = {d[3]:.6f} deg")
    print(f"Delta RY = {d[4]:.6f} deg")
    print(f"Delta RZ = {d[5]:.6f} deg")

    print("\n求得的工具X轴方向，表达在六轴末端坐标系下:")
    x = result["axis_6"]
    print(f"[{x[0]:.9f}, {x[1]:.9f}, {x[2]:.9f}]")

    print("\n固定线方向估计，表达在基坐标系下:")
    q = result["fixed_axis_base"]
    print(f"[{q[0]:.9f}, {q[1]:.9f}, {q[2]:.9f}]")

    print("\n残差角度:")
    print(f"RMS  = {result['rms_angle']:.6f} deg")
    print(f"Mean = {result['mean_angle']:.6f} deg")
    print(f"Max  = {result['max_angle']:.6f} deg")

    print("\n保留原始点序号:")
    print(kept_indices)

    if removed:
        print("\n========== 异常点剔除过程 ==========")
        for i, r in enumerate(removed, 1):
            item = r["removed_item"]
            print(f"\n第 {i} 次剔除:")
            print(f"  删除原始序号: {r['removed_original_index']}")
            print(f"  删除点名称: {item.name}")
            print(f"  删除点残差: {r['removed_residual']:.6f} deg")
            print(f"  删除位姿: {format_pose(item.pose)}")
            print(f"  删除前 RMS/Mean/Max = {r['before_rms']:.6f}, {r['before_mean']:.6f}, {r['before_max']:.6f}")
            print(f"  删除后 RMS/Mean/Max = {r['after_rms']:.6f}, {r['after_mean']:.6f}, {r['after_max']:.6f}")
    else:
        print("\n未剔除点。")

    print("\n逐点残差:")
    for local_i, s in enumerate(result["samples"]):
        print(f"[{local_i}] {s.name}: {result['residual_angles'][local_i]:.6f} deg | {format_pose(s.pose)}")


if __name__ == "__main__":
    if len(SAMPLES) < 3:
        print("至少需要3组X方向平行数据，推荐5~10组。")
        print("注意：完全圆形/圆柱标定靶没有X方向特征，本程序无法使用。")
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
        print("1. X轴方向必须由刻线、平边、槽、两个销孔等非圆对称特征定义。")
        print("2. 没有方向特征时，X轴方向不可观测。")
        print("3. 先完成Z轴方向优化，再做X轴方向优化。")
