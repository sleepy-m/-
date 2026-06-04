#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_tool_z_axis_direction_parallel_planes.py

功能：
    减少工具坐标系 Z 轴方向误差。
    也就是优化工具定义中的 RX / RY，使工具Z轴更接近实际标定靶轴线/端面法向。

本程序适用的测量方式：
    “多组面平行约束法”。

现场采集要求：
    每一组数据都要把：
        工具端标定靶 a 的接触面
        与固定标定靶 b 的接触面
    调到平行，或者四周间隙尽量一致。

    注意：
        只要求两个面平行；
        不要求两个圆心重合；
        不要求两个圆完全重合。

输入位姿：
    Tool[0] 下六轴末端坐标系在基坐标系下的位姿：
        [X, Y, Z, RX, RY, RZ]

输出：
    新工具定义：
        [X_old, Y_old, Z_old, RX_new, RY_new, RZ_old]
    默认保持 RZ 不变，只修改 RX / RY。

数学原理：
    每一组面平行时：
        B_R_6_j @ z_6 = n_B

    z_6 是工具Z轴在六轴末端坐标系下的方向；
    n_B 是固定标定靶b的面法向，在基坐标系下固定不变。

    因此多组数据中，B_R_6_j @ z_6 应尽量指向同一个方向。
    求 z_6 的优化问题为：
        max ||sum(B_R_6_j @ z_6)||²
    解为：
        M.T @ M 最大特征值对应的特征向量，M=sum(B_R_6_j)。

注意事项：
    1. 如果标定靶a端面本身不垂直于圆柱轴线，程序会把机械误差当作工具姿态误差。
    2. 如果气管/线束在某些RZ角度拉扯，应剔除这些角度。
    3. 建议采 5~10 组，RZ角度尽量分散。
    4. 写入新 RX/RY 后，需要重新验证，不能拿旧数据反复迭代。
"""

import numpy as np
from calib_common import (
    FlangePoseSample,
    pose_to_rot,
    rpy_to_rot,
    rot_to_rpy,
    solve_axis_from_parallel_constraints,
    solve_rx_ry_keep_rz_from_z_axis,
    rotate_vector_toward,
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

# None：不剔除。
# 例如采 8 个点，希望保留 6 个，则 KEEP_BEST_N = 6。
KEEP_BEST_N = None

# 建议先用 1.0。
# 如果数据不干净，也可以用 0.5 小步调整。
APPLY_GAIN = 1.0

SAMPLES = [
    # 示例：
    # FlangePoseSample("P1", [673.320541, 464.365070, 609.065887, -157.946968, -9.485592, -6.842088]),
    # FlangePoseSample("P2", [836.099017, 433.448597, 632.120229, -157.423040, 8.086322, -48.202005]),
]


# ============================================================
# 求解
# ============================================================

def solve(samples):
    flange_poses = [s.pose for s in samples]

    R_old = rpy_to_rot(TOOL_OLD[3], TOOL_OLD[4], TOOL_OLD[5])
    old_z_axis_6 = R_old[:, 2]

    result = solve_axis_from_parallel_constraints(
        flange_poses=flange_poses,
        old_axis_in_flange=old_z_axis_6,
    )

    solved_z = result["axis_6"]
    target_z = rotate_vector_toward(old_z_axis_6, solved_z, gain=APPLY_GAIN)

    # 保持旧 RZ 不变，只求新的 RX/RY
    new_rx, new_ry, new_rz = solve_rx_ry_keep_rz_from_z_axis(
        target_z,
        fixed_rz_deg=TOOL_OLD[5],
    )

    tool_new = np.array(TOOL_OLD, dtype=float)
    tool_new[3] = new_rx
    tool_new[4] = new_ry
    tool_new[5] = new_rz

    result["tool_new"] = tool_new
    result["tool_old"] = np.array(TOOL_OLD, dtype=float)
    result["applied_z_axis_6"] = target_z
    result["delta"] = tool_new - np.array(TOOL_OLD, dtype=float)
    result["samples"] = samples
    return result


def residual_getter(result):
    return result["residual_angles"]


def print_report(result, kept_indices, removed):
    print("\n========== Z轴方向优化结果 ==========")
    print("旧工具定义:")
    print(format_pose(result["tool_old"]))
    print("\n新工具定义:")
    print(format_pose(result["tool_new"]))

    d = result["delta"]
    print("\n修改量:")
    print(f"Delta RX = {d[3]:.6f} deg")
    print(f"Delta RY = {d[4]:.6f} deg")
    print(f"Delta RZ = {d[5]:.6f} deg")

    print("\n求得的工具Z轴方向，表达在六轴末端坐标系下:")
    z = result["axis_6"]
    print(f"[{z[0]:.9f}, {z[1]:.9f}, {z[2]:.9f}]")

    print("\n应用后的工具Z轴方向，表达在六轴末端坐标系下:")
    z2 = result["applied_z_axis_6"]
    print(f"[{z2[0]:.9f}, {z2[1]:.9f}, {z2[2]:.9f}]")

    print("\n固定面法向估计，表达在基坐标系下:")
    n = result["fixed_axis_base"]
    print(f"[{n[0]:.9f}, {n[1]:.9f}, {n[2]:.9f}]")

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
        print("至少需要3组面平行数据，推荐5~10组。")
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
        print("1. 如果残差角度很大，优先检查面平行判断、标定靶垂直度、气管拉扯。")
        print("2. 写入新RX/RY后，重新采集新数据验证，不能用旧数据继续迭代。")
        print("3. Z轴方向优化完成后，再运行XY原点优化程序。")
