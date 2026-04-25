# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def calculate_feeding_efficiency(T1_data, T2_data, T3_data, weights, T_ideal):
#     """
#     计算喂食效率量化指标η
#
#     参数：
#     T1_data : list - T1阶段时间数据列表
#     T2_data : list - T2阶段时间数据列表
#     T3_data : list - T3阶段时间数据列表
#     weights : list - 各阶段权重系数 [w1, w2, w3]
#     T_ideal : float - 理想总时间
#
#     返回：
#     eta : float - 喂食效率指标
#     report : dict - 包含所有中间计算值的报告
#     """
#     # 计算统计量
#     E_T = [np.mean(T1_data), np.mean(T2_data), np.mean(T3_data)]
#     sigma_T = [np.std(T1_data, ddof=1), np.std(T2_data, ddof=1), np.std(T3_data, ddof=1)]
#
#     # 计算公式组成部分
#     weighted_sum = np.dot(weights, E_T)
#     variance_sum = np.sqrt(sum(s ** 2 for s in sigma_T))
#
#     # 计算延迟时间因子
#     T_total = sum(E_T)
#     T_delay = max(0, T_total - T_ideal)
#     delay_factor = 1 - (T_delay / T_total) if T_total != 0 else 1.0
#
#     # 最终效率计算
#     eta = (weighted_sum / variance_sum) * delay_factor if variance_sum != 0 else 0
#
#     # 生成报告
#     report = {
#         "阶段期望值": E_T,
#         "阶段标准差": sigma_T,
#         "加权和": weighted_sum,
#         "方差和根": variance_sum,
#         "总时间": T_total,
#         "延迟时间": T_delay,
#         "延迟因子": delay_factor,
#         "效率指标": eta
#     }
#
#     return eta, report
#
#
# def visualize_efficiency(report):
#     """可视化分析结果"""
#     plt.figure(figsize=(12, 6))
#
#     # 阶段时间分布
#     plt.subplot(2, 2, 1)
#     plt.bar(['T1', 'T2', 'T3'], report["阶段期望值"],
#             yerr=report["阶段标准差"],
#             capsize=5,
#             color=['#2c7bb6', '#abd9e9', '#fdae61'])
#     plt.title("各阶段时间分布")
#     plt.ylabel("时间 (秒)")
#
#     # 权重分布
#     plt.subplot(2, 2, 2)
#     plt.pie(report["阶段标准差"],
#             labels=['T1', 'T2', 'T3'],
#             autopct='%1.1f%%',
#             colors=['#2c7bb6', '#abd9e9', '#fdae61'])
#     plt.title("阶段时间波动比例")
#
#     # 效率组成
#     plt.subplot(2, 2, 3)
#     components = [report["加权和"], report["方差和根"], report["延迟因子"]]
#     plt.barh(['加权和', '方差因子', '延迟补偿'], components, color='#d7191c')
#     plt.title("效率组成成分")
#     plt.xlabel("量值")
#
#     # 综合指标
#     plt.subplot(2, 2, 4)
#     plt.text(0.5, 0.5, f"η = {report['效率指标']:.2f}",
#              ha='center', va='center', fontsize=20)
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# # 示例数据
# np.random.seed(2023)
#
# # 生成模拟实验数据
# # T1_data = [21.60338593,21.64812422,19.37664914,19.38540936,19.39731598,18.5230341,18.53964734,18.3256433,18.31448913,18.31448913]
# # T2_data = [4.853384018,3.37183380,13.282758713,3.226257324,3.042696476,3.14252758,3.091706276,3.188364267,3.170049191,3.170049191]
# # T3_data = [4.244272947,4.347729921,4.29765749,4.220822573,4.339526892,4.443263292,4.358489275,4.489454508,4.453449488,4.453449488]
#
# T1_data = [14.1534276,14.84270144,13.62448692,12.64796782,11.83676815,12.63536787,12.63987994,11.83878922,14.84015131,14.84015131]
# T2_data = [2.628612518,2.359477758,2.199246883,2.035941362,2.21801281,2.218846083,2.06518364,2.665913582,2.66838479,2.66838479]
# T3_data = [3.314830303,3.130998373,2.971636057,2.794467926,2.998842239,2.992508173,2.850549221,3.365360022,3.375089407,3.375089407]
# # 参数设置
# weights = [0.35, 0.45, 0.20]  # 来自AHP分析的权重
# T_ideal = 15.2  # 系统最小理论时间
#
# # 计算效率指标
# eta, report = calculate_feeding_efficiency(T1_data, T2_data, T3_data, weights, T_ideal)
#
# # 打印详细报告
# print("Experimental Data Analysis Report：")
# print(
#     f"Phase Mean Values: T1={report['Phase Mean Values'][0]:.2f}s, T2={report['Phase Mean Values'][1]:.2f}s, T3={report['Phase Mean Values'][2]:.2f}s")
# print(
#     f"Phase Standard Deviations: T1±{report['Phase Standard Deviations'][0]:.2f}, T2±{report['Phase Standard Deviations'][1]:.2f}, T3±{report['Phase Standard Deviations'][2]:.2f}")
# print(f"Weighted Sum = {report['Weighted Sum']:.2f}")
# print(f"Root of Variance Sum = {report['Root of Variance Sum']:.2f}")
# print(f"Total Duration = {report['Total Duration']:.2f}s (Desired value={T_ideal}s)")
# print(f"Delay Compensation Factor = {report['Delay Compensation Factor']:.3f}")
# print(f"\nFinal Feeding Efficiency Index η = {eta:.2f}")
#
# # 可视化分析
# visualize_efficiency(report)
import numpy as np
import matplotlib.pyplot as plt


def calculate_feeding_efficiency(T1_data, T2_data, T3_data, weights, T_ideal):
    """
    Calculate feeding efficiency metric η

    Parameters:
    T1_data : list - Time data for pickup phase
    T2_data : list - Time data for feeding phase
    T3_data : list - Time data for return phase
    weights : list - Phase weights [w1, w2, w3]
    T_ideal : float - Theoretical minimum cycle time

    Returns:
    eta : float - Efficiency metric
    report : dict - Detailed computation report
    """
    # Calculate statistics
    phase_means = [np.mean(T1_data), np.mean(T2_data), np.mean(T3_data)]
    phase_stds = [np.std(T1_data, ddof=1), np.std(T2_data, ddof=1), np.std(T3_data, ddof=1)]

    # Formula components
    weighted_sum = np.dot(weights, phase_means)
    variance_root = np.sqrt(sum(s ** 2 for s in phase_stds))

    # Delay compensation
    total_time = sum(phase_means)
    T_delay = max(0, total_time - T_ideal)
    delay_factor = 1 - (T_delay / total_time) if total_time != 0 else 1.0

    # Final calculation
    eta = (weighted_sum / variance_root) * delay_factor if variance_root != 0 else 0

    # Generate report
    report = {
        "phase_means": phase_means,
        "phase_stds": phase_stds,
        "weighted_sum": weighted_sum,
        "variance_root": variance_root,
        "total_time": total_time,
        "T_delay": T_delay,
        "delay_factor": delay_factor,
        "efficiency_metric": eta
    }

    return eta, report


def visualize_efficiency(report):
    """Visualization of efficiency analysis"""
    plt.figure(figsize=(12, 6))

    # Phase time distribution
    plt.subplot(2, 2, 1)
    plt.bar(['T1', 'T2', 'T3'], report["phase_means"],
            yerr=report["phase_stds"],
            capsize=5,
            color=['#2c7bb6', '#abd9e9', '#fdae61'])
    plt.title("Phase Time Distribution")
    plt.ylabel("Time (seconds)")

    # Variance composition
    plt.subplot(2, 2, 2)
    plt.pie(report["phase_stds"],
            labels=['T1', 'T2', 'T3'],
            autopct='%1.1f%%',
            colors=['#2c7bb6', '#abd9e9', '#fdae61'])
    plt.title("Temporal Variance Composition")

    # Efficiency components
    plt.subplot(2, 2, 3)
    components = [report["weighted_sum"], report["variance_root"], report["delay_factor"]]
    plt.barh(['Weighted Sum', 'Variance Factor', 'Delay Compensation'], components, color='#d7191c')
    plt.title("Efficiency Components")
    plt.xlabel("Metric Value")

    # Composite metric
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, f"η = {report['efficiency_metric']:.2f}",
             ha='center', va='center', fontsize=20)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Sample data
np.random.seed(2023)

# Generate simulated data
T1_data = [21.60338593,21.64812422,19.37664914,19.38540936,19.39731598,18.5230341,18.53964734,18.3256433,18.31448913,18.31448913]
T2_data = [4.853384018,3.37183380,13.282758713,3.226257324,3.042696476,3.14252758,3.091706276,3.188364267,3.170049191,3.170049191]
T3_data = [4.244272947,4.347729921,4.29765749,4.220822573,4.339526892,4.443263292,4.358489275,4.489454508,4.453449488,4.453449488]
#
# T1_data = [14.1534276,14.84270144,13.62448692,12.64796782,11.83676815,12.63536787,12.63987994,11.83878922,14.84015131,14.84015131]
# T2_data = [2.628612518,2.359477758,2.199246883,2.035941362,2.21801281,2.218846083,2.06518364,2.665913582,2.66838479,2.66838479]
# T3_data = [3.314830303,3.130998373,2.971636057,2.794467926,2.998842239,2.992508173,2.850549221,3.365360022,3.375089407,3.375089407]

# Parameters
weights = [0.35, 0.45, 0.20]  # AHP-derived weights
T_ideal = 15.2  # System theoretical minimum

# Compute efficiency
eta, report = calculate_feeding_efficiency(T1_data, T2_data, T3_data, weights, T_ideal)

# Print report
print("Experimental Data Analysis Report:")
print(
    f"Phase Means: T1={report['phase_means'][0]:.2f}s, T2={report['phase_means'][1]:.2f}s, T3={report['phase_means'][2]:.2f}s")
print(
    f"Phase Stds: T1±{report['phase_stds'][0]:.2f}, T2±{report['phase_stds'][1]:.2f}, T3±{report['phase_stds'][2]:.2f}")
print(f"Weighted Sum = {report['weighted_sum']:.2f}")
print(f"Variance Root = {report['variance_root']:.2f}")
print(f"Total Cycle Time = {report['total_time']:.2f}s (Ideal={T_ideal}s)")
print(f"Delay Compensation Factor = {report['delay_factor']:.3f}")
print(f"\nFinal Feeding Efficiency Metric η = {eta:.2f}")

# Visualization
visualize_efficiency(report)