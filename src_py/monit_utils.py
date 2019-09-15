from plot_utils import plot_one_TH1D, plot_two_TH1D
import os
import numpy as np


def is_nan(x):
    return (x is np.nan or x != x)


def monit_plots(filedir, args, event, w_a, w_b):
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    if args.PLOT_FEATURES == "FILTER":
        filt = [x == 1 for x in event.cols[:, -1]]
    else:
        filt = [not is_nan(x) for x in event.cols[:, -1]]
    w_a = w_a[filt]
    w_b = w_b[filt]

    for i in range(len(event.cols[0, :]) - 1):
        plot_two_TH1D(event.cols[:, i], filedir, filename=event.labels[i], w_a=w_a, w_b=w_b, filt=filt)

    def plot_aco_angle(y1_index, y2_index, column_id, filename):
        y1y2_pos = np.array(event.cols[:, y1_index][filt] * event.cols[:, y2_index][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        plot_two_TH1D(event.cols[:, column_id], filedir, filename=filename + "_pos", w_a=w_a * y1y2_pos,
                      w_b=w_b * y1y2_pos, filt=filt)
        plot_two_TH1D(event.cols[:, column_id], filedir, filename=filename + "_neg", w_a=w_a * y1y2_neg,
                      w_b=w_b * y1y2_neg, filt=filt)

    if args.FEAT == "Variant-1.1" and args.TYPE == "nn_rhorho":  # acoangle depending on y1y2 sign
        for y1_index, y2_index, column_id, filename in [[-3, -2, -4, "aco_angle_y1y2"]]:
            plot_aco_angle(y1_index, y2_index, column_id, filename)

    if args.FEAT == "Variant-1.1" and args.TYPE == "nn_a1rho":  # acoangle depending on y1y2 sign
        for y1_index, y2_index, column_id, filename in [[-9, -8, -11, "aco_angle_1_y1y2"],
                                                        [-8, -7, -10, "aco_angle_2_y1y2"],
                                                        [-4, -3, -6, "aco_angle_3_y1y2"],
                                                        [-3, -2, -5, "aco_angle_4_y1y2"]]:
            plot_aco_angle(y1_index, y2_index, column_id, filename)


    if args.FEAT == "Variant-1.1" and args.TYPE == "nn_a1a1":  # acoangle depending on y1y2 sign
        for y1_index, y2_index, column_id, filename in [[-45, -44, -49, "aco_angle_1_y1y"],
                                                        [-43, -42, -48, "aco_angle_2_y1y"],
                                                        [-41, -40, -47, "aco_angle_3_y1y"],
                                                        [-39, -38, -46, "aco_angle_4_y1y"],
                                                        [-33, -32, -37, "aco_angle_5_y1y"],
                                                        [-31, -30, -36, "aco_angle_6_y1y"],
                                                        [-29, -28, -35, "aco_angle_7_y1y"],
                                                        [-27, -26, -34, "aco_angle_8_y1y"],
                                                        [-21, -20, -25, "aco_angle_9_y1y"],
                                                        [-19, -18, -24, "aco_angle_10_y1y"],
                                                        [-17, -16, -23, "aco_angle_11_y1y"],
                                                        [-15, -14, -22, "aco_angle_12_y1y"],
                                                        [-9, -8, -13, "aco_angle_13_y1y"],
                                                        [-7, -6, -12, "aco_angle_14_y1y"],
                                                        [-5, -4, -11, "aco_angle_15_y1y"],
                                                        [-3, -2, -10, "aco_angle_16_y1y"]]:
            plot_aco_angle(y1_index, y2_index, column_id, filename)


    if args.FEAT == "Variant-4.1":
        for i in range(len(event.labels_suppl)):
            plot_one_TH1D(event.cols_suppl[:, i], filedir, filename=event.labels_suppl[i], w=w_a, filt=filt)
