import matplotlib.pyplot as plt
import numpy as np
import math
import os


def average_over_several_runs(folder):
    data_all = []
    min_length = np.inf
    runs = os.listdir(folder)
    for i in range(len(runs)):
        data = np.loadtxt(folder+'/'+runs[i]+'/eval.csv', delimiter=',', skiprows=1)
        evaluation_freq = data[-1, -1]-data[-2, -1]
        data_all.append(data[:, 1])
        if len(data) < min_length:
            min_length = len(data)
    average = np.zeros([len(runs), min_length])
    for i in range(len(runs)):
        average[i, :] = data_all[i][:min_length]

    mean = np.mean(average, axis=0)
    std = np.std(average, axis=0)

    return mean, std, evaluation_freq


def plot_several_folders(prefix, folders, label_list=[], plot_or_save='save', title=""):
    fig, axs = plt.subplots(1, 1)
    for i in range(len(folders)):
        num_runs = len(os.listdir(prefix+folders[i]))
        mean, std, eval_freq = average_over_several_runs(prefix+folders[i])
        # plot variance
        axs.fill_between(eval_freq*range(len(mean)), mean - std/math.sqrt(num_runs), mean + std/math.sqrt(num_runs),
                         alpha=0.4)
        if len(label_list) == len(folders):
            # specify label
            axs.plot(eval_freq*range(len(mean)), mean, label=label_list[i])
        else:
            axs.plot(eval_freq*range(len(mean)), mean, label=folders[i])

    axs.set_xlabel('evaluation steps')
    axs.set_ylabel('episode reward')
    axs.legend(fontsize=8)
    axs.set_title(title)
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_fig/'+title)


prefix_1 = 'saved_runs/cartpole/'

prefix_2 = 'saved_runs/cheetah_run/'
folders_5 = ['drq', 'drq+regularization', 'drq+rotation', 'drq+rotation+regularization',
             'drq+hflip', 'drq+hflip+regularization']
folders_7 = ['drq+batch_256', 'drq+rotation+batch_256', 'drq+cycnn+batch_256', 'drq+rotation+cycnn+batch_256']
folders_10 = ['drq', 'drq+aug_when_act', 'drq+image_pad_6+aug_when_act', 'drq+image_pad_8+aug_when_act']
folders_13 = ['drq', 'drq+rotation', 'drq+rotation_30', 'drq+rotation_30+aug_when_act',
              'drq+rotation_90+aug_when_act',
              'drq+rotation_180+aug_when_act']
folders_14 = ['drq', 'drq+aug_when_act',
              'drq+aug_when_act+regularization_only_Q_diff', 'drq+aug_when_act+regularization_only_l2_regu',]
folders_17 = ['drq', 'drq+aug_when_act', 'drq+aug_when_act_average_2', 'drq+aug_when_act_average_5',
              'rad', 'rad+aug_when_act']
folders_20 = ['drq+aug_when_act', 'randnet', 'randnet_rand_both', 'randnet_with_fm_loss']
folders_23 = ['sac+visualize', 'rad+visualize+deterministic', 'rad+aug_when_act+visualize', 'drq+visualize',
              'drq+aug_when_act+visualize',
              'drq+remove_small_crop+aug_when_act+visualize',
              'drq_3_aug+aug_when_act+visualize', 'drq+rotation_shift+visualize']
folders_24 = ['rad+visualize+deterministic',
              'rad+aug_when_evaluate+visualize+deterministic',
              'rad+aug_when_act+visualize+deterministic',
              'drq+visualize+deterministic',
              'drq+aug_when_evaluate+visualize+deterministic',
              'drq+aug_when_act+visualize+deterministic']
folders_25 = ['drq+visualize+deterministic',
              'drq+aug_when_act+visualize+deterministic',
              'drq+remove_01_00_crop+aug_when_act+visualize+deterministic',
              'drq+rotation_5_crop+aug_when_act+visualize+deterministic',
              'drq+alpha_06_crop+aug_when_act+visualize+deterministic',
              'drq+alpha_08_crop+aug_when_act+visualize+deterministic']

folders_28 = ['drq+visualize+deterministic', 'drq+kl_loss+crop+visualize+deterministic',
              'drq+actor_obs_aug_loss+crop+visualize+deterministic',
              'drq+kl_loss+crop+aug_when_act+visualize+deterministic',
              'drq+beta_kl_loss+crop+visualize+deterministic']

prefix_3 = 'saved_runs/reacher_hard/'
folders_1 = ['drq', 'drq+regularization', 'drq+rotation', 'drq+rotation+regularization',
             'drq+hflip', 'drq+hflip+regularization']
folders_3 = ['drq+batch_256', 'drq+rotation+batch_256', 'drq+cycnn+batch_256', 'drq+rotation+cycnn+batch_256',
             'drq+rotation_10+cycnn+batch_256', 'drq+rotation_90+cycnn+batch_256', 'drq+rotation_90+batch_256']
folders_9 = ['drq', 'drq+rotation', 'drq+rotation+black_background', 'drq+aug_when_act',
             'drq+aug_when_act+rotation', 'drq+rotation_90', 'drq+rotation_90+aug_when_act',
             'drq+rotation_180+aug_when_act']
folders_15 = ['drq', 'drq+rotation_180+aug_when_act', 'drq+rotation_180+aug_when_act+regularization_only_Q_diff',
              'drq+rotation_180+aug_when_act+regularization_only_l2_regu']
folders_16 = ['drq', 'drq+rotation_180', 'drq+rotation_180+aug_when_act', 'drq+rotation_15_180+aug_when_act',
              'drq+rotation_180+aug_when_act_average_2', 'drq+rotation_180+aug_when_act_average_5',
              'rad+rotation_180', 'rad+rotation_180+aug_when_act']
folders_21 = ['drq+rotation_180+aug_when_act', 'randnet', 'randnet_rand_both', 'randnet_with_fm_loss']

folders_27 = ['drq_crop+visualize+deterministic', 'drq_180_rotation+visualize+deterministic',
              'drq_180_rotation+aug_when_act+visualize+deterministic',
              'drq+15_180_rotation+aug_when_act+visualize+deterministic',
              'drq+alpha_06_180_rotation+aug_when_act+visualize+deterministic',
              'drq+alpha_08_180_rotation+aug_when_act+visualize+deterministic']

prefix_4 = 'saved_runs/walker_walk/'
prefix_5 = 'saved_runs/pendulum/'
prefix_6 = 'saved_runs/hopper/'

prefix_7 = 'saved_runs/walker_run/'
folders_2 = ['drq', 'drq+regularization', 'drq+rotation', 'drq+rotation+regularization',
             'drq+hflip', 'drq+hflip+regularization']
folders_4 = ['drq', 'drq+cycnn']
folders_11 = ['drq', 'drq+aug_when_act', 'drq+rotation_90+aug_when_act', 'drq+rotation_30+aug_when_act']
folders_19 = ['drq', 'drq+aug_when_act', 'rad', 'rad+aug_when_act']
folders_22 = ['sac+visualize', 'rad+visualize', 'rad+aug_when_act+visualize', 'drq+visualize',
              'drq+aug_when_act+visualize',
              'drq+remove_small_crop+aug_when_act+visualize',
              'drq_3_aug+aug_when_act+visualize', 'drq+rotation_shift+visualize']
folders_26 = ['drq+visualize+deterministic', 'drq+aug_when_evaluate+visualize+deterministic',
              'drq+aug_when_act+visualize+deterministic',
              'drq+remove_01_00_crop+aug_when_act+visualize+deterministic',
              'drq+rotation_5_crop+aug_when_act+visualize+deterministic',
              'drq+alpha_06_crop+aug_when_act+visualize+deterministic',
              'drq+alpha_08_crop+aug_when_act+visualize+deterministic']

prefix_8 = 'saved_runs/ballincup_catch/'
folders_6 = ['drq', 'drq+regularization', 'drq+rotation', 'drq+rotation+regularization',
             'drq+hflip', 'drq+hflip+regularization']
folders_8 = ['drq+batch_256', 'drq+rotation+batch_256', 'drq+cycnn+batch_256', 'drq+rotation+cycnn+batch_256']
folders_12 = ['drq', 'drq+aug_when_act', 'drq+rotation_90+aug_when_act', 'drq+rotation_30+aug_when_act']
folders_18 = ['drq', 'drq+aug_when_act', 'rad', 'rad+aug_when_act']


# 11.7
# plot_several_folders(prefix=prefix_1, folders=folders_1, title='cartpole')
#
# plot_several_folders(prefix=prefix_2, folders=folders_2, title='cheetah_drq_attention')
# plot_several_folders(prefix=prefix_2, folders=folders_3, title='cheetah_normalization')
# plot_several_folders(prefix=prefix_2, folders=folders_4, title='cheetah_ssl')


# 11.14
# plot_several_folders(prefix=prefix_2, folders=folders_5, title='cheetah_attention_regularization')
#
# plot_several_folders(prefix=prefix_3, folders=folders_6, title='reacher_hard_attention_regularization')
# plot_several_folders(prefix=prefix_3, folders=folders_12, title='reacher_hard_attention_no_aug')
# plot_several_folders(prefix=prefix_3, folders=folders_7, title='reacher_hard_ssl')
#
# plot_several_folders(prefix=prefix_4, folders=folders_9, title='walker_walk_attention_regularization')
# plot_several_folders(prefix=prefix_4, folders=folders_10, title='walker_walk_regularization')
#
# plot_several_folders(prefix=prefix_5, folders=folders_11, title='pendulum_attention_regularization')

# 11.21
# plot_several_folders(prefix=prefix_2, folders=folders_8, title='cheetah_regularization_l2')
# plot_several_folders(prefix=prefix_2, folders=folders_14, title='cheetah_regularization_contra_loss')
#
# plot_several_folders(prefix=prefix_3, folders=folders_13, title='reacher_regularization')
# plot_several_folders(prefix=prefix_3, folders=folders_17, title='reacher_contra_loss')

# 11.28
# plot_several_folders(prefix=prefix_2, folders=folders_15, title='cheetah_regularization_l1')
#
# plot_several_folders(prefix=prefix_2, folders=folders_18, title='cheetah_regularization_contra_loss_better')
# plot_several_folders(prefix=prefix_2, folders=folders_19, title='cheetah_regularization_contra_loss_auto')
# plot_several_folders(prefix=prefix_2, folders=folders_20, title='cheetah_regularization_contra_loss_scheduler')
# plot_several_folders(prefix=prefix_2, folders=folders_16, title='cheetah_regularization_hidden_512_best')
#
# plot_several_folders(prefix=prefix_3, folders=folders_21, title='reacher_hard_contra_loss_scheduler')

# 12.5
# plot_several_folders(prefix=prefix_2, folders=folders_22, title='cheetah_averaged_embedding')
# plot_several_folders(prefix=prefix_2, folders=folders_25, title='cheetah_color')
# plot_several_folders(prefix=prefix_2, folders=folders_26, title='cheetah_background')
#
# plot_several_folders(prefix=prefix_4, folders=folders_24, title='walker_walk_contra_loss')

# 12.12
# plot_several_folders(prefix=prefix_1, folders=folders_27, title='cartpole_contra_loss')
# plot_several_folders(prefix=prefix_2, folders=folders_23, title='cheetah_spatial')
# plot_several_folders(prefix=prefix_2, folders=folders_28, title='cheetah_Q_contrastive_loss')
# plot_several_folders(prefix=prefix_2, folders=folders_30, title='cheetah_Q_log_tune')
# plot_several_folders(prefix=prefix_2, folders=folders_31, title='cheetah_Q_exp_minus_max_tune')
# plot_several_folders(prefix=prefix_2, folders=folders_32, title='cheetah_Q_exp_div_max_tune')

# 12.19
# plot_several_folders(prefix=prefix_2, folders=folders_29, title='cheetah_multi_level')
# plot_several_folders(prefix=prefix_2, folders=folders_36, title='cheetah_multi_level_128')
# plot_several_folders(prefix=prefix_2, folders=folders_37, title='cheetah_new_conv')
# plot_several_folders(prefix=prefix_2, folders=folders_34, title='cheetah_Q_regularized_loss')
# plot_several_folders(prefix=prefix_2, folders=folders_35, title='cheetah_Q_label_contra_loss')
#
# plot_several_folders(prefix=prefix_6, folders=folders_33, title='hopper_contra_loss')s
#
# plot_several_folders(prefix=prefix_4, folders=folders_38, title='walker_walk_new_conv')

# 12.26
# plot_several_folders(prefix=prefix_2_1, folders=folders_39, title='cheetah_new_contra_loss')
# plot_several_folders(prefix=prefix_2_1, folders=folders_42, title='cheetah_new_Q_regularization_loss')
# plot_several_folders(prefix=prefix_4_1, folders=folders_40, title='walker_walk_new')
# plot_several_folders(prefix=prefix_3_1, folders=folders_41, title='reacher_hard_new')

# 1.9
# plot_several_folders(prefix=prefix_4_1, folders=folders_43, title='walker_walk_Q_regularized_loss_1')
# plot_several_folders(prefix=prefix_4_1, folders=folders_44, title='walker_walk_Q_regularized_loss_2')
# plot_several_folders(prefix=prefix_4_1, folders=folders_45, title='walker_walk_Q_regularized_loss_3')
#
# plot_several_folders(prefix=prefix_4_1, folders=folders_46, title='walker_walk_better_aug')

# 1.16
# plot_several_folders(prefix=prefix_7, folders=folders_47, title='walker_run')
# plot_several_folders(prefix=prefix_4_1, folders=folders_48, title='walker_walk_rotation')

# 1.30
# plot_several_folders(prefix=prefix_2, folders=folders_5, title='cheetah_run_aug_regu')
# plot_several_folders(prefix=prefix_2, folders=folders_7, title='cheetah_run_cycnn')
# plot_several_folders(prefix=prefix_3, folders=folders_1, title='reacher_hard_aug_regu')
# plot_several_folders(prefix=prefix_7, folders=folders_2, title='walker_run_aug_regu')
# plot_several_folders(prefix=prefix_8, folders=folders_6, title='ballincup_catch_aug_regu')

# 2.6
# plot_several_folders(prefix=prefix_8, folders=folders_8, title='ballincup_catch_cycnn')
# plot_several_folders(prefix=prefix_3, folders=folders_9, title='reacher_hard_rotation')
# plot_several_folders(prefix=prefix_3, folders=folders_3, title='reacher_hard_cycnn')
# plot_several_folders(prefix=prefix_2, folders=folders_10, title='cheetah_run_aug_when_act')
# plot_several_folders(prefix=prefix_7, folders=folders_11, title='walker_run_aug_when_act')
# plot_several_folders(prefix=prefix_8, folders=folders_12, title='ballincup_catch_aug_when_act')
# plot_several_folders(prefix=prefix_2, folders=folders_13, title='cheetah_run_rotation')
# plot_several_folders(prefix=prefix_2, folders=folders_14, title='cheetah_run_new_regu')
# plot_several_folders(prefix=prefix_3, folders=folders_15, title='reacher_hard_new_regu')

# 2.13
# plot_several_folders(prefix=prefix_3, folders=folders_16, title='reacher_hard_rad')
# plot_several_folders(prefix=prefix_2, folders=folders_17, title='cheetah_run_rad')
# plot_several_folders(prefix=prefix_8, folders=folders_18, title='ballincup_catch_rad')
# plot_several_folders(prefix=prefix_7, folders=folders_19, title='walker_run_rad')
#
# plot_several_folders(prefix=prefix_2, folders=folders_20, title='cheetah_run_randnet')
# plot_several_folders(prefix=prefix_3, folders=folders_21, title='reacher_hard_randnet')

# 2.20
# plot_several_folders(prefix=prefix_7, folders=folders_22, title='walker_run_visualize')
# plot_several_folders(prefix=prefix_2, folders=folders_23, title='cheetah_run_visualize')
# plot_several_folders(prefix=prefix_2, folders=folders_24, title='cheetah_run_deterministic_aug_when_act')
# 2.27

# 3.6
plot_several_folders(prefix=prefix_2, folders=folders_25, title='cheetah_run_deterministic_different_aug')
plot_several_folders(prefix=prefix_7, folders=folders_26, title='walker_run_deterministic')
plot_several_folders(prefix=prefix_3, folders=folders_27, title='reacher_hard_deterministic')
plot_several_folders(prefix=prefix_2, folders=folders_28, title='cheetah_run_deterministic_kl')

# import torch
# alpha = 0.8
# beta = torch.distributions.Beta(alpha, alpha)
# num = 9
# fig, axs = plt.subplots(1, 1)
# y = np.zeros(num)
# for i in range(100000):
#     x = int(beta.sample().item()*num)
#     y[x] += 1
# axs.plot(range(num), y)
# plt.show()
