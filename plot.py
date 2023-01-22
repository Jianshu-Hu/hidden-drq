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


def plot_several_folders(prefix, folders, num_runs=3, label_list=[], plot_or_save='save', title=""):
    fig, axs = plt.subplots(1, 1)
    for i in range(len(folders)):
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

prefix_3 = 'saved_runs/reacher_hard/'
folders_1 = ['drq', 'drq+regularization', 'drq+rotation', 'drq+rotation+regularization',
             'drq+hflip', 'drq+hflip+regularization']
folders_3 = ['drq+batch_256', 'drq+cycnn+batch_256', 'drq+rotation+cycnn+batch_256']

prefix_4 = 'saved_runs/walker_walk/'
prefix_5 = 'saved_runs/pendulum/'
prefix_6 = 'saved_runs/hopper/'

prefix_7 = 'saved_runs/walker_run/'
folders_2 = ['drq', 'drq+regularization', 'drq+rotation', 'drq+rotation+regularization',
             'drq+hflip', 'drq+hflip+regularization']
folders_4 = ['drq', 'drq+cycnn']


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

# 2.
plot_several_folders(prefix=prefix_2, folders=folders_5, title='cheetah_run_aug_regu')
plot_several_folders(prefix=prefix_3, folders=folders_1, title='reacher_hard_aug_regu')
plot_several_folders(prefix=prefix_3, folders=folders_3, title='reacher_hard_cycnn')
plot_several_folders(prefix=prefix_7, folders=folders_2, title='walker_run_aug_regu')
# plot_several_folders(prefix=prefix_7, folders=folders_4, title='walker_run_cycnn')
