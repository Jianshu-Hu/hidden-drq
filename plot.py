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
    axs.legend()
    axs.set_title(title)
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_fig/'+title)


prefix_1 = 'saved_runs/cartpole/'
folders_1 = ['byol', 'drq', 'drq_with_attention', 'regularization']

prefix_2 = 'saved_runs/cheetah/'
folders_2 = ['drq', 'drq_attention', 'drq_attention_totally_refined']
folders_3 = ['drq', 'regularization', 'regularization_l2_weight_1',
             'regularization_l2_weight_05', 'regularization_l2_weight_15']
folders_4 = ['drq', 'regularization_l2_weight_1', 'byol', 'simclr',
             'regularization_l2_autoweight', 'byol_autoweight', 'simclr_autoweight']
folders_5 = ['drq', 'attention_regularization', 'attention_regularization_all_cnn_layers']
folders_8 = ['drq', 'regularization_l2_weight_1', 'regularization_l2_hidden_dim_512',
             'regularization_l2_averaged_target', 'regularization_contra_loss']

prefix_3 = 'saved_runs/reacher_hard/'
folders_6 = ['drq', 'drq_attention_regularization_all_layers', 'drq_attention_regularization_last_layer',
             'attention_regularization_last_layer_random_weight', 'attention_regularization_all_layers_random_weight',
             'attention_actor_critic_share_last_layer_random_weight',
             'attention_actor_critic_share_obs_spatial_weight_1_aug_after_spatial']
folders_12 = ['drq', 'attention_regularization_last_layer_random_weight_no_original_aug',
              'attention_spatial_obs_random_weight_no_original_aug',
              'attention_actor_critic_share_last_layer_random_weight_no_original_aug',
              'attention_actor_critic_share_obs_spatial_random_weight_no_original_aug']
folders_7 = ['drq', 'regularization_l2_weight_1', 'byol', 'simclr']

prefix_4 = 'saved_runs/walker_walk/'
folders_9 = ['drq', 'attention_regularization_all_layers', 'attention_regularization_last_layer']
folders_10 = ['drq', 'regularization_l2_weight_1']

prefix_5 = 'saved_runs/pendulum/'
folders_11 = ['drq', 'attention_regularization_last_layer']

# 11.7
# plot_several_folders(prefix=prefix_1, folders=folders_1, title='cartpole')
#
# plot_several_folders(prefix=prefix_2, folders=folders_2, title='cheetah_drq_attention')
# plot_several_folders(prefix=prefix_2, folders=folders_3, title='cheetah_normalization')
# plot_several_folders(prefix=prefix_2, folders=folders_4, title='cheetah_ssl')


# 11.14
plot_several_folders(prefix=prefix_2, folders=folders_5, title='cheetah_attention_regularization')

plot_several_folders(prefix=prefix_3, folders=folders_6, title='reacher_hard_attention_regularization')
plot_several_folders(prefix=prefix_3, folders=folders_12, title='reacher_hard_attention_no_aug')
plot_several_folders(prefix=prefix_3, folders=folders_7, title='reacher_hard_ssl')

plot_several_folders(prefix=prefix_4, folders=folders_9, title='walker_walk_attention_regularization')
plot_several_folders(prefix=prefix_4, folders=folders_10, title='walker_walk_regularization')

plot_several_folders(prefix=prefix_5, folders=folders_11, title='pendulum_attention_regularization')

# 11.21
plot_several_folders(prefix=prefix_2, folders=folders_8, title='cheetah_regularization')