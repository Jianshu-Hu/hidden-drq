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
folders_1 = ['byol', 'drq', 'drq_with_attention', 'regularization']

prefix_2 = 'saved_runs/cheetah/'
folders_2 = ['drq', 'drq_attention', 'drq_attention_totally_refined']
folders_3 = ['drq', 'regularization', 'regularization_l2_weight_1',
             'regularization_l2_weight_05', 'regularization_l2_weight_15']
folders_4 = ['drq', 'regularization_l2_weight_1', 'byol', 'simclr',
             'regularization_l2_autoweight', 'byol_autoweight', 'simclr_autoweight']
folders_5 = ['drq', 'attention_regularization', 'attention_regularization_all_cnn_layers']
folders_8 = ['drq', 'regularization_l2_weight_1', 'l2_weight_1_averaged_target',
             'regularization_l2_weight_05',
             'regularization_l2_hidden_dim_512',
             'regularization_l2_weight_05_hidden_dim_512',
             'drq+regularization_l2_weight_1_hidden_512']
folders_14 = ['drq', 'contra_loss_weight_01', 'contra_loss_weight_01_hidden_512',
              'contra_loss_weight_005_hidden_512',
              'contra_loss_weight_002_hidden_512', 'contra_loss_weight_001_hidden_512',
              'contra_loss_weight_002',
              'drq+contra_loss_weight_002', 'drq+contra_loss_weight_002_hidden_512',
              'drq+contra_loss_weight_005_hidden_512']
folders_15 = ['drq', 'smooth_l1_weight_1', 'smooth_l1_weight_1_averaged_target',
              'smooth_l1_weight_1_averaged_target_hidden_512',
              'smooth_l1_weight_05_averaged_target_hidden_512',
              'drq+smooth_l1_weight_05_averaged_target_hidden_512']
folders_16 = ['drq', 'drq_hidden_dim_512', 'drq+regularization_l2_weight_1_hidden_512',
              'drq+contra_loss_weight_002_hidden_512',
              'drq+smooth_l1_weight_05_averaged_target_hidden_512']
folders_18 = ['drq_hidden_dim_512', 'drq_hidden_dim_256', 'drq_hidden_dim_128',
              'drq+contra_loss_weight_002_hidden_512', 'drq+contra_loss_weight_002_hidden_256',
              'drq+contra_loss_weight_002_hidden_128',
              'averaged_embedding+contra_loss_weight_002_hidden_512'
              ]
folders_19 = ['drq_hidden_dim_256', 'drq+contra_loss_weight_002_hidden_256',
              'drq+contra_loss_auto_init_weight_01_target_rl_1_hidden_256',
              'drq+contra_loss_auto_init_weight_1_target_rl_10_hidden_256',
              'drq+contra_loss_auto_init_weight_1_target_rl_36_hidden_256',
              'drq+contra_loss_auto_weight_1_target_36_minus_target_hidden_256']
folders_20 = ['drq', 'drq_hidden_dim_128', 'drq_hidden_dim_256', 'drq_hidden_dim_512',
              'drq+contra_loss_weight_002_hidden_256',
              'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_128',
              'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_512']
folders_22 = ['drq', 'drq_hidden_dim_256', 'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'averaged_embedding_for_target_Q+contra_loss_hidden_256',
              'averaged_embedding_for_Q_target_Q+contra_loss_hidden_256']
folders_23 = ['drq', 'drq_hidden_dim_256', 'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+zoom_in_11_contra_loss_hidden_256','drq+crop_rotate+contra_loss_hidden_256']

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
folders_13 = ['drq', 'regularization_l2_weight_1',
              'smooth_l1_weight_1_averaged_target', 'smooth_l1_weight_1_averaged_target_hidden_512',
              'contra_loss_weight_01', 'contra_loss_weight_01_hidden_512']
folders_17 = ['drq', 'drq_hidden_512', 'contra_loss_weight_01', 'contra_loss_weight_01_hidden_512',
              'contra_loss_weight_005_hidden_512',
              'contra_loss_weight_002_hidden_512', 'drq+contra_loss_weight_002_hidden_512',
              'drq+contra_loss_weight_005_hidden_512',
              'drq+contra_loss_weight_01_hidden_512']
folders_21 = ['drq', 'drq_hidden_512', 'drq_hidden_256', 'drq_hidden_128', 'drq+contra_loss_weight_01_hidden_512',
              'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_512',
              'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_128']

prefix_4 = 'saved_runs/walker_walk/'
folders_9 = ['drq', 'attention_regularization_all_layers', 'attention_regularization_last_layer']
folders_10 = ['drq', 'regularization_l2_weight_1']
folders_24 = ['drq', 'drq_hidden_256', 'drq_hidden_128',
              'drq+contra_loss_hidden_256', 'drq+contra_loss_hidden_128']

prefix_5 = 'saved_runs/pendulum/'
folders_11 = ['drq', 'attention_regularization_last_layer']

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
plot_several_folders(prefix=prefix_2, folders=folders_22, title='cheetah_averaged_embedding')
plot_several_folders(prefix=prefix_2, folders=folders_23, title='cheetah_transformations')

plot_several_folders(prefix=prefix_4, folders=folders_24, title='walker_walk_contra_loss')

