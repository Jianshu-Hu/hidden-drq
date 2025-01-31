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
folders_27 = ['drq', 'drq_hidden_256', 'drq+contra_loss_hidden_256',
              'drq_hidden_128', 'drq+contra_loss_hidden_128']

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
              'drq+zoom_in_11_contra_loss_hidden_256', 'drq+zoom_in_11_bilinear+contra_loss_hidden_256',
              'drq+zoom_in_12_bilinear+contra_loss_hidden_256',
              'drq+zoom_in_105_bilinear+contra_loss_hidden_256','drq+zoom_out_08+crop+contra_loss_hidden_256',
              'drq+crop_rotate+contra_loss_hidden_256', 'drq+crop+rotate+contra_loss_hidden_256']
folders_25 = ['drq', 'drq_hidden_dim_256', 'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+color_jitter_03+contra_loss_hidden_256',
              'drq+brightness_01+contra_loss_hidden_256', 'drq+brightness_01+hidden_256',
              'drq+hue_01+contra_loss_hidden_256', 'drq+hue_01+crop+contra_loss_hidden_256']
folders_26 = ['drq', 'drq_hidden_dim_256', 'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'remove_back_hidden_256',
              'drq+gaussian_blur_3_2_5+contra_loss_hidden_256','drq+gaussian_blur_3_2_5+crop+contra_loss_hidden_256']

folders_28 = ['drq', 'drq_hidden_dim_256', 'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+exp_div_max_contra_loss_hidden_256', 'drq+init_weight_005_no_scheduler_log_contra_loss_hidden_256',
              'drq+exp_minus_max_contra_loss_hidden_256',
              'drq+Q_diff_contra_loss_hidden_256']
folders_30 = ['drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256', 'drq+log_contra_loss_hidden_256',
              'drq+init_weight_015_log_contra_loss_hidden_256', 'drq+init_weight_005_log_contra_loss_hidden_256',
              'drq+without_scheduler_log_contra_loss_hidden_256',
              'drq+init_weight_005_no_scheduler_log_contra_loss_hidden_256']
folders_31 = ['drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+exp_minus_max_contra_loss_hidden_256', 'drq+init_weight_015_exp_minus_max_contra_loss_hidden_256',
              'drq+init_weight_005_exp_minus_max_contra_loss_hidden_256',
              'drq+init_weight_005_no_scheduler_exp_minus_max_contra_loss_hidden_256',
              'drq+without_scheduler_exp_minus_max_contra_loss_hidden_256']
folders_32 = ['drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+exp_div_max_contra_loss_hidden_256', 'drq+init_weight_015_exp_div_max_contra_loss_hidden_256',
              'drq+init_weight_005_exp_div_max_contra_loss_hidden_256']
folders_29 = ['drq', 'drq_hidden_dim_256', 'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+simclr_on_last_conv+hidden_256',
              'drq+simclr_on_second_to_last+contra_loss_encoder+hidden_256',
              'drq+simclr_on_second_to_last+simclr_on_last+hidden_256',
              'drq+proj_512+simclr_on_third_second_to_last+contra_loss_encoder+hidden_256',
              'drq+proj_512+simclr_on_third_second_to_last+simclr_on_last+hidden_256',
              'drq+init_weight_015_simclr_on_second_to_last+simclr_on_last+hidden_256',
              'drq+init_weight_005_simclr_on_second_to_last+simclr_on_last+hidden_256']
folders_34 = ['drq', 'drq_hidden_dim_256', 'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+log_q_regularized_similarity_init_weight_1_hidden_256',
              'drq+log_q_regularized_similarity_init_weight_05_hidden_256',
              'drq+log_q_regularized_similarity_init_weight_15_hidden_256',
              'drq+log_q_regularized_similarity_init_weight_1_hidden_128',
              'drq+last_conv_proj+log_q_regularized_similarity_init_weight_1_hidden_256',
              'drq+q_regularized_similarity_init_weight_1_hidden_256',
              'drq+q_regularized_similarity_init_weight_1_hidden_128']
folders_35 = ['drq', 'drq_hidden_dim_256', 'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+label_q_div_1_contra_loss_init_weight_01_hidden_256',
              'drq+label_q_div_05_contra_loss_init_weight_01_hidden_256',
              'drq+label_100_cluster_contra_loss_init_weight_01_hidden_256']

folders_36 = ['drq', 'drq_hidden_dim_256', 'drq_hidden_dim_128',
              'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_256',
              'drq+contra_loss_cosine_scheduler_init_weight_01_hidden_128',
              'drq+init_weight_005_simclr_on_second_to_last+simclr_on_last+hidden_256',
              'drq+init_weight_005_simclr_on_second_to_last+simclr_on_last+hidden_128',
              'drq+init_weight_005_simclr_on_second_to_last+simclr_on_last+conv_24_hidden_128']

folders_37 = ['drq', 'drq_new_last_conv_stride_2', 'drq_new_last_conv_stride_2+contra_loss',
              'drq_new_last_conv_stride_2+q_regularized_similarity_init_weight_1',
              'drq_new_last_conv_stride_2+log_q_regularized_similarity_init_weight_1',
              'drq_new_last_conv_stride_2+label_50_cluster_contra_loss_init_weight_01',
              'drq_new_last_conv_stride_2+label_25_cluster_contra_loss_init_weight_01',
              'drq_new_last_conv_stride_2+label_10_cluster_contra_loss_init_weight_01']
prefix_2_1 = 'saved_runs/cheetah_new/'
folders_39 = ['drq', 'drq+contra_loss',
              'drq+contra_loss+without_scheduler',
              'drq+label_10_cluster_contra_loss',
              'drq+label_25_cluster_contra_loss',
              'drq+label_25_cluster_contra_loss+without_scheduler',
              'drq+label_50_cluster_contra_loss']
folders_42 = ['drq',
              'drq+q_regularized_similarity_type_1',
              'drq+q_regularized_similarity_type_1+without_scheduler',
              'drq+q_regularized_similarity_type_1_max_so_far+without_scheduler',
              'drq+q_regularized_similarity_type_2',
              'drq+q_regularized_similarity_type_2+without_scheduler',
              'drq+q_regularized_similarity_type_3',
              'drq+q_regularized_similarity_type_3+without_scheduler']


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

prefix_3_1 = 'saved_runs/reacher_hard_new/'
folders_41 = ['drq', 'drq+contra_loss',
              'drq+label_25_cluster_contra_loss',
              'drq+q_regularized_similarity_type_1',
              'drq+q_regularized_similarity_type_2',
              'drq+q_regularized_similarity_type_3']

prefix_4 = 'saved_runs/walker_walk/'
folders_9 = ['drq', 'attention_regularization_all_layers', 'attention_regularization_last_layer']
folders_10 = ['drq', 'regularization_l2_weight_1']
folders_24 = ['drq', 'drq_hidden_256', 'drq_hidden_128',
              'drq+contra_loss_hidden_256', 'drq+contra_loss_hidden_128']
prefix_4_1 = 'saved_runs/walker_walk_new/'
folders_40 = ['drq', 'drq+contra_loss',
              'drq+label_25_cluster_contra_loss',
              'drq+q_regularized_similarity_type_1',
              'drq+q_regularized_similarity_type_1_beta',
              'drq+q_regularized_similarity_type_2',
              'drq+q_regularized_similarity_type_3',
              'drq+q_regularized_similarity_type_4',
              'drq+q_regularized_similarity_type_4_without_square',]

folders_43 = ['drq',
              'drq+q_regularized_similarity_type_2',
              'drq+q_regularization_8',
              'drq+q_regularization_8_decay_faster',
              'drq+q_regularization_8_with_tanh',
              'drq+q_regularization_9',
              'drq+q_regularization_12',
              'drq+q_regularization_14']

folders_44 = ['drq',
              'drq+q_regularized_similarity_type_2',
              'drq+q_regularization_10',
              'drq+q_regularization_11',
              'drq+q_regularization_11_decay_faster',
              'drq+q_regularization_11_with_tanh',
              'drq+q_regularization_13',
              'drq+q_regularization_15',
              'drq+q_regularization_15_decay_faster',
              'drq+q_regularization_16']

folders_45 = ['drq',
              'drq+q_regularized_similarity_type_2',
              'drq+q_regularization_20',
              'drq+q_regularization_21',
              'drq+q_regularization_22',
              'drq+q_regularization_23',
              'drq+q_regularization_20_max_update_slower',
              'drq+q_regularization_21_max_update_slower',
              'drq+q_regularization_22_max_update_slower',
              'drq+q_regularization_23_max_update_slower']

folders_46 = ['drq', 'drq_larger_diff', 'drq_larger_error',
              'drq_smaller_diff_target_larger_error_critic',
              'drq_larger_diff_target_larger_error_critic',
              'drq_smaller_error_target_larger_diff_critic',
              'drq_smaller_error_target_larger_error_critic',
              'drq_average_all_target_larger_diff_critic',
              'drq_3_aug_smaller_error_target_larger_diff_critic']

folders_38 = ['drq', 'drq_new_last_conv_stride_2',
              'drq_new_last_conv_stride_2+label_50_cluster_contra_loss_init_weight_01+original_aug']

folders_48 = ['drq', 'drq_rotation', 'drq_rotation+regularization_16']

prefix_7 = 'saved_runs/walker_run/'
folders_47 = ['drq', 'drq_regularization_16']

prefix_5 = 'saved_runs/pendulum/'
folders_11 = ['drq', 'attention_regularization_last_layer']

prefix_6 = 'saved_runs/hopper/'
folders_33 = ['drq_original', 'drq_original_with_cuda_deterministic',
              'drq_new_last_conv_stride_2_with_cuda_deterministic', 'drq', 'drq_hidden_256', 'drq_hidden_128',
              'drq+contra_loss_hidden_256',
              'drq+contra_loss_hidden_128']

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
# plot_several_folders(prefix=prefix_6, folders=folders_33, title='hopper_contra_loss')
#
# plot_several_folders(prefix=prefix_4, folders=folders_38, title='walker_walk_new_conv')

# 12.26
# plot_several_folders(prefix=prefix_2_1, folders=folders_39, title='cheetah_new_contra_loss')
# plot_several_folders(prefix=prefix_2_1, folders=folders_42, title='cheetah_new_Q_regularization_loss')
# plot_several_folders(prefix=prefix_4_1, folders=folders_40, title='walker_walk_new')
# plot_several_folders(prefix=prefix_3_1, folders=folders_41, title='reacher_hard_new')

# 1.9
plot_several_folders(prefix=prefix_4_1, folders=folders_43, title='walker_walk_Q_regularized_loss_1')
plot_several_folders(prefix=prefix_4_1, folders=folders_44, title='walker_walk_Q_regularized_loss_2')
plot_several_folders(prefix=prefix_4_1, folders=folders_45, title='walker_walk_Q_regularized_loss_3')

plot_several_folders(prefix=prefix_4_1, folders=folders_46, title='walker_walk_better_aug')

# 1.16
plot_several_folders(prefix=prefix_7, folders=folders_47, title='walker_run')
plot_several_folders(prefix=prefix_4_1, folders=folders_48, title='walker_walk_rotation')


