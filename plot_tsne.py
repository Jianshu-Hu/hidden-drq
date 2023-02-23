from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import os
import math


def compare_pairs(file):
    data = np.load(file)
    Y = data['Y']
    batch_size = 512
    # calculate percentage of close samples
    distance = np.sqrt(np.sum((Y[:batch_size]-Y[batch_size:])**2, axis=-1))
    percentage = np.sum((distance < 2))/batch_size
    # print('percentage of pairs whose distance is smaller than 1: '+str(percentage))
    target_Q = data['target_Q']
    target_Q_aug = data['target_Q_aug']
    relative_diff = np.abs(target_Q-target_Q_aug)/np.maximum(target_Q, target_Q_aug)
    average_diff = np.mean(relative_diff)
    return percentage, average_diff


def plot_target_Q(regularization, step, prefix):
    fig, axs = plt.subplots(1, 1)
    data = np.load('../outputs/'+prefix + str(regularization) + '_tsne-' + str(step) + '.npz')
    target_Q = data['target_Q']
    Y = data['Y']
    batch_size = 512
    plot_sample_num = 200
    plot_idxs = np.random.randint(0, batch_size, size=plot_sample_num)
    color = target_Q[plot_idxs]
    axs.scatter(Y[plot_idxs, 0], Y[plot_idxs, 1], c=color, cmap='Oranges')
    axs.scatter(Y[batch_size + plot_idxs, 0], Y[batch_size + plot_idxs, 1], c=color, cmap='Oranges')
    axs.set_title('tsne-step'+str(step*2))
    plt.savefig('../outputs/' + prefix + str(regularization) + '_tsne-' + str(step) + '_target_Q.png')


def average_the_data(data):
    shortest_len = np.inf
    for i in range(len(data)):
        if len(data[i]) < shortest_len:
            shortest_len = len(data[i])
    all_data = np.zeros([len(data), shortest_len])
    for i in range(len(data)):
        all_data[i] = data[i][:shortest_len]
    mean = np.average(all_data, axis=0)
    std = np.std(all_data, axis=0)
    return mean, std


def plot_percentage(domain, prefix_list, title):
    plt.rcParams["figure.figsize"] = (8, 8)
    fig, axs = plt.subplots(2, 1)
    for prefix in prefix_list:
        all_data_percentage = []
        all_data_diff = []
        num_runs = len(os.listdir('../saved_features/' + domain + '/' + prefix))
        for seed in range(1, 1+num_runs):
            folder = '../saved_features/' + domain + '/' + prefix + '/'+'seed_'+str(seed)
            files = os.listdir(folder)
            percentage_list = []
            diff_list = []
            for num in range(len(files)):
                percentage, average_diff = compare_pairs(folder+'/tsne-'+str(5000*(num+1))+'.npz')
                percentage_list.append(percentage)
                diff_list.append(average_diff)
            all_data_percentage.append(percentage_list)
            all_data_diff.append(diff_list)
        percentage_mean, percentage_std = average_the_data(all_data_percentage)
        diff_mean, diff_std = average_the_data(all_data_diff)
        axs[0].fill_between(np.arange(1, percentage_mean.shape[0]+1)*10000,
                            percentage_mean - percentage_std/math.sqrt(num_runs),
                            percentage_mean + percentage_std/math.sqrt(num_runs), alpha=0.4)
        axs[0].plot(np.arange(1, percentage_mean.shape[0] + 1) * 10000, percentage_mean)
        axs[0].set_title('percentage')
        axs[1].fill_between(np.arange(1, diff_mean.shape[0]+1)*10000,
                            diff_mean - diff_std/math.sqrt(num_runs),
                            diff_mean + diff_std/math.sqrt(num_runs), alpha=0.4)
        axs[1].plot(np.arange(1, diff_mean.shape[0] + 1) * 10000, diff_mean, label=prefix)
        axs[1].set_title('mean relative error')
    fig.legend(fontsize=6)
    plt.savefig('../saved_features/saved_tsne_fig/'+domain+'_'+title+'.png')


# prefix_list = ['sac_cheetah_run_crop', 'RAD_cheetah_run_crop', 'DrQ_cheetah_run_crop',
#                'DrQ_aug_when_act_cheetah_run_crop', 'DrQ+t_sne+rotation_30']
# prefix_list_2 = ['sac_reacher_hard_rotation', 'RAD_reacher_hard_rotation', 'DrQ_reacher_hard_rotation',
#                  'DrQ_aug_when_act_reacher_hard_rotation', 'DrQ+t_sne+crop']

domain_1 = 'cheetah_run_new'
prefix_1 = ['cheetah_run+sac+visualize_crop',
            'cheetah_run+RAD_crop+visualize_crop', 'cheetah_run+RAD_crop+aug_when_act+visualize_crop',
            'cheetah_run+DrQ_crop+visualize_crop', 'cheetah_run+DrQ_crop+aug_when_act+visualize_crop',
            'cheetah_run+DrQ_remove_small_crop+aug_when_act+visualize_crop',
            'cheetah_run+DrQ_3_aug+aug_when_act+visualize_crop', 'cheetah_run+DrQ_rotation_shift+visualize']
# prefix_3 = ['cheetah_run+RAD_crop+visualize_crop+determinitic',
#             'cheetah_run+RAD_crop+aug_when_evaluate+visualize+determinitic',
#             'cheetah_run+RAD_crop+aug_when_act+visualize_crop+determinitic',
#             'cheetah_run+DrQ_crop+visualize+deterministic',
#             'cheetah_run+DrQ_crop+aug_when_act+visualize+deterministic']
# prefix_4 = ['cheetah_run+DrQ_crop+aug_when_act+visualize+deterministic',
#             'cheetah_run+DrQ_rotation_15_crop+visualize+deterministic',
#             'cheetah_run+DrQ_rotation_15_crop+aug_when_act+visualize+deterministic',
#             'cheetah_run+DrQ_remove_00_crop+aug_when_act+visualize+deterministic',
#             'cheetah_run+DrQ_remove_01_00_crop+aug_when_act+visualize+deterministic']
prefix_3 = ['cheetah_run+RAD_crop+visualize+determinitic',
            'cheetah_run+RAD_crop+aug_when_evaluate+visualize+determinitic',
            'cheetah_run+RAD_crop+aug_when_act+visualize+determinitic',
            'cheetah_run+DrQ_crop+visualize+deterministic',
            'cheetah_run+DrQ_crop+aug_when_act+visualize+deterministic']
prefix_4 = ['cheetah_run+DrQ_crop+visualize+deterministic',
            'cheetah_run+DrQ_crop+aug_when_act+visualize+deterministic',
            'cheetah_run+DrQ_rotation_15_crop+aug_when_act+visualize+deterministic',
            'cheetah_run+DrQ_remove_01_00_crop+aug_when_act+visualize+deterministic']
domain_2 = 'walker_run_new'
prefix_2 = ['walker_run+sac+visualize_crop',
            'walker_run+RAD+visualize_crop', 'walker_run+RAD+aug_when_act+visualize_crop',
            'walker_run+DrQ+visualize_crop', 'walker_run+DrQ+aug_when_act+visualize_crop',
            'walker_run+DrQ_remove_small_crop+aug_when_act+visualize_crop',
            'walker_run+DrQ_3_aug+aug_when_act+visualize_crop', 'walker_run+DrQ_rotation_shift+visualize']
prefix_5 = ['walker_run+DrQ_crop+visualize+deterministic',
            'walker_run+DrQ_crop+aug_when_act+visualize+deterministic',
            'walker_run+DrQ_remove_01_00_crop+aug_when_act+visualize+deterministic']

# 2.20
# plot_percentage(domain_1, prefix_1, title='original')
# plot_percentage(domain_2, prefix_2, title='original')

# 2.27
plot_percentage(domain_1, prefix_3, title='deterministic_aug_when_act_new')
plot_percentage(domain_1, prefix_4, title='deterministic_different_aug_new')
plot_percentage(domain_2, prefix_5, title='deterministic_aug_when_act_new')


# plot_target_Q(regularization, step, prefix)

# regularization = 16
# plot_pairs(regularization, step, prefix)
# plot_target_Q(regularization, step, prefix)