from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import os


def compare_pairs(prefix, file):
    # fig, axs = plt.subplots(1, 1)
    data = np.load(prefix+file)
    Y = data['Y']
    batch_size = 512
    # calculate percentage of close samples
    distance = np.sqrt(np.sum((Y[:batch_size]-Y[batch_size:])**2, axis=-1))
    percentage = np.sum((distance < 2))/batch_size
    print('percentage of pairs whose distance is smaller than 1: '+str(percentage))
    return percentage
    # # plot
    # plot_sample_num = 10
    # plot_idxs = np.random.randint(0, batch_size, size=plot_sample_num)
    # color = np.arange(plot_sample_num)
    # axs.scatter(Y[plot_idxs, 0], Y[plot_idxs, 1], c=color, cmap=plt.cm.Spectral)
    # axs.scatter(Y[batch_size+plot_idxs, 0], Y[batch_size + plot_idxs, 1], c=color, cmap=plt.cm.Spectral)
    # axs.set_title(file.replace('.npz', ''))
    # plt.savefig('../outputs/saved_tsne_fig/' + file.replace('.npz', '') + '.png')


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


def plot_percentage(prefix_list, fig_name):
    fig, axs = plt.subplots(1, 1)
    for prefix in prefix_list:
        print(prefix)
        files = os.listdir('../outputs/' + prefix + '/')
        files.sort()
        percentage_list = []
        for file in files:
            percentage = compare_pairs('../outputs/' + prefix + '/', file)
            percentage_list.append(percentage)
        axs.plot(np.arange(1, len(percentage_list)+1)*20000, percentage_list, label=prefix)
    plt.legend()
    plt.savefig('../outputs/saved_tsne_fig/'+fig_name+'.png')


prefix_list = ['sac_cheetah_run_crop', 'RAD_cheetah_run_crop', 'DrQ_cheetah_run_crop',
               'DrQ_aug_when_act_cheetah_run_crop', 'DrQ+t_sne+rotation_30']
prefix_list_2 = ['sac_reacher_hard_rotation', 'RAD_reacher_hard_rotation', 'DrQ_reacher_hard_rotation',
                 'DrQ_aug_when_act_reacher_hard_rotation', 'DrQ+t_sne+crop']

plot_percentage(prefix_list, fig_name='cheetah')
plot_percentage(prefix_list_2, fig_name='reacher')


# plot_target_Q(regularization, step, prefix)

# regularization = 16
# plot_pairs(regularization, step, prefix)
# plot_target_Q(regularization, step, prefix)