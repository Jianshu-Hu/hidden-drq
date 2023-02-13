from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import os


def compare_pairs(file):
    data = np.load(file)
    Y = data['Y']
    batch_size = 512
    # calculate percentage of close samples
    distance = np.sqrt(np.sum((Y[:batch_size]-Y[batch_size:])**2, axis=-1))
    percentage = np.sum((distance < 2))/batch_size
    # print('percentage of pairs whose distance is smaller than 1: '+str(percentage))
    return percentage


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
    average_data = np.average(all_data, axis=0)
    return average_data


def plot_percentage(domain, prefix_list, fig_name):
    fig, axs = plt.subplots(1, 1)
    for prefix in prefix_list:
        all_data = []
        for seed in range(1, 4):
            folder = '../saved_features/' + domain + '/' + prefix + '/'+'seed_'+str(seed)
            files = os.listdir(folder)
            percentage_list = []
            for num in range(len(files)):
                percentage = compare_pairs(folder+'/tsne-'+str(5000*(num+1))+'.npz')
                percentage_list.append(percentage)
            all_data.append(percentage_list)
        average_data = average_the_data(all_data)
        axs.plot(np.arange(1, average_data.shape[0]+1)*10000, average_data, label=prefix)
    plt.legend()
    plt.savefig('../saved_features/saved_tsne_fig/'+domain+'.png')


# prefix_list = ['sac_cheetah_run_crop', 'RAD_cheetah_run_crop', 'DrQ_cheetah_run_crop',
#                'DrQ_aug_when_act_cheetah_run_crop', 'DrQ+t_sne+rotation_30']
# prefix_list_2 = ['sac_reacher_hard_rotation', 'RAD_reacher_hard_rotation', 'DrQ_reacher_hard_rotation',
#                  'DrQ_aug_when_act_reacher_hard_rotation', 'DrQ+t_sne+crop']

domain_1 = 'cheetah_run'
prefix_1 = ['cheetah_run+sac+visualize_crop', 'cheetah_run+RAD_crop+visualize_crop',
            'cheetah_run+DrQ_crop+visualize_crop']
domain_2 = 'walker_run'
prefix_2 = ['walker_run+sac+visualize_crop', 'walker_run+RAD+visualize_crop',
            'walker_run+DrQ+visualize_crop']

plot_percentage(domain_1, prefix_1, fig_name='cheetah')
plot_percentage(domain_2, prefix_2, fig_name='reacher')


# plot_target_Q(regularization, step, prefix)

# regularization = 16
# plot_pairs(regularization, step, prefix)
# plot_target_Q(regularization, step, prefix)