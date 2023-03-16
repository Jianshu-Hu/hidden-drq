import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import kornia

import replay_buffer
import utils
import hydra
import os
import data_aug_new as new_aug

from sklearn.cluster import KMeans
from sklearn import manifold
from torch.distributions import kl_divergence


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim, device):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.device = device

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim,
                               2 * action_shape[0], hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def forward_mu_std(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        return mu, std

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class DRQAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, obs_shape, action_shape, action_range, device,
                 encoder_cfg, critic_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, image_pad, data_aug, RAD,
                 degrees, visualize, tag, seed, dist_alpha, add_kl_loss, init_beta, add_actor_obs_aug_loss,
                 update_beta, avg_target, critic_tangent_prop, critic_tangent_prop_weight):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.encoder_cfg = encoder_cfg
        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()
        self.critic_target.train()

        # data aug
        self.data_aug = data_aug

        self.image_pad = image_pad
        self.aug = new_aug.aug(data_aug, image_pad, obs_shape, degrees, dist_alpha)
        self.tangent_prop_regu = new_aug.TangentProp(data_aug, device)

        self.mse_loss = nn.MSELoss()
        self.RAD = RAD
        self.visualize = visualize
        self.tag = tag
        self.seed = seed
        self.add_kl_loss = add_kl_loss
        self.update_beta = update_beta
        self.avg_target = avg_target
        self.critic_tangent_prop = critic_tangent_prop
        self.critic_tangent_prop_weight = critic_tangent_prop_weight
        if self.add_kl_loss:
            self.init_beta = init_beta
            target_KL = 0.02
            self.log_beta = torch.tensor([np.log(self.init_beta)]).to(self.device)
            self.log_beta.requires_grad = True
            # set target KL divergence
            self.target_KL = target_KL
            self.log_beta_optimizer = torch.optim.Adam([self.log_beta], lr=lr)
        self.add_actor_obs_aug_loss = add_actor_obs_aug_loss

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def beta(self):
        return self.log_beta.exp()

    def act(self, obs, sample=False):

        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, obs_aug_1, obs_aug_2, action, reward, next_obs,
                      next_obs_aug_1, next_obs_aug_2, not_done, logger, step):
        with torch.no_grad():
            dist_aug_1 = self.actor(next_obs_aug_1)
            next_action_aug_1 = dist_aug_1.rsample()
            log_prob_aug_1 = dist_aug_1.log_prob(next_action_aug_1).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug_1,
                                                      next_action_aug_1)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug_1
            target_Q_aug_1 = reward + (not_done * self.discount * target_V)

            dist_aug_2 = self.actor(next_obs_aug_2)
            next_action_aug_2 = dist_aug_2.rsample()
            log_prob_aug_2 = dist_aug_2.log_prob(next_action_aug_2).sum(-1,
                                                                  keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug_2,
                                                      next_action_aug_2)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug_2
            target_Q_aug_2 = reward + (not_done * self.discount * target_V)

        # visualize the embedding
        if self.visualize:
            if step % 5000 == 0:
                t_sne = manifold.TSNE(n_components=2, init='pca', random_state=self.seed)
                with torch.no_grad():
                    X1 = self.critic.encoder(obs_aug_1).cpu().numpy()
                    X2 = self.critic.encoder(obs_aug_2).cpu().numpy()

                Y = t_sne.fit_transform(np.vstack((X1, X2)))
                KL = torch.mean(kl_divergence(dist_aug_1, dist_aug_2))
                # save the projected embedding
                prefix = '/bigdata/users/jhu/hidden-drq/outputs/'
                path = prefix+self.tag
                if not os.path.exists(path):
                    os.mkdir(path)
                if not os.path.exists(path + '/seed_' + str(self.seed)):
                    os.mkdir(path + '/seed_' + str(self.seed))
                np.savez(path + '/seed_' + str(self.seed) + '/tsne-' + str(step) + '.npz',
                         target_Q=target_Q_aug_1.cpu().numpy(), target_Q_aug=target_Q_aug_2.cpu().numpy(), Y=Y,
                         next_action=next_action_aug_1.cpu().numpy(), next_action_aug=next_action_aug_2.cpu().numpy(),
                         KL=KL.cpu().numpy())

        if self.critic_tangent_prop:
            with torch.no_grad():
                # calculate the expected transformed image and tangent vector
                expected_trans_obs, variance_trans_obs = self.tangent_prop_regu.moments_transformed_obs(obs_aug_1)
                tangent_vector1 = torch.pow(torch.abs(variance_trans_obs), 0.5)
                expected_trans_obs, variance_trans_obs = self.tangent_prop_regu.moments_transformed_obs(obs_aug_2)
                tangent_vector2 = torch.pow(torch.abs(variance_trans_obs), 0.5)
                # tangent_vector = self.tangent_prop_regu.tangent_vector(obs)
            obs_aug_1.requires_grad = True
            obs_aug_2.requires_grad = True
            # critic loss
            target_Q = (target_Q_aug_1 + target_Q_aug_2) / 2
            Q1_aug_1, Q2_aug_1 = self.critic(obs_aug_1, action)
            Q1_aug_2, Q2_aug_2 = self.critic(obs_aug_2, action)
            critic_loss = F.mse_loss(Q1_aug_1, target_Q) + F.mse_loss(Q2_aug_1, target_Q)
            critic_loss += F.mse_loss(Q1_aug_2, target_Q) + F.mse_loss(Q2_aug_2, target_Q)
            critic_loss = critic_loss / 2
            logger.log('train_critic/loss', critic_loss, step)

            # add regularization for tangent prop
            # calculate the Jacobian matrix for non-linear model
            Q1 = torch.min(Q1_aug_1, Q2_aug_1)
            jacobian1 = torch.autograd.grad(outputs=Q1, inputs=obs_aug_1,
                                           grad_outputs=torch.ones(Q1.size(), device=self.device),
                                           retain_graph=True, create_graph=True)[0]
            Q2 = torch.min(Q1_aug_2, Q2_aug_2)
            jacobian2 = torch.autograd.grad(outputs=Q2, inputs=obs_aug_2,
                                           grad_outputs=torch.ones(Q2.size(), device=self.device),
                                           retain_graph=True, create_graph=True)[0]
            tan_loss1 = torch.mean(torch.sum(torch.pow(
                torch.linalg.matrix_norm(jacobian1 * tangent_vector1), 2), dim=-1), dim=-1)
            tan_loss2 = torch.mean(torch.mean(torch.pow(
                torch.linalg.matrix_norm(jacobian2 * tangent_vector2), 2), dim=-1), dim=-1)

            tangent_prop_loss = tan_loss1+tan_loss2
            critic_loss += self.critic_tangent_prop_weight*tangent_prop_loss

            logger.log('train_critic/tangent_prop_loss', tangent_prop_loss, step)
        else:
            if not self.RAD:
                Q1_aug_1, Q2_aug_1 = self.critic(obs_aug_1, action)
                Q1_aug_2, Q2_aug_2 = self.critic(obs_aug_2, action)
                if self.avg_target:
                    # DrQ
                    target_Q = (target_Q_aug_1 + target_Q_aug_2) / 2
                    critic_loss = F.mse_loss(Q1_aug_1, target_Q) + F.mse_loss(Q2_aug_1, target_Q)
                    critic_loss += F.mse_loss(Q1_aug_2, target_Q) + F.mse_loss(Q2_aug_2, target_Q)
                else:
                    # DrQ without average target / RAD with two samples
                    critic_loss = F.mse_loss(Q1_aug_1, target_Q_aug_1) + F.mse_loss(Q2_aug_1, target_Q_aug_1)
                    critic_loss += F.mse_loss(Q1_aug_2, target_Q_aug_2) + F.mse_loss(Q2_aug_2, target_Q_aug_2)
                critic_loss = critic_loss/2
            else:
                # apply data augmentation
                current_Q1, current_Q2 = self.critic(obs_aug_1, action)
                critic_loss = F.mse_loss(current_Q1, target_Q_aug_1) + F.mse_loss(current_Q2, target_Q_aug_1)

            logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.critic_tangent_prop:
            obs_aug_1.grad.zero_()
            obs_aug_2.grad.zero_()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, obs_aug_1, obs_aug_2, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs_aug_1, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs_aug_1, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        if self.add_actor_obs_aug_loss:
            dist_aug = self.actor(obs_aug_2, detach_encoder=True)
            action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(action_aug).sum(-1, keepdim=True)
            # detach conv filters, so we don't update them with the actor loss
            actor_Q1_aug, actor_Q2_aug = self.critic(obs_aug_2, action_aug, detach_encoder=True)

            actor_Q_aug = torch.min(actor_Q1_aug, actor_Q2_aug)

            actor_loss_aug = (self.alpha.detach() * log_prob_aug - actor_Q_aug).mean()

            actor_loss += actor_loss_aug

            actor_loss = actor_loss/2

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor.log(logger, step)

        if self.add_kl_loss:
            # KL divergence between A(obs_aug_1) and A(obs_aug_2)
            with torch.no_grad():
                mu, std = self.actor.forward_mu_std(obs_aug_1)
            mu_aug, std_aug = self.actor.forward_mu_std(obs_aug_2, detach_encoder=True)
            dist1 = utils.SquashedNormal(mu, std)
            dist1_aug = utils.SquashedNormal(mu_aug, std_aug)

            KL = torch.mean(kl_divergence(dist1, dist1_aug))
            weighted_KL = self.beta.detach()*KL
            logger.log('train_actor/KL_loss', KL, step)

            self.actor_optimizer.zero_grad()
            weighted_KL.backward()
            self.actor_optimizer.step()

            if self.update_beta:
                # update beta
                beta_loss = -(self.beta * (KL - self.target_KL).detach()).mean()
                logger.log('train_beta/loss', beta_loss, step)
                logger.log('train_beta/value', self.beta, step)
                self.log_beta_optimizer.zero_grad()
                beta_loss.backward()
                self.log_beta_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, obs_aug_1, next_obs_aug_1, obs_aug_2, next_obs_aug_2 \
            = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, obs_aug_1, obs_aug_2, action, reward,
                           next_obs, next_obs_aug_1, next_obs_aug_2, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, obs_aug_1, obs_aug_2, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

