import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random

import utils
import hydra


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels=in_channel, reduction_ratio=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        out = x*self.channel_attention(x)
        out = out*self.spatial_attention(out)
        weight = random.random()
        return (1-weight)*out+residual


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim, add_attention_module):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.add_attention_module = add_attention_module

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        if self.add_attention_module:
            self.CBAMs = nn.ModuleList([
                CBAMBlock(self.num_filters),
                CBAMBlock(self.num_filters),
                CBAMBlock(self.num_filters)
            ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs, with_attention_module=False):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            # if with_attention_module:
            #     conv = self.CBAMs[i-1](conv)
            self.outputs['conv%s' % (i + 1)] = conv

        if with_attention_module:
            conv = self.CBAMs[self.num_layers - 2](conv)

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False, with_attention_module=False):
        h = self.forward_conv(obs, with_attention_module)

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

        self.projector = utils.mlp(self.encoder.feature_dim, hidden_dim, 128, hidden_depth)
        self.predictor = utils.mlp(128, hidden_dim, 128, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False, with_attention_module=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder, with_attention_module=with_attention_module)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def embedding(self, obs):
        features = self.encoder(obs, detach=False)
        return features

    def forward_projector(self, obs):
        features = self.encoder(obs, detach=False)
        return self.projector(features)

    def forward_predictor(self, obs):
        features = self.encoder(obs, detach=False)
        return self.predictor(self.projector(features))

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
                 critic_target_update_frequency, batch_size):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

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
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()
        self.critic_target.train()

        self.simclr_criterion = SupConLoss(temperature=0.5)
        self.mse = nn.MSELoss()
        # record the max Q error so far for calculating the weight for regularization term
        self.max_q_error = 0

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def cosine_similarity_loss(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        similarity_loss = torch.mean(2 - 2 * (x * y).sum(dim=-1))

        return similarity_loss

    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, logger, step, regularization):
        # regularization:
        # 0:SAC
        # 1:SAC + drq
        # 2:SAC + regularization

        # 4:SAC + BYOL regularization
        # 5:SAC + attention regularization
        # 6:SAC + SimCLR
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            if regularization in {1}:
                dist_aug = self.actor(next_obs_aug)
                next_action_aug = dist_aug.rsample()
                log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                      keepdim=True)
                target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                          next_action_aug)
                target_V = torch.min(
                    target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
                target_Q_aug = reward + (not_done * self.discount * target_V)

                target_Q = (target_Q + target_Q_aug) / 2
            if regularization in {5}:
                dist_aug = self.actor(next_obs)
                next_action_aug = dist_aug.rsample()
                log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                      keepdim=True)
                target_Q1, target_Q2 = self.critic_target.forward(next_obs,
                                                          next_action_aug, with_attention_module=True)
                target_V = torch.min(
                    target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
                target_Q_aug = reward + (not_done * self.discount * target_V)

                target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        if regularization in {1}:
            Q1_aug, Q2_aug = self.critic(obs_aug, action)
            critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
                Q2_aug, target_Q)

        # use attention for creating augmentation
        if regularization in {5}:
            Q1_aug, Q2_aug = self.critic.forward(obs, action, with_attention_module=True)
            critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
                Q2_aug, target_Q)

        # add regularization term for hidden layer output
        if regularization in {2}:
            with torch.no_grad():
                features_target = self.critic_target.embedding(obs)
                features_aug_target = self.critic_target.embedding(obs_aug)
                lambda_weight = 1.0
                # # calculate the weight for regularization
                # Q1, Q2 = self.critic.forward(obs, action)
                # Q = torch.min(Q1, Q2)
                # Q1_aug, Q2_aug = self.critic.forward(obs_aug, action)
                # Q_aug = torch.min(Q1_aug, Q2_aug)
                # Q_error = self.mse(Q, Q_aug)
                # if Q_error > self.max_q_error:
                #     self.max_q_error = Q_error
                # lambda_weight = 1.0*Q_error/self.max_q_error
            features = self.critic.embedding(obs)
            features_aug = self.critic.embedding(obs_aug)

            # similarity_loss = self.cosine_similarity_loss(features, features_aug_target) + \
            #                   self.cosine_similarity_loss(features_aug, features_target)
            # critic_loss += similarity_loss
            l2_loss = self.mse(features, features_aug_target) + self.mse(features_aug, features_target)
            critic_loss += lambda_weight*l2_loss
        # add regularization similar with BYOL
        if regularization == 4:
            with torch.no_grad():
                byol_target = self.critic_target.forward_projector(obs)
                byol_aug_target = self.critic_target.forward_projector(obs_aug)
                lambda_weight = 1.0
                # # calculate the weight for regularization
                # Q1, Q2 = self.critic.forward(obs, action)
                # Q = torch.min(Q1, Q2)
                # Q1_aug, Q2_aug = self.critic.forward(obs_aug, action)
                # Q_aug = torch.min(Q1_aug, Q2_aug)
                # Q_error = self.mse(Q, Q_aug)
                # if Q_error > self.max_q_error:
                #     self.max_q_error = Q_error
                # lambda_weight = 1.0*Q_error/self.max_q_error
            byol_prediction = self.critic.forward_predictor(obs)
            byol_aug_prediction = self.critic.forward_predictor(obs_aug)

            similarity_loss = self.cosine_similarity_loss(byol_prediction, byol_aug_target) + \
                              self.cosine_similarity_loss(byol_aug_prediction, byol_target)
            critic_loss += lambda_weight*similarity_loss

        # add regularization similar to SimCLR
        if regularization == 6:
            with torch.no_grad():
                lambda_weight = 0.1
                # # calculate the weight for regularization
                # Q1, Q2 = self.critic.forward(obs, action)
                # Q = torch.min(Q1, Q2)
                # Q1_aug, Q2_aug = self.critic.forward(obs_aug, action)
                # Q_aug = torch.min(Q1_aug, Q2_aug)
                # Q_error = self.mse(Q, Q_aug)
                # if Q_error > self.max_q_error:
                #     self.max_q_error = Q_error
                # lambda_weight = 0.1*Q_error/self.max_q_error
            proj1 = F.normalize(self.critic.forward_projector(obs), dim=-1)
            proj2 = F.normalize(self.critic.forward_projector(obs_aug), dim=-1)

            # features: [bsz, n_views, f_dim]
            # `n_views` is the number of crops from each image
            # better be L2 normalized in f_dim dimension
            features = torch.cat((torch.unsqueeze(proj1, dim=1), torch.unsqueeze(proj2, dim=1)), dim=1)
            contrastive_loss = self.simclr_criterion(features)
            critic_loss += lambda_weight*contrastive_loss

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step, regularization, CBAM):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
            self.batch_size, CBAM)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step, regularization)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)


# SimCLR loss from https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
