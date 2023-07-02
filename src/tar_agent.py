import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from transfer_util import initialize_model
from stage1_models import BasicBlock, ResNet84
import os
import copy
from PIL import Image
import platform
from numbers import Number
import utils
from r3m import load_r3m
from torchvision.models import resnet18, resnet34, resnet50

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class RLEncoder(nn.Module):
    def __init__(self, obs_shape, model_name, device):
        super().__init__()
        # a wrapper over a non-RL encoder model
        self.device = device
        assert len(obs_shape) == 3
        self.n_input_channel = obs_shape[0]
        assert self.n_input_channel % 3 == 0
        self.n_images = self.n_input_channel // 3
        self.model = self.init_model(model_name)
        self.model.fc = Identity()
        self.repr_dim = self.model.get_feature_size()

        self.normalize_op = transforms.Normalize((0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225))
        self.channel_mismatch = True

    def init_model(self, model_name):
        # model name is e.g. resnet6_32channel
        n_layer_string, n_channel_string = model_name.split('_')
        layer_string_to_layer_list = {
            'resnet6': [0, 0, 0, 0],
            'resnet10': [1, 1, 1, 1],
            'resnet18': [2, 2, 2, 2],
        }
        channel_string_to_n_channel = {
            '32channel': 32,
            '64channel': 64,
        }
        layer_list = layer_string_to_layer_list[n_layer_string]
        start_num_channel = channel_string_to_n_channel[n_channel_string]
        return ResNet84(BasicBlock, layer_list, start_num_channel=start_num_channel).to(self.device)

    def expand_first_layer(self):
        # convolutional channel expansion to deal with input mismatch
        multiplier = self.n_images
        self.model.conv1.weight.data = self.model.conv1.weight.data.repeat(1,multiplier,1,1) / multiplier
        means = (0.485, 0.456, 0.406) * multiplier
        stds = (0.229, 0.224, 0.225) * multiplier
        self.normalize_op = transforms.Normalize(means, stds)
        self.channel_mismatch = False

    def freeze_bn(self):
        # freeze batch norm layers (VRL3 ablation shows modifying how
        # batch norm is trained does not affect performance)
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def get_parameters_that_require_grad(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                params.append(param)
        return params

    def transform_obs_tensor_batch(self, obs):
        # transform obs batch before put into the pretrained resnet
        new_obs = self.normalize_op(obs.float()/255)
        return new_obs

    def _forward_impl(self, x):
        x = self.model.get_features(x)
        return x

    def forward(self, obs):
        o = self.transform_obs_tensor_batch(obs)
        h = self._forward_impl(o)
        return h

class TaskClassifier(nn.Module):
    def __init__(self, action_dim, seq_len, hidden_dim, num_classes):
        super(TaskClassifier, self).__init__()
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(action_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
class TaskDiscriminator(nn.Module):
    def __init__(self, action_dim, seq_len, hidden_dim):
        super(TaskDiscriminator, self).__init__()
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(action_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    

class ActionAE(nn.Module):
    def __init__(self, action_dim, act_rep_dim, hidden_dim):
        super(ActionAE, self).__init__()
        self.action_dim = action_dim
        self.latent_dim = act_rep_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, act_rep_dim)
        self.enc_log_sigma = torch.nn.Linear(hidden_dim, act_rep_dim)
        
        self.fc3 = nn.Linear(act_rep_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_mu = nn.Linear(hidden_dim, action_dim)
        self.dec_log_sigma = nn.Linear(hidden_dim, action_dim)
        
    def reparamize(self, mu, log_sigma):
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        z = self.reparamize(mu, log_sigma)
        return z, mu, log_sigma
    
    def decode(self, z):
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc4(x))
        mu = self.dec_mu(x)
        log_sigma = self.dec_log_sigma(x)
        x = self.reparamize(mu, log_sigma)
        return x, mu, log_sigma
    
    def forward(self, x):
        z, enc_mu, enc_log_sigma = self.encode(x)
        x_rec, dec_mu, dec_log_sigma = self.decode(z)
        return x_rec, x, enc_mu, enc_log_sigma
    
    
class LatentInvDynMLP(nn.Module):
    def __init__(self, ob_rep_dim, act_rep_dim, hidden_dim):
        super(LatentInvDynMLP, self).__init__()
        self.ob_rep_dim = ob_rep_dim
        self.act_rep_dim = act_rep_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(ob_rep_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_rep_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.action_shift=0
        self.action_scale=1
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        mu = mu * self.action_scale + self.action_shift
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

    def forward_with_pretanh(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        pretanh = mu
        mu = torch.tanh(mu)
        mu = mu * self.action_scale + self.action_shift
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist, pretanh

class LatentActor(nn.Module):
    def __init__(self, ob_rep_dim, act_rep_dim, feature_dim, hidden_dim):
        super(LatentActor, self).__init__()

        self.trunk = nn.Sequential(nn.Linear(ob_rep_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, act_rep_dim))

        self.action_shift=0
        self.action_scale=1
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        mu = mu * self.action_scale + self.action_shift
        std = torch.ones_like(mu) * std

        # dist = torch.distributions.Normal(mu, std)
        dist = utils.ClampedGaussian(mu, std)
        return dist

    def forward_with_pretanh(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        pretanh = mu
        mu = torch.tanh(mu)
        mu = mu * self.action_scale + self.action_shift
        std = torch.ones_like(mu) * std
        
        # dist = torch.distributions.Normal(mu, std)
        dist = utils.ClampedGaussian(mu, std)
        return dist, pretanh

class LatentCritic(nn.Module):
    def __init__(self, ob_rep_dim, act_rep_dim, feature_dim, hidden_dim):
        super(LatentCritic, self).__init__()

        self.trunk = nn.Sequential(nn.Linear(ob_rep_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + act_rep_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + act_rep_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class TARAgent:
    def __init__(self, obs_shape, action_shape, act_rep_dim, seq_len, device, use_sensor, lr, feature_dim, hidden_dim, policy_output_type,
                 cls_weight, aln_weight, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_clip, use_wb, use_data_aug, encoder_lr_scale,
                 visual_model_name, safe_q_target_factor, safe_q_threshold, pretanh_penalty, pretanh_threshold,
                 stage2_update_encoder, stage2_update_autoencoder, cql_weight, cql_temp, cql_n_random, stage2_std, stage2_bc_weight,
                 stage3_update_encoder, stage3_update_autoencoder, std0, std1, std_n_decay,
                 stage3_bc_lam0, stage3_bc_lam1,
                 data_dir):
        self.obs_shape = obs_shape
        assert self.obs_shape[0] % 3 == 0
        self.n_images = int(self.obs_shape[0] / 3)
        self.action_shape = action_shape
        self.device = device
        self.policy_output_type = policy_output_type
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_wb = use_wb
        self.num_expl_steps = num_expl_steps
        self.data_dir = data_dir

        self.stage2_std = stage2_std
        self.stage2_update_encoder = stage2_update_encoder
        self.stage2_update_autoencoder = stage2_update_autoencoder

        if std1 > std0:
            std1 = std0
        self.stddev_schedule = "linear(%s,%s,%s)" % (str(std0), str(std1), str(std_n_decay))

        self.stddev_clip = stddev_clip
        self.use_data_aug = use_data_aug
        self.safe_q_target_factor = safe_q_target_factor
        self.q_threshold = safe_q_threshold
        self.pretanh_penalty = pretanh_penalty

        self.cql_temp = cql_temp
        self.cql_weight = cql_weight
        self.cql_n_random = cql_n_random

        self.pretanh_threshold = pretanh_threshold

        self.stage2_bc_weight = stage2_bc_weight
        self.stage3_bc_lam0 = stage3_bc_lam0
        self.stage3_bc_lam1 = stage3_bc_lam1

        if stage3_update_encoder and encoder_lr_scale > 0 and len(obs_shape) > 1:
            self.stage3_update_encoder = True
        else:
            self.stage3_update_encoder = False

        self.stage3_update_autoencoder = stage3_update_autoencoder

        # self.encoder = RLEncoder(obs_shape, visual_model_name, device).to(device)
        if visual_model_name == 'r3m50':
            self.encoder = load_r3m('resnet50')
        elif visual_model_name == 'r3m34':
            self.encoder = load_r3m('resnet34')
        elif visual_model_name == 'r3m18':
            self.encoder = load_r3m('resnet18')
        elif visual_model_name == 'resnet50':
            self.encoder = resnet50(pretrained=True)
            self.encoder.fc = nn.Identity()
        elif visual_model_name == 'resnet34':
            self.encoder = resnet34(pretrained=True)
            self.encoder.fc = nn.Identity()
        elif visual_model_name == 'resnet18':
            self.encoder = resnet18(pretrained=True)
            self.encoder.fc = nn.Identity()
        elif visual_model_name == 'vrl3':
            self.encoder = RLEncoder(obs_shape, 'resnet6_32channel', device)
            self.load_pretrained_encoder(self.get_pretrained_model_path('resnet6_32channel'))

        self.expand_encoder(visual_model_name)
        
        self.encoder.to(device)
        self.encoder.eval()
        
        with torch.no_grad():
            dummy_input = torch.zeros([1, self.obs_shape[0], 84, 84]).to(device)
            dummy_input = self.encoder(dummy_input)
            ob_rep_dim = dummy_input.shape[1]
            
        if use_sensor:
            ob_rep_dim = ob_rep_dim + 24
            
        self.ob_rep_dim = ob_rep_dim
            
        self.action_ae = ActionAE(action_shape[0], act_rep_dim, hidden_dim).to(device)
        self.task_cls = TaskDiscriminator(act_rep_dim, seq_len, hidden_dim).to(device)
        self.inv_dyn = LatentInvDynMLP(ob_rep_dim, act_rep_dim, hidden_dim).to(device)
        
        self.act_dim = action_shape[0]
        self.act_rep_dim = act_rep_dim
        self.seq_len = seq_len

        if self.policy_output_type == 'latent':
            self.actor = LatentActor(ob_rep_dim, act_rep_dim, feature_dim,
                           hidden_dim).to(device)
        else:
            self.actor = Actor(ob_rep_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = LatentCritic(ob_rep_dim, act_rep_dim, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = LatentCritic(ob_rep_dim, act_rep_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.cls_weight = cls_weight
        self.aln_weight = aln_weight
        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.action_ae_opt = torch.optim.Adam(self.action_ae.parameters(), lr=lr)
        self.action_ae_ft_opt = torch.optim.Adam(self.action_ae.parameters(), lr=lr*0.1)
        self.task_cls_opt = torch.optim.Adam(self.task_cls.parameters(), lr=lr)
        self.inv_dyn_opt = torch.optim.Adam(self.inv_dyn.parameters(), lr=lr)

        encoder_lr = lr * encoder_lr_scale
        """ set up encoder optimizer """
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.train()
        self.critic_target.train()

    def get_pretrained_model_path(self, stage1_model_name):
        # given a stage1 model name, return the path to the pretrained model
        data_folder_path = self.data_dir
        model_folder_path = os.path.join(data_folder_path, "trained_models")
        model_path = os.path.join(model_folder_path, stage1_model_name + '_checkpoint.pth.tar')
        return model_path
    
    def load_pretrained_encoder(self, model_path, verbose=True):
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        state_dict = checkpoint['state_dict']

        pretrained_dict = {}
        # remove `module.` if model was pretrained with distributed mode
        for k, v in state_dict.items():
            if 'module.' in k:
                name = k[7:]
            else:
                name = k
            pretrained_dict[name] = v
        self.encoder.model.load_state_dict(pretrained_dict, strict=False)
    
    def expand_encoder(self, model_name):
        if model_name[:4] == 'vrl3':
            multiplier = self.n_images
            self.encoder.model.conv1.weight.data = self.encoder.model.conv1.weight.data.repeat(1,multiplier,1,1) / multiplier
            means = (0.485, 0.456, 0.406) * multiplier
            stds = (0.229, 0.224, 0.225) * multiplier
            self.encoder.normalize_op = transforms.Normalize(means, stds)
            self.encoder.channel_mismatch = False
        elif model_name[:3] == 'r3m':
            self.encoder.module.convnet.conv1.weight.data = self.encoder.module.convnet.conv1.weight.repeat(1, self.n_images, 1, 1) / self.n_images
            means = (0.485, 0.456, 0.406) * self.n_images
            stds = (0.229, 0.224, 0.225) * self.n_images
            self.encoder.module.normlayer = transforms.Normalize(means, stds)
        else:
            self.encoder.conv1.weight.data = self.encoder.conv1.weight.repeat(1, self.n_images, 1, 1) / self.n_images
    
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode, obs_sensor=None, is_tensor_input=False, force_action_std=None):
        # eval_mode should be False when taking an exploration action in stage 3
        # eval_mode should be True when evaluate agent performance
        if force_action_std == None:
            stddev = utils.schedule(self.stddev_schedule, step)
            if step < self.num_expl_steps and not eval_mode:
                action = np.random.uniform(0, 1, (self.act_dim,)).astype(np.float32)
                return action
        else:
            stddev = force_action_std
        obs = obs.astype(np.float32)
        if is_tensor_input:
            obs = self.encoder(obs)
        else:
            obs = torch.as_tensor(obs, device=self.device)
            obs = self.encoder(obs.unsqueeze(0))

        if obs_sensor is not None:
            obs_sensor = torch.as_tensor(obs_sensor, device=self.device)
            obs_sensor = obs_sensor.unsqueeze(0)
            obs_combined = torch.cat([obs, obs_sensor], dim=1)
        else:
            obs_combined = obs

        dist = self.actor(obs_combined, stddev)
        if eval_mode:
            act_rep = dist.mean
        else:
            act_rep = dist.sample(clip=None)
            if step < self.num_expl_steps:
                act_rep.uniform_(-1.0, 1.0)
        
        if self.policy_output_type == 'latent':
            action = self.action_ae.decode(act_rep)[0]
            action = torch.clamp(action, -1.0, 1.0)
        else:
            action = act_rep
        
        return action.cpu().numpy()[0]
    
    def update_representation(self, replay_iter, traj_iter, step, use_sensor):
        
        metrics = dict()
        
        batch = next(replay_iter)
       
        if use_sensor: # TODO might want to...?
            obs, action, reward, discount, next_obs, obs_sensor, obs_sensor_next = utils.to_torch(batch, self.device)
        else:
            obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
            
        # augment
        if self.use_data_aug:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        
        with torch.no_grad():  
            ob_rep = self.encoder(obs)
            ob_rep_next = self.encoder(next_obs)
        
        if use_sensor:
            ob_rep = torch.cat([ob_rep, obs_sensor], dim=1)
            ob_rep_next = torch.cat([ob_rep_next, obs_sensor_next], dim=1)

        act_rep = self.action_ae.encode(action)[0]
        act_rec= self.action_ae.forward(action)[0]
        
        loss_rec = F.mse_loss(act_rec, action)

        if self.aln_weight > 0:
            act_rep_forward = self.inv_dyn(torch.cat([ob_rep, ob_rep_next], dim=1))
            loss_aln = F.mse_loss(act_rep_forward, act_rep) * self.aln_weight
        else:
            loss_aln = 0
        
        if self.cls_weight > 0:
            batch_traj = next(traj_iter)
            batch_seq, batch_label = utils.to_torch(batch_traj, self.device)
            with torch.no_grad():
                batch_seq_rep, _, _ = self.action_ae.encode(batch_seq.reshape(-1, batch_seq.shape[-1]))
                batch_seq_rep = batch_seq_rep.reshape(batch_seq.shape[0],self.seq_len, self.act_rep_dim)
            logits = self.task_cls(batch_seq_rep)
            loss_cls = F.binary_cross_entropy_with_logits(logits, batch_label) * self.cls_weight
        else:
            loss_cls = 0
        
        loss = loss_rec + loss_cls + loss_aln
        
        metrics['loss_rec'] = loss_rec.item()
        metrics['loss_cls'] = loss_cls.item() if self.cls_weight > 0 else 0
        metrics['loss_aln'] = loss_aln.item() if self.aln_weight > 0 else 0
        
        self.action_ae_opt.zero_grad()
        self.task_cls_opt.zero_grad()
        self.inv_dyn_opt.zero_grad()
        loss.backward()
        self.action_ae_opt.step()
        self.task_cls_opt.step()
        self.inv_dyn_opt.step()
        
        return metrics

    def update(self, replay_iter, step, stage, use_sensor):
        # for stage 2 and 3, we use the same functions but with different hyperparameters
        assert stage in ["BC", "DAPG"]
        metrics = dict()
        self.action_ae.eval()

        if stage == "BC":
            update_encoder = self.stage2_update_encoder
            stddev = self.stage2_std
            conservative_loss_weight = self.cql_weight
            bc_weight = self.stage2_bc_weight
            update_autoencoder = self.stage2_update_autoencoder

        if stage == "DAPG":
            if step % self.update_every_steps != 0:
                return metrics
            update_encoder = self.stage3_update_encoder
            update_autoencoder = self.stage3_update_autoencoder
            stddev = utils.schedule(self.stddev_schedule, step)
            conservative_loss_weight = 0

            # compute stage 3 BC weight
            bc_data_per_iter = 40000
            i_iter = step // bc_data_per_iter
            bc_weight = self.stage3_bc_lam0 * self.stage3_bc_lam1 ** i_iter

        # batch data
        batch = next(replay_iter)
        if use_sensor: # TODO might want to...?
            obs, action, reward, discount, next_obs, obs_sensor, obs_sensor_next = utils.to_torch(batch, self.device)
        else:
            obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
            obs_sensor, obs_sensor_next = None, None

        # augment
        if self.use_data_aug:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        else:
            obs = obs.float()
            next_obs = next_obs.float()

        # encode
        if update_encoder:
            obs = self.encoder(obs)
        else:
            with torch.no_grad():
                obs = self.encoder(obs)

        with torch.no_grad():
            next_obs = self.encoder(next_obs)
            act_rep = self.action_ae.encode(action)[0]
        if update_autoencoder:
            act_rec = self.action_ae.forward(action)[0]
        else:
            with torch.no_grad():
                act_rec = self.action_ae.forward(action)[0]
        
        loss_rec_rl = F.mse_loss(act_rec, action)
        metrics['loss_rec_rl'] = F.mse_loss(act_rec, action).item()
        if update_autoencoder:
            self.action_ae_opt.zero_grad()
            loss_rec_rl.backward()
            self.action_ae_opt.step()
            

        # concatenate obs with additional sensor observation if needed
        obs_combined = torch.cat([obs, obs_sensor], dim=1) if obs_sensor is not None else obs
        obs_next_combined = torch.cat([next_obs, obs_sensor_next], dim=1) if obs_sensor_next is not None else next_obs

        # update critic
        metrics.update(self.update_critic(obs_combined, act_rep, reward, discount, obs_next_combined,
                                               stddev, update_encoder, conservative_loss_weight))

        # update actor, following previous works, we do not use actor gradient for encoder update
        if self.policy_output_type == "action":
            metrics.update(self.update_actor(obs_combined.detach(), action, stddev, bc_weight,
                                                  self.pretanh_penalty, self.pretanh_threshold))
        else:
            metrics.update(self.update_actor(obs_combined.detach(), act_rep, stddev, bc_weight,
                                              self.pretanh_penalty, self.pretanh_threshold))

        metrics['batch_reward'] = reward.mean().item()

        # update critic target networks
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, stddev, update_encoder, conservative_loss_weight):
        metrics = dict()
        batch_size = obs.shape[0]

        """
        STANDARD Q LOSS COMPUTATION:
        - get standard Q loss first, this is the same as in any other online RL methods
        - except for the safe Q technique, which controls how large the Q value can be
        """
        with torch.no_grad():
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            if self.policy_output_type == 'action': 
                next_action = self.action_ae.encode(next_action)[0]
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

            if self.safe_q_target_factor < 1:
                target_Q[target_Q > (self.q_threshold + 1)] = self.q_threshold + (target_Q[target_Q > (self.q_threshold+1)] - self.q_threshold) ** self.safe_q_target_factor

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        """
        CONSERVATIVE Q LOSS COMPUTATION:
        - sample random actions, actions from policy and next actions from policy, as done in CQL authors' code
          (though this detail is not really discussed in the CQL paper)
        - only compute this loss when conservative loss weight > 0
        """
        if conservative_loss_weight > 0:
            random_actions = (torch.rand((batch_size * self.cql_n_random, self.act_dim), device=self.device) - 0.5) * 2
            random_actions = self.action_ae.encode(random_actions)[0]

            dist = self.actor(obs, stddev)
            current_actions = dist.sample(clip=self.stddev_clip)

            dist = self.actor(next_obs, stddev)
            next_current_actions = dist.sample(clip=self.stddev_clip)
            
            if self.policy_output_type == 'action': 
                current_actions = self.action_ae.encode(current_actions)[0]
                next_current_actions = self.action_ae.encode(next_current_actions)[0]

            # now get Q values for all these actions (for both Q networks)
            obs_repeat = obs.unsqueeze(1).repeat(1, self.cql_n_random, 1).view(obs.shape[0] * self.cql_n_random,
                                                                               obs.shape[1])

            Q1_rand, Q2_rand = self.critic(obs_repeat,
                                           random_actions)  # TODO might want to double check the logic here see if the repeat is correct
            Q1_rand = Q1_rand.view(obs.shape[0], self.cql_n_random)
            Q2_rand = Q2_rand.view(obs.shape[0], self.cql_n_random)

            Q1_curr, Q2_curr = self.critic(obs, current_actions)
            Q1_curr_next, Q2_curr_next = self.critic(obs, next_current_actions)

            # now concat all these Q values together
            Q1_cat = torch.cat([Q1_rand, Q1, Q1_curr, Q1_curr_next], 1)
            Q2_cat = torch.cat([Q2_rand, Q2, Q2_curr, Q2_curr_next], 1)

            cql_min_q1_loss = torch.logsumexp(Q1_cat / self.cql_temp,
                                              dim=1, ).mean() * conservative_loss_weight * self.cql_temp
            cql_min_q2_loss = torch.logsumexp(Q2_cat / self.cql_temp,
                                              dim=1, ).mean() * conservative_loss_weight * self.cql_temp

            """Subtract the log likelihood of data"""
            conservative_q_loss = cql_min_q1_loss + cql_min_q2_loss - (Q1.mean() + Q2.mean()) * conservative_loss_weight
            critic_loss_combined = critic_loss + conservative_q_loss
        else:
            critic_loss_combined = critic_loss

        # logging
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # if needed, also update encoder with critic loss
        if update_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss_combined.backward()
        self.critic_opt.step()
        if update_encoder:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, action, stddev, bc_weight, pretanh_penalty, pretanh_threshold):
        metrics = dict()

        """
        get standard actor loss
        """
        dist, pretanh = self.actor.forward_with_pretanh(obs, stddev)
        current_action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(current_action).sum(-1, keepdim=True)
        if self.policy_output_type == 'action':
            current_action = self.action_ae.encode(current_action)[0]
        Q1, Q2 = self.critic(obs, current_action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()

        """
        add BC loss
        """
        if bc_weight > 0:
            # get mean action with no action noise (though this might not be necessary)
            stddev_bc = 0
            dist_bc = self.actor(obs, stddev_bc)
            current_mean_action = dist_bc.sample(clip=self.stddev_clip)
            actor_loss_bc = F.mse_loss(current_mean_action, action) * bc_weight
        else:
            actor_loss_bc = torch.FloatTensor([0]).to(self.device)

        """
        add pretanh penalty (might not be necessary for Adroit)
        """
        pretanh_loss = 0
        if pretanh_penalty > 0:
            pretanh_loss = pretanh.abs() - pretanh_threshold
            pretanh_loss[pretanh_loss < 0] = 0
            pretanh_loss = (pretanh_loss ** 2).mean() * pretanh_penalty

        """
        combine actor losses and optimize
        """
        actor_loss_combined = actor_loss + actor_loss_bc + pretanh_loss

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_combined.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_loss_bc'] = actor_loss_bc.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        metrics['abs_pretanh'] = pretanh.abs().mean().item()
        metrics['max_abs_pretanh'] = pretanh.abs().max().item()

        return metrics


class Identity(nn.Module):
    def __init__(self, input_placeholder=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Stage3ShallowEncoder(nn.Module):
    def __init__(self, obs_shape, n_channel):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = n_channel * 35 * 35

        self.n_input_channel = obs_shape[0]
        self.conv1 = nn.Conv2d(obs_shape[0], n_channel, 3, stride=2)
        self.conv2 = nn.Conv2d(n_channel, n_channel, 3, stride=1)
        self.conv3 = nn.Conv2d(n_channel, n_channel, 3, stride=1)
        self.conv4 = nn.Conv2d(n_channel, n_channel, 3, stride=1)
        self.relu = nn.ReLU(inplace=True)

        # TODO here add prediction head so we can do contrastive learning...

        self.apply(utils.weight_init)
        self.normalize_op = transforms.Normalize((0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225))

        self.compress = nn.Sequential(nn.Linear(self.repr_dim, 50), nn.LayerNorm(50), nn.Tanh())
        self.pred_layer = nn.Linear(50, 50, bias=False)

    def transform_obs_tensor_batch(self, obs):
        # transform obs batch before put into the pretrained resnet
        # correct order might be first augment, then resize, then normalize
        # obs = F.interpolate(obs, size=self.pretrained_model_input_size)
        new_obs = obs / 255.0 - 0.5
        # new_obs = self.normalize_op(new_obs)
        return new_obs

    def _forward_impl(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

    def forward(self, obs):
        o = self.transform_obs_tensor_batch(obs)
        h = self._forward_impl(o)
        h = h.view(h.shape[0], -1)
        return h

    def get_anchor_output(self, obs, actions=None):
        # typically go through conv and then compression layer and then a mlp
        # used for UL update
        conv_out = self.forward(obs)
        compressed = self.compress(conv_out)
        pred = self.pred_layer(compressed)
        return pred, conv_out

    def get_positive_output(self, obs):
        # typically go through conv, compression
        # used for UL update
        conv_out = self.forward(obs)
        compressed = self.compress(conv_out)
        return compressed

class Encoder(nn.Module):
    def __init__(self, obs_shape, n_channel):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = n_channel * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], n_channel, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(n_channel, n_channel, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(n_channel, n_channel, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(n_channel, n_channel, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 1
        self.repr_dim = obs_shape[0]

    def forward(self, obs):
        return obs