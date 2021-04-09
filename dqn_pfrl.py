import pfrl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QFunction(torch.nn.Module):

    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, stride=2)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, stride=2)))
        linear_input_size = convw * convh * 16

        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.reshape(x.size(0), -1))
        return pfrl.action_value.DiscreteActionValue(x)


class DQN_PFRL_ACTOR:

    def __init__(self, screen_height, screen_width, n_channels, n_actions, eps_start=1,
                 eps_end=0.01, eps_test=0.001, eps_decay=250_000, eps_const=None, gamma=0.999,
                 batch_size=32, target_update=2000, learning_rate=0.00025, buffer_size=1_000_000,
                 update_interval=1, update_start=5000, env=None):
        self.n_actions = n_actions
        self.env = env
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps_threshold = eps_start
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_interval = update_interval
        self.update_start = update_start

        self.input_shape = (screen_height, screen_width, n_channels)
        # self.q_function = QFunction(screen_height, screen_width, n_actions)
        self.q_function = pfrl.q_functions.DuelingDQN(n_actions, n_channels)
        self.optimizer = optim.RMSprop(
            self.q_function.parameters(), lr=learning_rate)
        self.explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
            eps_start, eps_end, eps_decay, random_action_func=env.action_space.sample)
        self.memory = pfrl.replay_buffers.PrioritizedReplayBuffer(buffer_size)

        self.phi = lambda x: torch.from_numpy(x.transpose((2, 0, 1))).to(device, dtype=torch.float)
        self.gpu = 0

        self.agent = pfrl.agents.DQN(
            self.q_function,
            self.optimizer,
            self.memory,
            self.gamma,
            self.explorer,
            replay_start_size=self.update_start,
            minibatch_size=self.batch_size,
            update_interval=self.update_interval,
            target_update_interval=self.target_update,
            phi=self.phi,
            gpu=self.gpu,
            max_grad_norm=1.,

        )
