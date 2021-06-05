import torch
import torch.nn as nn
import torch.optim as optim
import pfrl
from pfrl.policies import SoftmaxCategoricalHead
# from pfrl.experiments.hooks import StepHook
# from pfrl import experiments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


def phi(x):
    # Feature extractor
    return torch.from_numpy(x.transpose((2, 0, 1))).to(device, dtype=torch.float32) / 255


class PolicyPPO(torch.nn.Module):

    def __init__(self, h, w, obs_n_channels, n_actions):
        super().__init__()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, stride=2)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, stride=2)))
        linear_input_size = convw * convh * 16

        self.model = nn.Sequential(
            lecun_init(nn.Conv2d(obs_n_channels, 8, kernel_size=3, stride=2)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            lecun_init(nn.Conv2d(8, 16, kernel_size=3, stride=1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            lecun_init(nn.Conv2d(16, 16, kernel_size=3, stride=1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            pfrl.nn.Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(linear_input_size, n_actions), 1e-2),
                    SoftmaxCategoricalHead(),
                ),
                lecun_init(nn.Linear(linear_input_size, 1)),
            )
        )

    def forward(self, *args):
        return self.model(*args)


def PPO_PFRL_ACTOR(screen_height, screen_width, n_channels, n_actions, gamma=0.99, batch_size=32, update_interval=1024,
                   learning_rate=0.00025, learning_rate_decay_steps=1e7, epochs=10, clip_eps=0.2, entropy_coef=0.01):
    policy = PolicyPPO(screen_height, screen_width, n_channels, n_actions)
    optimizer = optim.Adam(policy.model.parameters(), lr=learning_rate, eps=1e-8)

    return pfrl.agents.PPO(
        policy.model,
        optimizer,
        clip_eps=clip_eps,
        clip_eps_vf=None,
        entropy_coef=entropy_coef,
        epochs=epochs,
        gamma=gamma,
        gpu=0,
        max_grad_norm=0.5,
        minibatch_size=batch_size,
        phi=phi,
        standardize_advantages=True,
        update_interval=update_interval,
    )
