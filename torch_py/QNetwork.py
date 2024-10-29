from abc import ABC

import torch.nn as nn
import torch


class QNetwork(nn.Module, ABC):
    """Actor (Policy) Model."""

    def __init__(self, state_num: int, state_size: int, action_size: int, seed: int):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.embedding_1 = nn.Parameter(torch.randn(state_num, 128, 4), requires_grad=True)
        self.embedding_2 = nn.Parameter(torch.randn(state_num, 128, 4), requires_grad=True)
        nn.init.xavier_normal_(self.embedding_1)
        nn.init.xavier_normal_(self.embedding_2)

    def forward(self, state):
        if len(state.shape) > 1:
            x = state[:, 0].long()
            y = state[:, 1].long()
        else:
            x = state[0].long()
            y = state[1].long()

        x_h = self.embedding_1[x.detach()]
        y_h = self.embedding_2[y.detach()]

        if len(state.shape) > 1:
            return torch.einsum("bdn, bdn -> bn", x_h, y_h)
        else:
            return torch.einsum("dn, dn -> n", x_h, y_h)
        

if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    net = QNetwork(2, 4, 0).to(device)

    x = torch.tensor([1, 1]).float().unsqueeze(0).to(device)
    #
    # torch.nn.DataParallel(net, device_ids=[0])
    print(net(x))
