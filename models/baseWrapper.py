from typing import Tuple
from abc import abstractmethod, ABC
import torch
from torchvision.transforms import Normalize


class BaseWrapper(ABC):
    def __init__(self, net, criterion, cls_norm, targeted=False, x_val_min=0,
                 x_val_max=1):
        self.net = net
        self.criterion = criterion
        self.targeted = targeted
        self.x_val_min = x_val_min
        self.x_val_max = x_val_max
        self.cls_norm = Normalize(*cls_norm)

    @abstractmethod
    def attack(self, x: torch.Tensor, y: torch.Tensor, device: torch.device,
               training=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """perform attack here

        Args:
            x (torch.Tensor): input x
            y (torch.Tensor): true label y
            device (torch.device): device for the model
            training (bool, optional): attack as tringing. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: x_adv, cls_pred
        """
        raise NotImplementedError()

    def adv_by_suss(self, x, y, device, training=False):
        x_adv, pred_y = self.attack(x, y, device, training=training)
        y = y.to(device)
        fail_mask = torch.where(pred_y == y)
        suss_mask = torch.where(pred_y != y)

        if training:
            return x_adv[suss_mask], pred_y[suss_mask], y[suss_mask], \
                x_adv[fail_mask], y[fail_mask]
        else:
            return x_adv[suss_mask], pred_y[suss_mask], \
                x_adv[fail_mask], pred_y[fail_mask]

    def print_stat(self):
        """print statistic info. Defult do nothing.
        """
        return
