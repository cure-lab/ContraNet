import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from advertorch.attacks.base import Attack, LabelMixin
from advertorch.utils import tanh_rescale
from advertorch.utils import torch_arctanh
from advertorch.utils import clamp
from advertorch.utils import to_one_hot
from advertorch.utils import replicate_input
from advertorch.attacks.utils import is_successful

CARLINI_L2DIST_UPPER = 1e10
CARLINI_LinfDIST_UPPER = 1
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
NUM_CHECKS = 10

c_con = 0.


class CarliniWagnerLinfAttack(Attack, LabelMixin):
    """
    The Carlini and Wagner Linfity Attack, https://arxiv.org/abs/1608.04644

    :param predict: forward pass function.
    :param num_classes: number of clasess.
    :param confidence: confidence of the adversarial examples.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm
    :param binary_search_steps: number of binary search times to find the
        optimum
    :param max_iterations: the maximum number of iterations
    :param abort_early: if set to true, abort early if getting stuck in local
        min
    :param initial_const: initial value of the constant c
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function
    """

    def __init__(self, predict, num_classes, confidence=0,
                 targeted=False, learning_rate=0.01,
                 binary_search_steps=9, max_iterations=10000,
                 abort_early=True, initial_const=1e-3,
                 clip_min=0., clip_max=1., loss_fn=None, normalize_fn=None,
                 adaptive_evi=False, evi_train_median=None,
                 adaptive_con=False, con_train_median=None):
        """Carlini Wagner L2 Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently do not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        loss_fn = None

        super(CarliniWagnerLinfAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.confidence = confidence
        self.initial_const = initial_const
        self.num_classes = num_classes
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        self.targeted = targeted
        self.normalize_fn = normalize_fn

        self.adaptive_evi = adaptive_evi
        self.evi_train_median = evi_train_median
        self.adaptive_con = adaptive_con
        self.con_train_median = con_train_median

    def _loss_fn(self, output, y_onehot, linfdistsq, const):
        # TODO: move this out of the class and make this the default loss_fn
        #   after having targeted tests implemented
        real = (y_onehot * output).sum(dim=1)

        # TODO: make loss modular, write a loss class
        other, label_o = ((1.0 - y_onehot) * output - (y_onehot * TARGET_MULT)
                          ).max(1)
        label_0 = F.one_hot(label_o, num_classes=self.num_classes)
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if self.adaptive_con:
            c = c_con
        else:
            c = self.confidence

        if self.targeted:
            loss1 = clamp(other - real + c, min=0.)
        else:
            loss1 = clamp(real - other + c, min=0.)

        # adaptive loss for evading evidence detector
        if self.adaptive_evi:
            loss1 += clamp(self.evi_train_median - output.logsumexp(dim=1), min=0.)
        if self.adaptive_con:
            l = F.softmax(output, dim=1) * label_0
            loss1 = clamp(self.con_train_median - l.sum(dim=1), min=0.)

        loss2 = (linfdistsq).sum()
        loss1 = torch.sum(const * loss1)
        loss = loss1 + loss2
        return loss

    def _is_successful(self, output, label, is_logits, pred_labels=None):
        # determine success, see if confidence-adjusted logits give the right
        #   label

        if is_logits:
            output = output.detach().clone()
            if self.targeted:
                output[torch.arange(len(label)).long(),
                       label] -= self.confidence
            else:
                output[torch.arange(len(label)).long(),
                       label] += self.confidence
            con, pred = F.softmax(output, dim=1).max(1)
            evidence = output.logsumexp(dim=1)
        else:
            pred = pred_labels
            if pred == INVALID_LABEL:
                return pred.new_zeros(pred.shape).byte()
            con = F.softmax(output, dim=0).max(0)[0]
            evidence = output.logsumexp(dim=0)

        if self.adaptive_evi:
            return is_successful(
                pred, label, self.targeted) & (
                evidence > self.evi_train_median)
        elif self.adaptive_con:
            return is_successful(
                pred, label, self.targeted) & (
                con > self.con_train_median)
        else:
            return is_successful(pred, label, self.targeted)

    def _forward_and_update_delta(
            self, optimizer, x_atanh, delta, y_onehot, loss_coeffs):

        optimizer.zero_grad()
        adv = tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)
        transimgs_rescale = tanh_rescale(x_atanh, self.clip_min, self.clip_max)
        if self.normalize_fn == None:
            output = self.predict(adv)
        else:
            output = self.predict(self.normalize_fn(adv))
        linfdistsq, _ = torch.max(
            torch.abs(adv - transimgs_rescale).view(adv.size()[0], -1), dim=1)
        loss = self._loss_fn(output, y_onehot, linfdistsq, loss_coeffs)
        loss.backward()
        optimizer.step()

        return loss.item(), linfdistsq.data, output.data, adv.data

    def _get_arctanh_x(self, x):
        result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
                       min=0., max=1.) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)

    def _update_if_smaller_dist_succeed(
            self, adv_img, labs, output, linfdistsq, batch_size,
            cur_linfdistsqs, cur_labels,
            final_linfdistsqs, final_labels, final_advs,
            cur_output):

        target_label = labs
        output_logits = output
        _, output_label = torch.max(output_logits, 1)

        mask = (linfdistsq < cur_linfdistsqs) & self._is_successful(
            output_logits, target_label, True)

        cur_linfdistsqs[mask] = linfdistsq[mask]  # redundant
        cur_labels[mask] = output_label[mask]
        cur_output[mask, :] = output_logits[mask, :]

        mask = (linfdistsq < final_linfdistsqs) & self._is_successful(
            output_logits, target_label, True)
        final_linfdistsqs[mask] = linfdistsq[mask]
        final_labels[mask] = output_label[mask]
        final_advs[mask] = adv_img[mask]

    def _update_loss_coeffs(
            self, labs, cur_labels, batch_size, loss_coeffs,
            coeff_upper_bound, coeff_lower_bound,
            cur_output):

        # TODO: remove for loop, not significant, since only called during each
        # binary search step
        for ii in range(batch_size):
            cur_labels[ii] = int(cur_labels[ii])
            if self._is_successful(
                    cur_output[ii],
                    labs[ii],
                    False, pred_labels=cur_labels[ii]):
                coeff_upper_bound[ii] = min(
                    coeff_upper_bound[ii], loss_coeffs[ii])

                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (
                        coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
            else:
                coeff_lower_bound[ii] = max(
                    coeff_lower_bound[ii], loss_coeffs[ii])
                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (
                        coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
                else:
                    loss_coeffs[ii] *= 10

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)
        x = replicate_input(x)
        batch_size = len(x)
        coeff_lower_bound = x.new_zeros(batch_size)
        coeff_upper_bound = x.new_ones(batch_size) * CARLINI_COEFF_UPPER
        loss_coeffs = torch.ones_like(y).float() * self.initial_const
        final_linfdistsqs = [CARLINI_LinfDIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = x
        x_atanh = self._get_arctanh_x(x)
        y_onehot = to_one_hot(y, self.num_classes).float()

        final_linfdistsqs = torch.FloatTensor(final_linfdistsqs).to(x.device)
        final_labels = torch.LongTensor(final_labels).to(x.device)

        # Start binary search
        for outer_step in range(self.binary_search_steps):
            delta = nn.Parameter(torch.zeros_like(x))
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            cur_linfdistsqs = [CARLINI_LinfDIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size
            cur_linfdistsqs = torch.FloatTensor(cur_linfdistsqs).to(x.device)
            cur_labels = torch.LongTensor(cur_labels).to(x.device)
            prevloss = PREV_LOSS_INIT

            # record current output
            cur_output = torch.zeros(
                x.size()[0],
                self.num_classes).float().cuda()

            if (self.repeat and outer_step == (self.binary_search_steps - 1)):
                loss_coeffs = coeff_upper_bound
            for ii in range(self.max_iterations):
                loss, linfdistsq, output, adv_img = \
                    self._forward_and_update_delta(
                        optimizer, x_atanh, delta, y_onehot, loss_coeffs)
                if self.abort_early:
                    if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                        if loss > prevloss * ONE_MINUS_EPS:
                            break
                        prevloss = loss

                self._update_if_smaller_dist_succeed(
                    adv_img, y, output, linfdistsq, batch_size,
                    cur_linfdistsqs, cur_labels,
                    final_linfdistsqs, final_labels, final_advs,
                    cur_output)

            self._update_loss_coeffs(
                y, cur_labels, batch_size,
                loss_coeffs, coeff_upper_bound, coeff_lower_bound,
                cur_output)

        return final_advs
