import torch
import torch.nn.functional as F
from torch import nn


class Confidence(nn.Module):
    def __init__(self):
        super(Confidence, self).__init__()

    def forward(self, outputs):
        outputs = F.softmax(outputs, dim=1)
        log_outputs = torch.log(outputs + 1e-8)
        loss = torch.sum(-outputs * log_outputs, dim=-1)
        loss = loss.mean()

        return loss


class Focal_loss(nn.Module):
    def __init__(self, num_class, gamma=1):
        super(Focal_loss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.alpha = torch.ones(self.num_class, 1)

    def forward(self, outputs, labels):
        outputs = F.softmax(outputs, dim=1)  # 这里看情况选择，如果之前softmax了，后续就不用了
        labels = labels.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha.to(outputs.device)
        gamma = self.gamma

        idx = labels.cpu().long()
        one_hot_key = torch.FloatTensor(labels.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key = one_hot_key.to(outputs.device)

        pt = (one_hot_key * outputs).sum(1) + epsilon
        log_pt = pt.log()
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * log_pt
        loss = loss.mean()

        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, num_cls, smoothing=0.02):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.num_cls = num_cls

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        logprobs = F.log_softmax(x, dim=-1)
        target = F.one_hot(target, self.num_cls)
        target = (1.0 - self.smoothing) * target + self.smoothing / self.num_cls
        target = torch.clamp(target.float(), min=self.smoothing / (self.num_cls - 1), max=1.0 - self.smoothing)
        loss = -1 * torch.sum(target * logprobs, 1)

        return loss.mean()


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class _ECE_EM_loss(nn.Module):

    def __init__(self, num_bins, device):

        super(_ECE_EM_loss, self).__init__()
        self.num_bins = num_bins
        self.device = device

    def forward(self, logits, labels):

        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        sorted_confidence, _ = torch.sort(confidences)
        sorted_confidence = torch.reshape(sorted_confidence, (1, -1))
        # ten bins
        index_list = [i / self.num_bins for i in range(1, self.num_bins)]
        index_list = torch.tensor(index_list)
        index_list = index_list.to(self.device)
        # bin boundaries
        boundaries = torch.quantile(sorted_confidence, index_list)
        # bin start
        start = torch.tensor([0])
        start = start.to(self.device)
        # bin end
        end = torch.tensor([1])
        end = end.to(self.device)
        # bins
        bins = torch.cat([start, boundaries, end])
        # calculate gaps
        indices = torch.bucketize(confidences, bins)
        bin_accuracies = torch.zeros(self.num_bins, dtype=torch.float64)
        bin_confidences = torch.zeros(self.num_bins, dtype=torch.float64)
        for b in range(self.num_bins):
            selected = torch.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_acc_bool = labels[selected] == predictions[selected]
                bin_accuracies[b] = torch.mean(torch.tensor([1.0 if value else 0.0 for value in bin_acc_bool]))
                bin_confidences[b] = torch.mean(confidences[selected])
        gaps = bin_confidences - bin_accuracies
        ece = torch.mean(torch.abs(gaps))

        return ece


class _BS_loss(nn.Module):

    def __init__(self, num_cls):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_BS_loss, self).__init__()
        self.num_class = num_cls

    def forward(self, outputs, labels):
        softmax_outputs = outputs.softmax(dim=-1)
        labels = labels.view(-1, 1)

        idx = labels.cpu().long()
        one_hot_key = torch.FloatTensor(labels.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key = one_hot_key.to(outputs.device)

        loss = torch.mean(torch.sum((softmax_outputs - one_hot_key) ** 2, dim=-1))

        return loss

def _compute_entropy(outputs):

    softmax_outputs = outputs.softmax(dim=-1)
    log_outputs = torch.log(softmax_outputs + 1e-8)
    entropy = torch.sum(-softmax_outputs * log_outputs, dim=-1)

    return entropy


class UA_CP_loss(nn.Module):
    def __init__(self, num_cls):
        super(UA_CP_loss, self).__init__()
        self.num_cls = num_cls

    def forward(self, outputs, labels, trade):
        outputs = F.softmax(outputs, dim=1)
        log_outputs = torch.log(outputs + 1e-8)
        target = F.one_hot(labels, self.num_cls)

        entropy_loss = -1 * torch.sum(target * log_outputs, 1)

        CP_loss = torch.sum(-outputs * log_outputs, dim=-1)
        CP_loss = -1 * trade * CP_loss

        loss = entropy_loss + CP_loss

        return loss.mean()

class LogitNormLoss(nn.Module):

    def __init__(self, device, t=0.2):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)









