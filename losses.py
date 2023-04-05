# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F

class DistillLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 alpha=0.5, tau=1.0, print_mode=True):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.count = 0
        self.print_mode = print_mode
        self.base_loss = 0
        self.distill_loss = 0

        self.alpha = alpha
        self.tau = tau

        print('alpha in distillation loss: ', alpha, 'tau in distillation loss: ', tau)

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        # pred, token_pred, out_pred_score = outputs
        outputs, outputs_kd = outputs

        base_loss = self.base_criterion(outputs, labels)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        T = self.tau
        distill_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=-1),
                F.log_softmax(teacher_outputs / T, dim=-1),
                reduction='batchmean',
                log_target=True
            )

        loss = (1-self.alpha) * base_loss + self.alpha * distill_loss
        loss_part = []

        if self.print_mode:
            self.base_loss += base_loss.item()
            self.distill_loss += distill_loss.item()
            loss_part.append(base_loss)
            loss_part.append(distill_loss)
            self.count += 1
            if self.count == 100:
                print('loss info: base_loss=%.4f, distill_loss=%.4f' % (self.base_loss / 100, self.distill_loss / 100))
                self.count = 0
                self.base_loss = 0
                self.distill_loss = 0
        return loss, loss_part