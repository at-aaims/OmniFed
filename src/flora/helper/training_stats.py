# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch


class AverageMeter(object):
    """Computes and stores the average and current value for model loss, accuracy etc."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topK_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def test_img_accuracy(epoch, device, model, test_loader, loss_fn, iteration):
    model.eval()
    with torch.no_grad():
        test_loss, top1acc, top5acc, top10acc = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)
            loss = loss_fn(output, label)
            topKaccuracy = topK_accuracy(output=output, target=label, topk=(1, 5, 10))
            top1acc.update(topKaccuracy[0], input.size(0))
            top5acc.update(topKaccuracy[1], input.size(0))
            top10acc.update(topKaccuracy[2], input.size(0))
            test_loss.update(loss.item(), input.size(0))

        logging.info(
            f"Logging test_metrics iteration {iteration} epoch {epoch} test_loss {test_loss.avg} top1_acc "
            f"{top1acc.avg.cpu().numpy().item()} top5_acc {top5acc.avg.cpu().numpy().item()} top10_acc "
            f"{top10acc.avg.cpu().numpy().item()}"
        )
        model.train()


def train_img_accuracy(
    epoch, iteration, input, label, output, loss, train_loss, top1acc, top5acc, top10acc
):
    with torch.no_grad():
        topKaccuracy = topK_accuracy(output=output, target=label, topk=(1, 5, 10))
        top1acc.update(topKaccuracy[0], input.size(0))
        top5acc.update(topKaccuracy[1], input.size(0))
        top10acc.update(topKaccuracy[2], input.size(0))
        train_loss.update(loss.item(), input.size(0))
        logging.info(
            f"Logging train_metrics iteration {iteration} epoch {epoch} train_loss {train_loss.avg} top1_acc "
            f"{top1acc.avg.cpu().numpy().item()} top5_acc {top5acc.avg.cpu().numpy().item()} top10_acc "
            f"{top10acc.avg.cpu().numpy().item()}"
        )
