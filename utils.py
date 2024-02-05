import torch

def hinge_loss(output, target):
    num_classes = output.size(1)
    correct_scores = output.gather(1, target.view(-1, 1)).expand_as(output)
    margin = 1.0  # You can adjust this margin
    loss = output - correct_scores + margin
    loss.clamp_(min=0)
    loss.scatter_(1, target.view(-1, 1), 0)
    return loss.mean()
