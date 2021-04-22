import torch
from torch import nn, optim
from util import RunningAverage

class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        fc1 = nn.Linear(input_size, hidden_size)
        relu1 = nn.ReLU()
        fc2 = nn.Linear(hidden_size, hidden_size)
        relu2 = nn.ReLU()
        logit = nn.Linear(hidden_size, 1)
        sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        logit = self.logit(x)
        prob = self.sigmoid(logit)
        return logit, prob

def train(data_loader, input_size, epoch):
    classifier = Classifier(input_size)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-3)
    # TODO: Is BCEWithLogitsLoss same with tf.nn.sigmoid_cross_entropy_with_logits?
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in epoch:
        for data_batch, label_batch in data_loader:
            data_batch = data_batch.cuda()
            label_batch = label_batch.cuda()

            # forward
            logit, prob = classifier(data_batch)
            loss = loss_fn(logit, label_batch)

            # backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return classifier

def estimate_divergence(data_loader, classifier):
    eps = 1e-8
    div = RunningAverage()
    for joint, marginal in data_loader:
        _, prob_joint = classifier(joint)
        ratio_joint = (prob_joint+eps) / (1-prob_joint-eps)
        log_joint = torch.log(torch.abs(ratio_joint))
        _, prob_marginal = classifier(marginal)
        ratio_marginal = (prob_marginal+eps) / (1-prob_marginal-eps)
        log_marginal = torch.log(torch.abs(ratio_marginal))
        # TODO: spliting data into batches makes the result different from Formula (3) in original paper.
        div.update(torch.mean(log_joint) - torch.logsumexp(log_marginal, 0) + torch.log(marginal.shape[0]), marginal.shape[0])
    return div.value()
