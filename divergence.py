import torch
from torch import nn, optim
from util import RunningAverage


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.logit = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        logit = self.logit(x)
        prob = self.sigmoid(logit)
        return logit, prob


class Estimator:
    def __init__(self, input_size, hidden_size=64):
        self.classifier = Classifier(input_size, hidden_size)
        self.optimizer = optim.Adam(
            self.classifier.parameters(), lr=1e-3, weight_decay=1e-3)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def train(self, data_loader, epoch=20):
        for _ in range(epoch):
            for data_batch, label_batch in data_loader:
                self.train_batch(data_batch, label_batch)

    def train_batch(self, batch, label):
        '''Train a batch

        Keyword arguments:
        batch -- stack of joint and marginal, in shape [2*B, N]
        label -- stack of ones and zeros, in shape [2*B, 1]
        '''
        if torch.cuda.is_available():
            batch = batch.cuda()
            label = label.cuda()
        # forward
        logit, _ = self.classifier(batch)
        loss = self.loss_fn(logit, label)

        # backward and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def estimate_divergence(self, data_loader):
        '''Estimate divergence

        Keyword arguments:
        data_loader -- iterable of (joint, marginal)
        '''
        eps = 1e-8
        div = RunningAverage()
        for joint, marginal in data_loader:
            _, prob_joint = self.classifier(joint)
            ratio_joint = (prob_joint+eps) / (1-prob_joint-eps)
            log_joint = torch.log(torch.abs(ratio_joint))
            _, prob_marginal = self.classifier(marginal)
            ratio_marginal = (prob_marginal+eps) / (1-prob_marginal-eps)
            log_marginal = torch.log(torch.abs(ratio_marginal))
            # TODO: spliting data into batches makes the result different
            # from Formula (3) in original paper.
            div.update(torch.mean(log_joint) - torch.logsumexp(log_marginal, 0)
                       + torch.log(torch.tensor(marginal.shape[0])), marginal.shape[0])
        return div.value()
