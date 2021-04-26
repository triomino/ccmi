import torch
from divergence import Estimator


def make_batch(x, y):
    perm = torch.randperm(y.shape[0])
    return torch.hstack((x, y)), torch.hstack((x, y[perm]))


def estimate(x, y, z, epoch=100):
    perm = torch.randperm(x.shape[0])
    x, y, z = x[perm], y[perm], z[perm]

    # split train, eval
    batch_size = x.shape[0] * 2 // 3
    x, x_eval = torch.split(x, batch_size)
    y, y_eval = torch.split(y, batch_size)
    z, z_eval = torch.split(z, batch_size)

    # xyz batch data
    data_batch = torch.vstack(make_batch(x, torch.hstack((y, z))))
    label_batch = torch.vstack((torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))))

    # estimate (x|yz)
    estimator_xyz = Estimator(x.shape[1] + y.shape[1] + z.shape[1])
    for _ in range(epoch):
        estimator_xyz.train_batch(data_batch, label_batch)
    xyz_i = estimator_xyz.estimate_divergence([make_batch(x_eval, torch.hstack((y_eval, z_eval)))])

    # estimate (x|z)
    data_batch = torch.vstack(make_batch(x, z))
    estimator_xz = Estimator(x.shape[1] + z.shape[1])
    for _ in range(epoch):
        estimator_xz.train_batch(data_batch, label_batch)
    xz_i = estimator_xz.estimate_divergence([make_batch(x_eval, z_eval)])

    return xyz_i - xz_i


if __name__ == '__main__':
    # independent
    b = 512*3//2
    x = torch.randn((b, 15))
    y = torch.randn((b, 10))
    z = torch.randn((b, 1))
    res = [estimate(x, y, z).item() for _ in range(20)]
    print(res)
    print(sum(res) / len(res))

    # dependent
    x = z * 10
    y = z * 1.5
    res = [estimate(x, y, z).item() for _ in range(20)]
    print(res)
    print(sum(res) / len(res))
