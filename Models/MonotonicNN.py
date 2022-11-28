# Credit goes to Antoine Wehenkel for this entire python file.



import torch
import torch.nn as nn
from UMNN import NeuralIntegral
from UMNN import ParallelNeuralIntegral
import math

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers, n_out=1):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [n_out]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.


class OneDimensionnalNF(nn.Module):
    def __init__(self, in_d, hidden_layers, nb_steps=200, n_out=1, dev="cpu"):
        super(OneDimensionnalNF, self).__init__()
        self.device = dev
        self.nb_steps = nb_steps
        self.n_out = n_out
        self.net = MonotonicNN(in_d, hidden_layers, nb_steps=nb_steps, n_out=n_out, dev=dev)
        self.register_buffer("pi", torch.tensor(math.pi))

    '''
    The forward procedure takes as input x which is the variable for which the integration must be made, h are just other conditionning variables.
    It returns the $log(p(x|h; \theta))$.
    '''
    def forward(self, x, h):
        x0 = torch.zeros(x.shape).to(self.device)
        out = self.net.net(h)
        offset = out[:, :self.n_out]
        scaling = torch.exp(out[:, self.n_out:])
        jac = scaling * self.net.integrand(x, h)
        z = scaling*ParallelNeuralIntegral.apply(x0, x, self.net.integrand, _flatten(self.net.integrand.parameters()), h, self.nb_steps) + offset
        z.clamp_(-10., 10.)
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2)
        ll = log_prob_gauss + torch.log(jac + 1e-10)
        return ll

    def expectation(self, h, x_func, min=-10, max=10, npts=1000):
        # Using first order Euler method .
        b_size = h.shape[0]
        n_out = self.n_out
        dx = (max-min)/(npts - 1)
        emb_size = h.shape[1]

        x = torch.arange(min, max+(max-min)/(npts - 1), dx).to(h.device)
        npts = x.shape[0]
        zero_idx = torch.argmin(x**2).item()

        out = self.net.net(h)
        offset = out[:, :self.n_out].unsqueeze(1).expand(b_size, npts, n_out)
        scaling = torch.exp(out[:, self.n_out:]).unsqueeze(1).expand(b_size, npts, n_out)

        h_values = h.unsqueeze(1).expand(b_size, npts, emb_size).reshape(-1, emb_size)
        x_values = x.unsqueeze(0).expand(b_size, npts).reshape(-1, 1)

        f_values = self.net.integrand(x_values, h_values)
        f_values = f_values.reshape(b_size, npts, n_out) * scaling

        z = (dx * f_values.cumsum(1))
        z = (z - z[:, [zero_idx], :].expand(-1, npts, -1)) + offset
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2)
        ll = log_prob_gauss + torch.log(f_values + 1e-10)


        expectations = (x_func(x).unsqueeze(0).unsqueeze(2).expand(b_size, npts, n_out) * torch.exp(ll)).sum(1) * dx

        return expectations

class MonotonicNN(nn.Module):
    '''
    in_d : The total number of inputs
    hidden_layers : a list a the number of neurons, to be used by a network that compresses the non-monotonic variables and by the integrand net.
    nb_steps : Number of integration steps
    n_out : the number of output (each output will be monotonic w.r.t one variable)
    '''
    def __init__(self, in_d, hidden_layers, nb_steps=200, n_out=1, dev="cpu"):
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers, n_out)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2 * n_out]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()
        self.net = nn.Sequential(*self.net)
        self.device = dev
        self.nb_steps = nb_steps
        self.n_out = n_out

    '''
    The forward procedure takes as input x which is the variable for which the integration must be made, h are just other conditionning variables.
    '''
    def forward(self, x, h, only_derivative=False):
        x0 = torch.zeros(x.shape).to(self.device)
        out = self.net(h)
        offset = out[:, :self.n_out]
        scaling = torch.exp(out[:, self.n_out:])
        if only_derivative:
            return scaling * self.integrand(x, h)
        return scaling*ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset




    '''
    The inverse procedure takes as input y which is the variable for which the inverse must be computed, h are just other conditionning variables.
    One output per n_out.
    y should be a scalar.
    '''
    def inverse(self, y, h, min=-10, max=10, nb_iter=10):
        idx = (torch.arange(0, self.n_out**2, self.n_out + 1).view(1, -1) + torch.arange(0, (self.n_out**2)*y.shape[0], self.n_out**2).view(-1, 1)).view(-1)
        h = h.unsqueeze(1).expand(-1, self.n_out, -1).contiguous().view(y.shape[0]*self.n_out, -1)

        # Old inversion by binary search
        x_max = torch.ones(y.shape[0], self.n_out).to(y.device) * max
        x_min = torch.ones(y.shape[0], self.n_out).to(y.device) * min
        y_max = self.forward(x_max.view(-1, 1), h).view(-1)[idx].view(-1, self.n_out)
        y_min = self.forward(x_min.view(-1, 1), h).view(-1)[idx].view(-1, self.n_out)

        for i in range(nb_iter):
            x_middle = (x_max + x_min) / 2
            y_middle = self.forward(x_middle.view(-1, 1), h).view(-1)[idx].view(-1, self.n_out)
            left = (y_middle > y).float()
            right = 1 - left
            x_max = left * x_middle + right * x_max
            x_min = right * x_middle + left * x_min
            y_max = left * y_middle + right * y_max
            y_min = right * y_middle + left * y_min
        return (x_max + x_min) / 2

    def expectation(self, h, x_func, out_deriv, min=-10, max=10, npts=1000):
        # Using first order Euler method .
        b_size = h.shape[0]
        n_out = self.n_out
        dx = (max-min)/(npts - 1)
        emb_size = h.shape[1]

        x = torch.arange(min, max+(max-min)/(npts - 1), dx).to(h.device)
        npts = x.shape[0]
        zero_idx = torch.argmin(x**2).item()

        out = self.net(h)
        offset = out[:, :self.n_out].unsqueeze(1).expand(b_size, npts, n_out)
        scaling = torch.exp(out[:, self.n_out:]).unsqueeze(1).expand(b_size, npts, n_out)

        h_values = h.unsqueeze(1).expand(b_size, npts, emb_size).reshape(-1, emb_size)
        x_values = x.unsqueeze(0).expand(b_size, npts).reshape(-1, 1)

        f_values = self.integrand(x_values, h_values)
        f_values = f_values.reshape(b_size, npts, n_out) * scaling

        F_values = (dx * f_values.cumsum(1))
        F_values = (F_values - F_values[:, [zero_idx], :].expand(-1, npts, -1)) + offset
        corrected_F_values = out_deriv(F_values)

        expectations = (x_func(x).unsqueeze(0).unsqueeze(2).expand(b_size, npts, n_out) * f_values * corrected_F_values).sum(1) * dx

        return expectations
