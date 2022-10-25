import matplotlib as mpl
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tqdm.notebook

'''
Useful functions for performing Bayesian nonparametric fitting
on the dataset, and for visualizing the data / results.
'''


class BNPFitter:
    def __init__(self, times, a_tau_hp, b_tau_hp, a_gamma_hhp=None,
                 b_gamma_hhp=None, fixed_gamma=None, t_censor=np.nan,
                 do_censoring=False, do_truncation=False):
        '''
        `times`: a (N,M) numpy array of transition times, where
            * np.nan is a placeholder for a bead not tracked;
            * np.inf is a placeholder for a right-censored data point;
            * -np.inf is a placeholder for the loop never closing before the pull.
            It is treated the same as np.nan.
        `a_tau_hp` and `b_tau_hp`: Hyperparameters for the prior on tau. These are
            the shape and scale parameters of an inverse gamma distribution.
        `a_gamma_hhp` and `b_gamma_hhp`: Hyper-hyper-parameters for the hyperprior
            on gamma, which is the concentration parameter of the Dirichlet Process.
        `fixed_gamma`: The concentration parameter of the Dirichlet Process.
        `t_censor`: the length of the experiment. If no transition is observed
            before `t_censor` seconds, it is recorded as np.inf.
        `do_censoring`: whether to use the np.inf labeled points
        `do_truncation`: whether to use a truncated exponential likelihood function

        Note: Either the `fixed_gamma` parameter must be provided, or the
        `a_gamma_hhp` and `b_gamma_hhp` parameters must be provided. In the
        first case, the concentration parameter of the DP will be held fixed.
        In the scond case, it will be sampled from a gamma hyperprior.
        '''
        if fixed_gamma is None:
            if a_gamma_hhp is None or b_gamma_hhp is None:
                raise Exception("Either gamma or its hyperprior must be provided.")

        self.a_tau_hp = a_tau_hp
        self.b_tau_hp = b_tau_hp
        self.a_gamma_hhp = a_gamma_hhp
        self.b_gamma_hhp = b_gamma_hhp 
        self.fixed_gamma  = fixed_gamma 

        self.t_censor = t_censor
        self.do_truncation = do_truncation

        # Clean up data table with how we want to censor/truncate
        ts = self.clean_transition_time_table(times, do_censoring)
        N,M = ts.shape

        # Calculate sufficient statistics
        self.sum_of_lifetimes_per_bead, self.uncensored_events_per_bead = \
            self.calculate_sufficient_statistics(ts, do_censoring)
        self.N = N

    def fit(self, K_max=10, num_iterations=50000, rng_seed=43,
            init_from_equal_classes=False, stick_breaking=False,
            verbose=1):
        '''
        Perform Gibbs sampling for `num_iterations` iterations.

        `K_max`: the maximum number of classes. The actual K at any iteration
            will be 0 < K <= K_max. Note that K_max <= N. Smaller K_max makes
            the code run faster.
        `num_iterations`: number of iterations of Gibbs sampling to run. Note that
            the user should inspect the trajectories and autocorrelation functions
            afterwards to make sure you ran enough iterations for the posterior to
            be well-sampled.
        `rng_seed`: Random number generator seed.
        `verbose`: How verbose to be with output. If >= 1, it displays a progress bar.

        Returns: a pandas DataFrame. Each row is a Monte Carlo sample. Columns are:
            'gamma': The concentration hyperparameter.
            'p_0', 'p_1', ...: The mixture weights of each component
            'tau_0', 'tau_1', ...: The lifetimes of each component
            'n_0', 'n_1', ...: The number of beads in each component
        '''
        self.K_max = K_max

        np.random.seed(rng_seed)
        gamma, taus, log_ps, classes = self.sample_from_prior(init_from_equal_classes, stick_breaking)

        self.traj_columns = self.make_traj_columns()
        trajectory = []

        for t in self.make_iterator(num_iterations, verbose):
            # --- Step 1: Update the class assignments ---
            classes = self.sample_marginal_class_assignments(log_ps, taus)

            # --- Step 2: Update the class probabilities ---
            one_hot_class = np.eye(K_max)[classes]
            class_counts = one_hot_class.sum(axis=0)

            log_ps, iVs = self.sample_log_dirichlet(gamma, class_counts, stick_breaking=stick_breaking)

            # --- Step 3: Update the lifetime estimates for each class ---
            sum_of_lifetimes_per_class = self.sum_of_lifetimes_per_bead @ one_hot_class
            uncensored_events_per_class = self.uncensored_events_per_bead @ one_hot_class

            taus = self.sample_marginal_lifetimes(sum_of_lifetimes_per_class, uncensored_events_per_class,
                                            self.a_tau_hp, self.b_tau_hp)

            # --- Step 4: Update the hyperparameter gamma
            if self.fixed_gamma is None:
                gamma = self.sample_gamma(iVs, self.a_gamma_hhp, self.b_gamma_hhp) 

            # --- Calculate the posterior for logging purposes
            log_prior, log_likelihood, log_posterior = self.calculate_joint_log_posterior(
                taus, log_ps, gamma, one_hot_class, class_counts)

            trajectory.append(
                self.make_trajectory_entry([gamma, log_prior, log_likelihood, log_posterior], np.exp(log_ps), taus, class_counts)
            )

        traj_df = pd.DataFrame(trajectory)
        return self.clean_trajectory(traj_df)


    def clean_transition_time_table(self, times, do_censoring):
        ts = times.copy()
        ts[ts == -np.inf] = np.nan
        if do_censoring:
            ts[ts == np.inf] = self.t_censor
        else:
            ts[ts == np.inf] = np.nan
        return ts.values

    def calculate_sufficient_statistics(self, ts, do_censoring):
        sum_of_lifetimes_per_bead = np.nansum(ts, axis=1)
        total_events_per_bead = np.sum(~np.isnan(ts), axis=1).astype(int)
        if do_censoring:
            censored_events_per_bead = np.sum(ts == self.t_censor, axis=1).astype(int)
        else:
            censored_events_per_bead = np.zeros(ts.shape[0], dtype=int)
        uncensored_events_per_bead = total_events_per_bead - censored_events_per_bead
        return sum_of_lifetimes_per_bead, uncensored_events_per_bead

    def sample_from_prior(self, init_from_equal_classes, stick_breaking):
        taus = 1 / np.random.gamma(self.a_tau_hp, 1 / self.b_tau_hp, size=self.K_max)

        if self.fixed_gamma is not None:
            gamma = self.fixed_gamma
        else:
            gamma = np.random.gamma(self.a_gamma_hhp, 1 / self.b_gamma_hhp)

        if init_from_equal_classes:
            log_ps = - np.log(self.K_max) * np.ones(self.K_max)
        else:
            log_ps,_ = self.sample_log_dirichlet(gamma=np.repeat(gamma, self.K_max), 
                                                 stick_breaking=stick_breaking)
            # For easier label identification, call the largest component the first one
            if(np.all(np.isnan(log_ps))):
                log_ps = -np.inf * np.ones(self.K_max)
                log_ps[0] = 0
            log_ps = np.array(sorted(log_ps)[::-1])

        classes = np.random.choice(self.K_max, size=self.N, p=np.exp(log_ps))

        return gamma, taus, log_ps, classes

    def sample_marginal_class_assignments(self, log_ps, taus):
        # Calculate posterior probability of each class for each bead
        x = log_ps[:,np.newaxis] - np.outer(np.log(taus), self.uncensored_events_per_bead)
        x += np.outer(-1 / taus, self.sum_of_lifetimes_per_bead)
        if self.do_truncation:
            x -= np.outer(np.log(1 - np.exp(-self.t_censor/taus)), self.uncensored_events_per_bead)

        # Convert into normalized class probabilities
        x -= x.max(axis=0)
        x = np.exp(x)
        x /= x.sum(axis=0)

        # Sample from these class probabilities
        class_cumprob = x.cumsum(axis=0)
        classes = (class_cumprob < np.random.random(size=self.N)).sum(axis=0)
        return classes

    def sample_log_dirichlet(self, gamma, class_counts=None, stick_breaking=False):
        if class_counts is None:
            class_counts = np.zeros_like(gamma)
        if stick_breaking:
            return BNPFitter.sample_log_dirichlet_stick_breaking(gamma, class_counts)
        else:
            return BNPFitter.sample_log_dirichlet_stable((gamma / self.K_max) + class_counts), None

    @staticmethod
    def sample_log_dirichlet_stable(alphas):
        '''
        Numerically stable implementation of Dirichlet sampling that returns log x's
        without issues of underflow for small alpha.

        `alphas`: length K array of concentration parameters

        Returns length K array of *log* Dirichlet random variates, summing to 1.

        Thanks stack overflow for this solution of how to deal with underflow!
        https://stats.stackexchange.com/questions/7969/how-to-quickly-sample-x-if-expx-gamma

        It exploits the fact that if u ~ Unif(0,1) and x ~ Gamma(alpha + 1, 1), then
           y = u^(1/alpha) * x ~ Gamma(alpha, 1).
        
        We take the log of both sides of this. We avoid underflow errors since neither
        x or u is small.

        Finally, normalizing the y's to sum up to 1 leads to a Dirichlet distributed rv.
        '''
        return np.log(np.random.dirichlet(alphas))
        # u = np.random.uniform(size=alphas.shape)
        # x = np.random.gamma(alphas+1)
        # logy = (np.log(u) / alphas) + np.log(x)
        # norm = scipy.special.logsumexp(logy)
        # return logy - norm

    @staticmethod
    def sample_log_dirichlet_stick_breaking(gamma, class_counts):
        counts_after = np.cumsum(class_counts[::-1])[::-1][1:]
        # Stick-breaking weights are Vs = 1 - iVs. This is done for numerical 
        # stability when taking the logs.
        for _ in range(10000):
            iVs = np.random.beta(gamma + counts_after, 1 + class_counts[:-1])
            if not np.any(iVs == 0):
                break
        else:
            raise Exception("Gamma too small, numerical instability")
        ps = np.zeros_like(class_counts)
        ps[:-1] = 1 - iVs
        ps[-1] = 1
        ps[1:] *= np.cumprod(iVs)
        return np.log(ps), iVs

    def sample_marginal_lifetimes(self, sum_of_lifetimes_per_class, uncensored_events_per_class,
                                        a_tau_hp, b_tau_hp):
        # Note: the unoccupied classes still get assigned lifetimes. They are
        #       drawn from the prior distribution for tau.
        if not self.do_truncation:
            taus = 1 / np.random.gamma(a_tau_hp + uncensored_events_per_class, 
                                        1 / (b_tau_hp + sum_of_lifetimes_per_class))
        else:
            taus = np.zeros_like(sum_of_lifetimes_per_class)
            for k in range(len(taus)):
                taus[k] = self.sample_from_trunc_exp_posterior(
                            uncensored_events_per_class[k],
                            sum_of_lifetimes_per_class[k],
                            self.t_censor,
                            alpha=a_tau_hp,
                            beta=b_tau_hp
                        )
        
        return taus
    
    @staticmethod
    def inv_gamma_log_pdf(tau, alpha, beta):
        return (- beta / tau) - (alpha + 1) * np.log(tau)
    
    @staticmethod
    def trunc_exp_log_pdf(tau, sum_T, N, tc):
        return (- sum_T / tau) - N * np.log(tau) - N*np.log(1 - np.exp(-tc/tau)) 

    @staticmethod
    def sample_from_trunc_exp_posterior(N, sum_T, tc, 
                                        tau_min=1, tau_max=500, n_trials=500, 
                                        alpha=-1, beta=0):
        '''
        Draw one sample from the posterior of tau, with a truncated exponential likelihood
        and an inverse gamma prior, using rejection sampling.

        * `alpha`: parameter of inverse gamma prior 
        * `beta`: parameter of inverse gamma prior
        * `N`: number of observations of truncated data points
        * `sum_T`: sum of the N observed transition times
        * `tc`: the truncation time
        * `tau_min` and `tau_max`: the range of taus within which to sample. The likelihood should be
            very small outside this range, relative to the maximum.
        * `n_trials`: number of simultaneous trials to do: numpy parallelization is much faster
            than python
        '''
        if N == 0:
            # Likelihood is uniform, so just sample from the prior
            return 1 / np.random.gamma(alpha, 1/beta)

        for i in range(100):
            # Try out a number of taus between the min and max value
            taus_sample = np.random.uniform(tau_min, tau_max, size=n_trials)
            
            # Calculate the log posterior at each tau
            logp_sample = BNPFitter.inv_gamma_log_pdf(taus_sample, alpha, beta)
            logp_sample += BNPFitter.trunc_exp_log_pdf(taus_sample, sum_T, N, tc)

            # Normalize so that max is 1
            logp_sample -= logp_sample.max()
            yval_curve = np.exp(logp_sample)
            
            # Pick a y value between 0 and 1 for each tau
            yval_sample = np.random.random(size=n_trials)

            # Look for the first sample which lies 'under the curve'
            if np.any(yval_sample < yval_curve):
                return taus_sample[np.argmax(yval_sample < yval_curve)]

        raise Exception('Rejected too many samples! N=%d, sum_T=%d, tc=%d' % (N,sum_T,tc))

    def sample_gamma(self, iVs, a_gamma_hhp, b_gamma_hhp):
        np.random.gamma(a_gamma_hhp + self.K_max - 1, 
                                1 / (b_gamma_hhp - np.log(iVs).sum()))

    def calculate_joint_log_posterior(self, taus, log_ps, gamma, one_hot_class, class_counts):
        # first calculate the log likelihood
        x = -np.outer(np.log(taus), self.uncensored_events_per_bead)
        x += np.outer(-1 / taus, self.sum_of_lifetimes_per_bead)
        if self.do_truncation:
            x -= np.outer(np.log(1 - np.exp(-self.t_censor/taus)), self.uncensored_events_per_bead)
        log_likelihood = (x.T * one_hot_class).sum()

        # and then the log prior
        log_prior = 0
        # x = log_ps * (class_counts + gamma - 1)
        # log_prior += x[class_counts != 0].sum()  
        # log_prior += - (1/gamma + np.sum(class_counts))* (class_counts != 0).sum() # ??
        log_prior += scipy.stats.invgamma.logpdf(taus, a=self.a_tau_hp, scale=self.b_tau_hp).sum()
        log_posterior = log_prior + log_likelihood

        return log_prior, log_likelihood, log_posterior

    def clean_trajectory(self, traj_df):
        # If a class is empty at an iteration, its tau should be set to NaN
        for k in range(self.K_max):
            traj_df['tau_%d' % k][traj_df['n_%d' % k] <= 1] = np.nan

        # Add a column to count the number of nonzero classes
        class_counts = traj_df[['n_%d' % k for k in range(self.K_max)]]
        traj_df['K'] = (class_counts > 1).sum(axis=1)

        # And relabel
        traj_df = relabel_trajectory(traj_df)

        return traj_df

    def make_trajectory_entry(self, *args):
        return dict(zip(self.traj_columns, np.concatenate((args))))

    def make_traj_columns(self):
        return ['gamma', 'log_prior', 'log_likelihood', 'log_posterior'] + \
                ['%s_%d' % (s,k) for s in ['p', 'tau', 'n'] for k in range(self.K_max)]

    @staticmethod
    def make_iterator(num_iterations, verbose):
        ts = range(num_iterations)
        if verbose >= 1:
            ts = tqdm.notebook.tqdm(ts)
        return ts


def relabel_trajectory(traj_df_original, by='tau', reverse=False):
    '''
    Takes a Monte Carlo trajectory and adds columns for 'relabeled' parameters.
    The components of each sample are sorted and relabeled as a naive way to
    deal with the label ambiguity.

    `by`: either 'tau' or 'p'; which parameter to sort by.
    `reverse`: whether to sort in ascending order.
    '''
    traj_df = traj_df_original.copy()
    K_max = len([s for s in traj_df.columns if s.startswith('tau')])
    tauss = traj_df[[('tau_%d' % i) for i in range(K_max)]].values
    pss = traj_df[[('p_%d' % i) for i in range(K_max)]].values
    nss = traj_df[[('n_%d' % i) for i in range(K_max)]].values

    rev = 1 if reverse else -1
    if by == 'tau':
        relabel = (rev * tauss).argsort(axis=1)
    elif by == 'p':
        relabel = (rev * pss).argsort(axis=1)
    else:
        raise Exception("by must be tau or p")

    static_indices = np.indices(tauss.shape)
    tauss_relabeled = tauss[static_indices[0], relabel]
    pss_relabeled = pss[static_indices[0], relabel]
    nss_relabeled = nss[static_indices[0], relabel]

    for i in range(K_max):
        traj_df['relabeled_tau_%d'%i] = tauss_relabeled[:,i]
        traj_df['relabeled_p_%d'%i] = pss_relabeled[:,i]
        traj_df['relabeled_n_%d'%i] = nss_relabeled[:,i]
    
    return traj_df

def plot_dataset(times, t_censor=np.nan, raw_bins=50, mean_bins=20, title='', figsize=(15, 4), plot_pulls=True, min_num_events=1, log_y=True):
    ts = np.array(times).copy()
    ts[ts == np.inf] = t_censor
    ts[ts == -np.inf] = np.nan

    ncols = 3 if plot_pulls else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    ts = ts[np.sum(~np.isnan(ts), axis=1) >= min_num_events]
    N,M = ts.shape

    tmin = np.nanmin(ts)
    tmax = np.nanmax(ts)
    axes[0].hist(ts.reshape(-1), color='gray', alpha=0.7,
                 bins = raw_bins) #np.linspace(tmin, tmax, raw_bins))
    axes[0].set_ylabel('Number of statistics')
    axes[0].set_xlabel('Raw observed lifetimes')
    if log_y:
        axes[0].set_yscale('log')
        # axes[0].set_ylim(1, 400)
        axes[0].set_xlim(1, 480)

    xs = np.arange(M+1)
    means = np.nanmean(ts, axis=1)
    meanmin = np.nanmin(means)
    meanmax = np.nanmax(means)
    num_good_events = np.sum(~np.isnan(ts), axis=1)
    means_by_m = [means[num_good_events == x] for x in xs]
    axes[1].hist(means_by_m, alpha=0.7, stacked=True,
                 color = plt.cm.cool(np.linspace(0, 1, M+1)),
                 bins = np.linspace(meanmin, meanmax, mean_bins))
    axes[1].set_ylabel('Number of beads')
    axes[1].set_xlabel('Per-molecule-averaged\nobserved lifetime')

    if plot_pulls:
        axes[2].bar(xs, [np.sum(num_good_events == x) for x in xs],
                     color = plt.cm.cool(np.linspace(0, 1, M+1)))
        axes[2].set_xlabel('Number of pulls')
        axes[2].set_ylabel('Number of beads')
        axes[2].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    fig.suptitle(title)
    plt.tight_layout()
    return fig, axes

def plot_trajectories(traj_df, relabel=False,
                      linewidth=1, window_size=1, skip=1, 
                      log_lifetimes=False, log_proportions=False, 
                      gamma=False, n_components=True):
    nrow = 3
    if n_components:
        nrow += 1
    if gamma:
        nrow += 1

    fig, axes = plt.subplots(nrow, 1, figsize=(15, 3*nrow), sharex=True)
    
    def sliding_average(x, window_size=window_size, skip=skip):
        slider = np.ones(window_size) / window_size
        x_padded = np.pad(x, (window_size//2, window_size-1-window_size//2), mode='edge')
        return np.repeat(np.convolve(x_padded[::skip], slider, mode='valid'), skip)

    rel = 'relabeled_' if relabel else ''
    K_max = len([s for s in traj_df.columns if s.startswith('tau')])
    for i in range(K_max):
        axes[0].plot(sliding_average(traj_df[rel + 'p_%d' % i]), linewidth=linewidth)
        axes[1].plot(sliding_average(traj_df[rel + 'tau_%d' % i]), linewidth=linewidth)

    row = 2

    if n_components:
        axes[row].step(traj_df['K'], '-', lw=1)
        axes[row].yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        axes[row].set_ylabel('Number of\npopulated\ncomponents')
        row += 1
    if gamma:
        axes[row].plot(sliding_average(traj_df['gamma']), linewidth=linewidth)
        axes[row].set_ylabel('Gamma')
        row += 1

    # axes[row].plot(sliding_average(traj_df['log_prior']), linewidth=linewidth, label='prior')
    # axes[row].plot(sliding_average(traj_df['log_likelihood']), linewidth=linewidth, label='likelihood')
    axes[row].plot(sliding_average(traj_df['log_posterior']), linewidth=linewidth, label='posterior')
    axes[row].set_ylabel('Log posterior\n(unnormalized)')
    # axes[row].legend()

    axes[0].set_ylabel('Proportions\nof species')
    axes[0].set_ylim(0,1)
    axes[1].set_ylabel('Lifetime of\nspecies (s)')
    if log_lifetimes:
        axes[1].set_yscale('log')
    if log_proportions:
        axes[0].set_yscale('log')
        axes[0].set_ylim(1e-3, 1)
    
    axes[-1].set_xlabel('MC iteration number')
    
    return fig, axes
    
def plot_posterior_histograms(traj, k, relabel=True, 
                              tau_bins=30, p_bins=40, suptitle=True,
                              legend=False, figsize=(12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    p = sum(traj['K'] == k) / len(traj['K'])
    if suptitle:
        fig.suptitle('%d classes; posterior probability %.1f%%' % (k, 100*p))
    traj = traj[traj['K'] == k]
    
    ps = np.linspace(0.001, 1, p_bins)
    rel = 'relabeled_' if relabel else ''
    for i in range(k):
        let = 'abcdefghikj'[i]
        axes[0].hist(traj['%stau_%d' % (rel, i)], label=r'posterior of $\tau_%c$' % let,
                     alpha=0.5, bins=tau_bins)
        axes[1].hist(traj['%sp_%d' % (rel, i)], label=r'posterior of $p_%c$' % let,
                     alpha=0.5, bins=ps)
    if legend:
        axes[0].legend()
        axes[1].legend()
    axes[0].set_xlabel(r'Lifetime $\tau_c$')
    axes[1].set_xlabel(r'Proportions $p_c$')
    axes[0].set_xscale('log')
    plt.tight_layout()

def get_param_names(K, relabel=True, f=False):
    params = ['tau', 'p']
    if f:
        params += ['f']
    rel = 'relabeled_' if relabel else ''
    return [rel + param + '_%d' % k for param in params for k in range(K)]

def subset_trajectory(traj_df, K, break_in=2500, sample_every=30):
    return traj_df[traj_df['K'] == K][break_in::sample_every]

def get_class_prob_trajectories(times, traj, K, relabel=True, t_censor=np.nan):
    # should probably rewrite code so that you don't have to pass so many things in...
    ts = times.copy()
    N,M = ts.shape
    ts[ts == -np.inf] = np.nan
    ts[ts == np.inf] = t_censor
    Ts = np.nansum(ts, axis=1)
    ms = np.sum(~np.isnan(ts), axis=1)
    if not np.isnan(t_censor):
        ms -= np.sum(ts == t_censor, axis=1)

    params = get_param_names(K, relabel)
    pss = [[] for i in range(K)]
    for i in range(len(traj)):
        taus_and_ps = traj[params].iloc[i].values
        taus = taus_and_ps[:K]
        ps = taus_and_ps[K:]
        x = np.log(ps)[:,np.newaxis] - np.outer(np.log(taus), ms) + np.outer(-1 / taus, Ts)
        # convert and normalize into class probabilities
        x -= x.max(axis=0)
        x = np.exp(x)
        x /= x.sum(axis=0)
        
        for k in range(K):
            pss[k].append(x[k])

    for i in range(K):
        pss[i] = pd.DataFrame(pss[i])
        pss[i].columns = ts.index
        pss[i].columns.name = 'Bead ID'
        pss[i].index = traj.index
        pss[i].index.name = 'MC iteration'

    return pss

def autocorr_func_1d(x, norm=True):
    def next_pow_two(n):
        i = 1
        while i < n:
            i = i << 1
        return i

    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    # impute any np.nan's....this probably makes it smoother than it deserves to be
    mask = np.isfinite(x)
    xi = np.arange(len(x))
    xfiltered = np.interp(xi, xi[mask], x[mask])
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(xfiltered - np.mean(xfiltered), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

def _make_latex_label(param_name):
    if param_name.startswith('relabeled_'):
        param_name = param_name[len('relabeled_'):]
    if param_name.startswith('tau'):
        param_name = '\\' + param_name
    return '$' + param_name + '$'

def plot_autocorr(traj_df, break_in=0, relabel=True, num_components=1, xmax=500):
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))

    for param_name in get_param_names(num_components, relabel):
        label = _make_latex_label(param_name)
        ax.plot(autocorr_func_1d(traj_df[param_name][break_in:])[:xmax], label=label)
    ax.set_ylabel('autocorrelation')
    ax.axhline(y=0, color='gray', ls=':')
    ax.legend()
    ax.set_xlabel('Difference in MC iteration number')
    return fig, ax

def plot_tau_distro(traj, ax=None, title='', 
        xmin=10, xmax=500, n_bins=100,
        label_peaks=True, ha='right', logscale=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    K_max = len([s for s in traj.columns if s.startswith('tau')])
    taus = traj[[('tau_%d' % i) for i in range(K_max)]].values.flatten().copy()
    ns = traj[[('n_%d' % i) for i in range(K_max)]].astype(int).values.flatten().copy()
    # for each sample, weight each "tau" by the number of beads which have that "tau".
    weighted_taus = np.repeat(taus, ns)

    ax2 = ax.twinx()
    if logscale:
        bins = 10**np.linspace(np.log10(xmin), np.log10(xmax), n_bins)
    else:
        bins = np.linspace(xmin, xmax, n_bins)
    heights, bins, _ = ax2.hist(weighted_taus, bins=bins, alpha=0.7)
    ax2.set_yticks([])
    ax.plot(bins[1:], np.cumsum(heights) / len(weighted_taus), 'k:')
    if logscale:
        ax.set_xscale('log')

    if label_peaks:
        K_best = np.argmax(np.bincount(traj['K']))
        for k in range(K_best):
            tau_traj = traj[traj['K'] == K_best]['relabeled_tau_%d' % k]
            ax.text(tau_traj.mean(), 0.1 * (k + 1),
                    r'  $%.1f \pm %.1f$  '%(tau_traj.mean(), tau_traj.std()),
                    ha=ha, fontsize=14)

    ax.set_ylabel('Probability density')
    ax.set_xlabel(r'lifetime $\tau$ (s)')
    ax.set_title(title)

def plot_K_distro(traj, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4,3))
    import collections
    c = collections.Counter(traj['K'])
    locs, heights = zip(*c.items())
    ax.bar(locs, np.array(heights) / len(traj['K']))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_xlabel('# species')
    ax.set_ylabel('Posterior probability')
    ax.set_title(title)
    ax.set_xlim(0.4, 9.6)

def get_param_summary(traj, k, break_in=0, sample_every=1):
    traj = traj[break_in::sample_every]
    traj = traj[traj['K'] == k]

    rel_params = get_param_names(k, relabel=True)

    df = traj[rel_params].describe().loc[['mean','std']]
    df.columns = df.columns.map(lambda s: s.replace('relabeled_',''))

    return df

def plot_inv_gamma_prior(a, b, xmin=-2, xmax=3.5):
    xs = np.logspace(xmin, xmax, 100)
    ys = scipy.stats.invgamma.pdf(xs, a=a, scale=b) * xs * np.log(10)
    plt.plot(xs, ys)
    plt.xscale('log')
