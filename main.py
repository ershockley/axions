import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import time
import pandas as pd
import pickle
from scipy import stats
import blueice as bi
from batchq import submit_job
from Axion import ALP, SolarAxion, full_path


# types of signals we can search for at the moment
implemented_signals = ['ALP', 'solar_axion']
# the mass ranges to search for the above signals
signal_config = {'ALP': {'masses': np.linspace(1, 10, 37),
                         'class': ALP
                         },
                      'solar_axion': {'masses': np.logspace(-5, 0, 6),
                                      'class': SolarAxion}
                      }

minimize_kwargs = {'method': "Powell", "options": {'maxiter' : 10000000}}

def main():
    parser = argparse.ArgumentParser(description="Axion Analysis")
    parser.add_argument('--mode', help='modes/stages for running this analysis',
                        choices=['signal_model', 'submit_limit', 'make_limit', 'post'])
    parser.add_argument('--signal', help='Which signal are you searching for?', choices=implemented_signals)
    parser.add_argument('--mass', help='Mass of the axion searched for', type=float, default=None)
    parser.add_argument('--output_file', help='Where to save the aggregated limit pickle file', default=None)

    args = parser.parse_args()

    if args.mode == 'signal_model':
        # get the mass values for whatever signal we passed
        masses = signal_config[args.signal]['masses']
        signal_model(masses, args.signal)

    elif args.mode == 'submit_limit':
        masses = signal_config[args.signal]['masses']
        for m in masses:
            submit_limit(m, args.signal)

    elif args.mode == 'make_limit':
        if args.mass is None:
            raise ValueError('Mass is required to make a limit')
        m = args.mass
        # read in SR1 data
        data = pd.read_hdf(full_path('data/none_SR1_pax6.8.0_hax2.4.0_lax1.5.1_cs1LT200_fv1_cuts1.h5'),
                           'table')
        make_limit(m, args.signal, data)

    elif args.mode == 'post':
        if args.output_file is None:
            print("--output_file required to run in post mode.")
            return
        masses, limit = collect(args.signal, 'g_limit')
        _, sensitivity = collect(args.signal, 'band')
        with open(args.output_file, 'wb') as f:
            pickle.dump(dict(masses=masses, limit=limit, sensitivity=sensitivity), f)


def signal_model(masses, signal):
    axion = signal_config[signal]['class']
    with open('%s_lookup.txt' % signal, 'w') as f:
        for i, m in enumerate(masses):
            f.write('%d,%f\n' % (i, m))
            # if needed, make a new directory to keep things like signal model, limit/sensitivity data, etc
            this_dir = full_path('work/%s_%d' % (signal, i))
            os.makedirs(this_dir, exist_ok=True)
            # next few lines ONLY APPLY TO SOLAR AXION
            # where to save the signal E spectrum, only care for solar axion
            signal_Espectrum = os.path.join(this_dir, 'signal_Espectrum.json')
            # where to save the signal model (in cs1 cs2 space)
            sig_model_output = os.path.join(this_dir, 'signal.json')
            A = axion(m, Espectrum=signal_Espectrum, signal_model_path=sig_model_output)
            # ALP doesn't do anything with input_file, so this is OK
            extra_args = dict(input_file=signal_Espectrum) if signal == 'solar_axion' else {}
            A.build_signal_model(sig_model_output, extra_args=extra_args, partition='dali', qos='dali',
                                 # dry_run=True,
                                 mem_per_cpu=6000, jobname='%s_%d' % (signal, i))
            time.sleep(1)


def submit_limit(mass, signal):
    job = "python {dir}/main.py --mode make_limit --signal {signal} --mass {mass}"
    job = job.format(dir=full_path('.'), signal=signal, mass=mass)
    jobname = "%s_%f" % (signal, mass)
    log = os.path.join(directory(mass, signal), 'limit.log')
    submit_job(job, jobname=jobname, log=log, partition='dali', qos='dali',
               env='stats')
    time.sleep(1)

def make_limit(mass, signal, data):
    # initialize axion class and likelihood
    axion = signal_config[signal]['class']
    workdir = directory(mass, signal)
    sigmodelpath = os.path.join(workdir, 'signal.json')
    A = axion(mass, signal_model_path=sigmodelpath)
    lf = bi.UnbinnedLogLikelihood(A.config)
    lf.add_rate_parameter(signal)
    lf.add_rate_parameter('er')
    lf.prepare()
    lf.set_data(data)
    axion_rm = '%s_rate_multiplier' % signal

    bestfit, max_ll = bi.inference.bestfit_scipy(lf, minimize_kwargs=minimize_kwargs)
    axion_best = bestfit[axion_rm]
    print("Axion best: %g" % axion_best)
    er_best = bestfit['er_rate_multiplier']

    # get p value
    # see arXiv:1007.1727v3
    newargs = {axion_rm: 0,
               'minimize_kwargs': minimize_kwargs}
    null_fit, null_ll = bi.inference.bestfit_scipy(lf, **newargs)

    # print(max_ll, null_ll, max_ll - null_ll)
    q0 = 2 * (max_ll - null_ll) if axion_best > 0 else 0
    # print(q0)
    p = 1 - stats.norm.cdf(np.sqrt(q0))

    # set limit
    multiplier_limit = bi.inference.one_parameter_interval(lf, axion_rm, 1e4,
                                                           bestfit_routine=bi.inference.bestfit_scipy,
                                                           minimize_kwargs=minimize_kwargs)

    print(multiplier_limit)
    g_limit = axion.limit_on_g(multiplier_limit)
    print("g limit: %0.3e" % g_limit)

    # make some plots
    f, ax = plt.subplots()
    space_axion = ('%s_rate_multiplier'%signal, np.logspace(-5, 5, 500))
    bi.inference.plot_likelihood_ratio(lf, space_axion, bestfit_routine=bi.inference.bestfit_scipy,
                                       minimize_kwargs=minimize_kwargs)
    plt.axvline(multiplier_limit, color='orange')
    plt.axhline(stats.norm.ppf(0.9) ** 2 / 2,
                label='Asymptotic limit', color='y', linestyle='--')
    plt.xscale('log')
    plt.savefig(full_path('plots/likelihoods/ALP_%d.png' % mass_to_int(mass, signal)))
    plt.show()


    # sensitivity
    sensitivity = []
    for _ in range(1000):
        # simulate with 0 signal
        d = lf.base_model.simulate(rate_multipliers={signal: 0})
        lf.set_data(d)
        # limit for this fake experiment with no injected signal
        lim = bi.inference.one_parameter_interval(lf, axion_rm, 1e4,
                                                  bestfit_routine=bi.inference.bestfit_scipy,
                                                  minimize_kwargs=minimize_kwargs)
        sensitivity.append(axion.limit_on_g(lim))

    sensitivity = np.array(sensitivity)
    np.savez(workdir + "/sensitivity.npz", sensitivity)
    # get median, +/- 1,2 sigma bands. Multiply by 100 cause it's in units of %
    qs = stats.norm.cdf([-2,-1,0,1,2])*100
    print(qs)
    band = np.percentile(sensitivity, qs)
    # convert band to more user friendly dictionary
    band = dict(minus2=band[0], minus1=band[1], median=band[2], plus1=band[3], plus2=band[4])
    limit_file = os.path.join(workdir, 'limit.pkl')
    store_data = dict(mass=mass, g_limit=g_limit, axion_best=axion_best, er_best=er_best, p=p, band=band)
    with open(limit_file, 'wb') as f:
        pickle.dump(store_data, f)


def collect(signal, value):
    masses = signal_config[signal]['masses']
    # special instructions if we're collecting the sensitivity band
    if value == 'band':
        ret = dict(minus2=[], minus1=[], median=[], plus1=[], plus2=[])
    else:
        ret = np.ones_like(masses)
    for i, m in enumerate(masses):
        file = os.path.join(directory(m, signal), 'limit.pkl')
        if not os.path.exists(file):
            print("limit file for %s mass %f does not exist!" % (signal, m))
            continue
        with open(file, 'rb') as f:
            output = pickle.load(f)
            if value == 'band':
                for key, item in output[value].items():
                    ret[key].append(item)
            else:
                ret[i] = output[value]

    return masses, ret

def mass_to_int(mass, axion_type):
    mapping = pd.read_csv(full_path('%s_lookup.txt' % axion_type), index_col=0, header=None,
                           names=['mass'])['mass'].values
    return np.where(mapping == mass)[0][0]


def directory(mass, axion_type):
    return full_path('work/%s_%d' % (axion_type, mass_to_int(mass, axion_type)))


if __name__ == '__main__':
    main()