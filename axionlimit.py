import matplotlib
matplotlib.use('Agg')
import os
from Axion import ALP, SolarAxion
import numpy as np
import pandas as pd
from scipy import stats
import sys
import matplotlib.pyplot as plt
import blueice as bi
from tqdm import tqdm

from blueice.inference import bestfit_scipy

PLOT = True
minimize_kwargs = {'method' : "Powell", "options" : {'maxiter' : 10000000}}

datadir = '/home/ershockley/analysis/axions/data'

sciencerun = "SR1"


def make_limit(axion_type_str, mass, file_counter):
    if axion_type_str == 'solar_axion':
        axionobject = SolarAxion
        bound = 1e4
    elif axion_type_str == 'ALP':
        axionobject = ALP
        bound = 1e4
    else:
        raise ValueError("axion type must be \'ALP\' or \'solar_axion\'")

    bi.data_reading.CACHE = dict()

    axion=axionobject(mass)
    axion_type=None

    if sciencerun == 'SR0':
        data = pd.read_csv(os.path.join(datadir, 'xe1t_sr0.csv'))
        data['cs2'] = data['cs2_new']
    elif sciencerun == 'SR1':
        data = pd.read_hdf(os.path.join(datadir,'none_SR1_pax6.8.0_hax2.4.0_lax1.5.1_cs1LT200_fv1_cuts1.h5'), 'table')
        analysis_space = {var: space for var, space in axion.config['analysis_space']}
        for var in ['cs1', 'cs2']:
            data = data[data.apply(lambda x: min(analysis_space[var]) <= x[var] <= max(analysis_space[var]), axis=1)]
    else:
        raise NotImplementedError

    for type in ['solar_axion', 'ALP']:
        if type in axion.sources:
            axion_type = type
    if axion_type is None:
        print(axion.sources)
        raise ValueError("There are no axion sources")

    lf = bi.UnbinnedLogLikelihood(axion.config)
    lf.add_rate_parameter(axion_type)
    lf.add_rate_parameter('erbkg')
    lf.prepare()
    lf.set_data(data)
    axion_rm = '%s_rate_multiplier' % axion_type

    bestfit, max_ll = bestfit_scipy(lf, minimize_kwargs=minimize_kwargs)

    axion_best = bestfit[axion_rm]
    print("Axion best: %g" % axion_best)
    er_best = bestfit['erbkg_rate_multiplier']

    # get p value
    # see arXiv:1007.1727v3
    newargs = {axion_rm: 0,
               'minimize_kwargs': minimize_kwargs}
    null_fit, null_ll = bestfit_scipy(lf, **newargs)

    print(max_ll, null_ll, max_ll-null_ll)
    q0 = 2*(max_ll - null_ll) if axion_best>0 else 0
    print(q0)
    p = 1 - stats.norm.cdf(np.sqrt(q0))

    # set limit
    multiplier_limit = bi.inference.one_parameter_interval(lf, axion_rm, bound,
                                                           bestfit_routine=bestfit_scipy,
                                                           minimize_kwargs=minimize_kwargs)

    if axion_type == 'ALP':
        g_limit = np.sqrt(multiplier_limit) * axion.g_scale
    elif axion_type == 'solar_axion':
        g_limit = (multiplier_limit**0.25) * axion.g_scale
    print("g limit: %0.3e" % g_limit)

    fc_str = str(file_counter).zfill(3)

    with open("/home/ershockley/analysis/axions/limit_data/%s_limit_%s.txt" % (axion_type, fc_str), "w") as f:
        f.write("mass,glimit,axion_best,er_best,pvalue\n")
        f.write("%f,%e,%f,%f,%f" % (mass,g_limit,axion_best,er_best,p))

    ##################### MAKE PLOTS #####################################################
    if PLOT:
        print("Plotting")
        axion_space = (axion_rm, np.logspace(-5, 5, 500))
        er_space = ('erbkg_rate_multiplier', np.linspace(0, 2, 100))

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 8))
        label = 'LL ratio'
        filename = "/home/ershockley/analysis/axions/likelihood_plots/%s_likelihood_%s.png" % (axion_type, fc_str)

        ##### PLOT 1 #######
        plt.sca(ax1)
        bi.inference.plot_likelihood_ratio(lf, axion_space, bestfit_routine=bestfit_scipy,
                                           minimize_kwargs=minimize_kwargs)
        plt.axhline(stats.norm.ppf(0.9) ** 2 / 2,
                    label='p=0.1', color='k', linestyle='--')
        plt.axvline(multiplier_limit, ls='--', color='purple', label='90% U.L.')
        plt.xscale('log')
        plt.ylim(0, 15)
        plt.xlabel(axion_rm)
        plt.ylabel(label)

        ##### PLOT 2 #######
        plt.sca(ax2)
        bi.inference.plot_likelihood_ratio(lf, axion_space, er_space, bestfit_routine=bestfit_scipy,
                                           minimize_kwargs=minimize_kwargs)
        plt.xlabel(axion_rm)
        plt.ylabel('erbkg_rate_multiplier')
        plt.xscale('log')
        plt.colorbar(label=label)

        plt.savefig(filename)

        del f

        print("Plots made")

    ######################### SENSITIVITY CHECK ############################################
    n_trials = 1000
    limits_lf = np.zeros(n_trials)
    background_datasets = [lf.base_model.simulate(rate_multipliers={axion_type : 0}) for _ in range(n_trials)]

    print("got to for loop")
    for i, d in enumerate(background_datasets):
        lf.set_data(d)
        limits_lf[i] = bi.inference.one_parameter_interval(lf, axion_rm, 1e4,
                                                           bestfit_routine=bestfit_scipy,
                                                           minimize_kwargs=minimize_kwargs)

    with open("/home/ershockley/analysis/axions/limit_data/%s_sensitivity_%s.txt" % (axion_type, fc_str), "w") as f:
        f.write("%s mass (keV) : %f\n" % (axion_type, mass))
        for l in limits_lf:
            f.write("%e\n" % l)

if __name__ == '__main__':
    make_limit(sys.argv[1], float(sys.argv[2]), sys.argv[3])
