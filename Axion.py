import numpy as np
from scipy.interpolate import interp1d
import blueice as bi
import pandas as pd
import matplotlib.pyplot as plt
from bbfsource import BBFSource

plt.rcParams['figure.figsize'] = (8.0, 6.0)

A = 131.3 # g/mol
target_mass = 1042
N_A = 6.022e23 # atoms/mol
alpha = 1/137
m_e = 511 # keV/c^2

_bound = 1e6

minimize_kwargs = {'method': "Powell", "options": {'maxiter': 10000000}}

SR0 = {'livetime_days' : 34.2,
       'fiducial_mass' : 1042}

SR1 = {'livetime_days' : 246.7,
       'fiducial_mass' : 1042}

sciencerun = SR1

er_ignore_parameters = ['nr_gamma', 'nr_alpha', 'nr_lambda', 'nr_ee', 'wimp_mass', 'nr_eta', 'efficiency']


class Axion:
    """ Base class for axion analysis """
    g_scale = 'Not Implemented'

    def __init__(self, mass, g=None, **kwargs):
        g = g if g else self.g_scale
        xsec_data = pd.read_csv('/home/ershockley/analysis/axions/data/photoelectric_xsecs.csv',
                                header=0, names=["energy", "xsec"])
        # energy in keV, cross section in barns/atom
        self.pe_xsec = interp1d(xsec_data['energy'], xsec_data['xsec'])
        self.mass = mass
        self.g = g
        self.config = self.build_config(**kwargs)
        self.write_file(m=mass, g=g)

    def build_config(self, **kwargs):
        # setup config for blueice/laidbax. No axion info yet, that comes in inherited classes
        from laidbax.base_model import config

        myconfig = config.copy()

        myconfig['force_recalculation'] = True
        myconfig['never_save_to_cache'] = True
        myconfig['data_dirs'] = ['.', '/home/ershockley/analysis/axions/data']

        myconfig['analysis_space'] = (('cs1', tuple(np.linspace(0, 100, 101))),
                                      ('cs2_bottom', tuple(np.linspace(500, 8000, 126))))

        myconfig['livetime_days'] = sciencerun['livetime_days']
        myconfig['fiducial_mass'] = sciencerun['fiducial_mass']
        myconfig['default_source_class'] = BBFSource

        er_background = {'color': 'blue',
                         'jsonname': '/home/ershockley/er_cs1cs2.json',
                         'extra_dont_hash_settings': er_ignore_parameters,
                         'label': 'ER',
                         'n_events_for_pdf': 20000000.0,
                         'name': 'er',
                         'recoil_type': 'er',
                         'in_events_per_bin': True}
        myconfig['sources'] = er_background

        return myconfig

    def write_file(self, m=None, g=None):
        raise NotImplementedError

    def flux(self, E, m, g):
        raise NotImplementedError

    def cross_section(self, E, m=None, g=None):
        ''' returns cross section in units of barns'''
        if m is None:
            m = self.mass
        if g is None:
            g = self.g

        beta = np.sqrt(E**2 - m**2) / E
        # if E is an array, set values where E<m to 0
        if hasattr(E, 'shape'):
            #E = E * (E >= m)
            return np.nan_to_num((self.pe_xsec(E) * (g**2) / beta * 3 * (E**2) / (16 * np.pi * alpha * m_e**2) *
                                  ( 1 - beta **(2/3) / 3)) * (E >= m))
        else:
            if E < m:
                return 0
            return self.pe_xsec(E) * (g**2)/beta * 3*(E**2)/(16*np.pi*alpha*m_e**2) * (1 - beta**(2/3)/3)


    def dRdE(self, E,  m=None, g=None):
        ''' returns energy spectrum in units of #/day/kg/keV '''
        return (N_A*1000/A) * self.flux(E, m, g) * self.cross_section(E, m, g) * 1e-24 #convert barns-->cm^2

class ALP(Axion):
    g_scale = 1e-13
    def build_config(self):

        # add ALP source to baseclass config
        ALPs = {'color': 'red',
                  'energy_distribution': 'ALP.csv',
                  'extra_dont_hash_settings': ['leff', 'qy', 'nr_photon_yield_field_quenching', 'nr_poly_order',
                                               'p_nr_electron_fluctuation', 'nr_p_electron_a', 'nr_p_electron_b',
                                               'nr_p_detectable_a', 'nr_p_detectable_b', 'nr_p_electron_0',
                                               'nr_p_electron_1', 'nr_p_electron_2', 'nr_p_electron_3',
                                               'nr_p_electron_4', 'nr_p_electron_5', 'nr_p_electron_6',
                                               'nr_p_electron_7', 'nr_p_electron_8', 'nr_p_electron_9',
                                               'nr_p_detectable_0', 'nr_p_detectable_1', 'nr_p_detectable_2',
                                               'nr_p_detectable_3', 'nr_p_detectable_4', 'nr_p_detectable_5',
                                               'nr_p_detectable_6', 'nr_p_detectable_7', 'nr_p_detectable_8',
                                               'nr_p_detectable_9'],
                  'label': 'ALP %0.1f keV' % self.mass,
                  'n_events_for_pdf': 20000000.0,
                  'name': 'ALP',
                  'recoil_type': 'er'}

        myconfig = super().build_config()
        myconfig['sources'].append(ALPs)
        self.sources = [source['name'] for source in myconfig['sources']]

        return myconfig

    def write_file(self, m=None, g=g_scale):
        """ writes csv file with delta function at axion mass"""

        if m is None:
            m = self.mass

        # clear the cache dict in bi.data_reading, took me forever to figure this out
        bi.data_reading.CACHE = dict()

        # the rate in this csv file is varied by blueice to set a limit on g
        es = np.linspace(0, 2 * m, 101)
        bin_width = es[1] - es[0]

        # our energy should be indexed by es[50]
        assert abs(es[50] - m) < 1e-6

        # delta function centered around our mass value
        rates = np.zeros(len(es))
        rates[50] = self.rate(m) / bin_width

        pd.DataFrame(np.vstack([es, rates]).T,
                     columns=['kev', 'events_per_day']).to_csv(self.config['data_dirs'][0] + '/ALP.csv')

    def rate(self, mass=None, g=None):
        if mass is None:
            mass = self.mass
        if g is None:
            g = self.g
        return (1.29e19 / A) * mass * self.pe_xsec(mass) * g**2

    def flux(self, E, m=None, g=None):
        return NotImplementedError

    def total_events(self, mass=None, g=None):
        return self.rate(mass, g) * self.config['livetime_days'] * self.config['fiducial_mass']


class SolarAxion(Axion):
    '''Solar Axion class. Inherits from Axion'''
    g_scale = 1e-11

    def __init__(self, mass, g=g_scale, **kwargs):
        self.data = pd.read_csv('/home/ershockley/analysis/axions/data/solaraxion_flux.csv',
                                usecols=[1,2])
        self.base_flux = interp1d(self.data['E'], self.data['flux'])
        self.data['flux'] = self.flux(self.data['E'], m=mass, g=g)

        super().__init__(mass, g, **kwargs)


    def build_config(self, **kwargs):
        # add ALP source to baseclass config
        solar_axion = {'color': 'red',
                       'jsonname': '/home/ershockley/solaraxion_cs1cs2.json',
                       'extra_dont_hash_settings': er_ignore_parameters,
                       'label': 'solar axion',
                       'n_events_for_pdf': 20000000.0,
                       'name': 'solar_axion',
                       'recoil_type': 'er',
                       'in_events_per_bin': True}

        myconfig = super().build_config()
        myconfig['sources'].append(solar_axion)
        self.sources = [source['name'] for source in myconfig['sources']]

        return myconfig

    def flux(self, E, m=None, g=None):
        '''returns flux in units of #/cm^2/day'''
        if g is None:
            g = self.g
        return self.base_flux(E) * (g/0.511e-10)**2 # from flux data taken from arxiv:1310.0823

    def write_file(self, m=None, g=g_scale):
        if m is None:
            m = self.mass
        bi.data_reading.CACHE = dict()

        self.data['energy_spectrum'] = self.dRdE(self.data['E'], m=m, g=g)
        self.data.to_csv(self.config['data_dirs'][0] + '/solar_axion.csv', columns=("E", "energy_spectrum"))
