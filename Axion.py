import os
import numpy as np
from scipy.interpolate import interp1d
import blueice as bi
import pandas as pd
import json
from bbfsource import BBFSource
from batchq import submit_job

# change this to wherever you install the repo
workdir = "/home/ershockley/analysis/axions"

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

    signal_generator_template = 'Not Implemented'

    def __init__(self, mass, g=None, **kwargs):
        xsec_data = pd.read_csv(full_path('data/photoelectric_xsecs.csv'),
                                header=0, names=["energy", "xsec"])
        # energy in keV, cross section in barns/atom
        self.pe_xsec = interp1d(xsec_data['energy'], xsec_data['xsec'])
        self.mass = mass
        self.g = g if g is not None else self.g_scale
        Espectrum = kwargs.pop('Espectrum', None)
        self.config = self.build_config(**kwargs)
        self.write_file(m=mass, g=g, Espectrum=Espectrum)

    def build_config(self, signal_model_path=None):
        # setup config for blueice/laidbax. No axion info yet, that comes in inherited classes
        from laidbax.base_model import config

        myconfig = config.copy()

        myconfig['force_recalculation'] = True
        myconfig['never_save_to_cache'] = True
        myconfig['data_dirs'] = ['.', full_path('data')]

        myconfig['analysis_space'] = (('cs1', tuple(np.linspace(0, 100, 11))),
                                      ('cs2_bottom', tuple(np.linspace(500, 8000, 21))))

        myconfig['livetime_days'] = sciencerun['livetime_days']
        myconfig['fiducial_mass'] = sciencerun['fiducial_mass']
        myconfig['default_source_class'] = BBFSource

        er_background = {'color': 'blue',
                         'jsonname': full_path('data/er_cs1cs2.json'),
                         'extra_dont_hash_settings': er_ignore_parameters,
                         'label': 'ER',
                         'n_events_for_pdf': 20000000.0,
                         'name': 'er',
                         'recoil_type': 'er',
                         'in_events_per_bin': True}
        myconfig['sources'] = [er_background]

        return myconfig

    def write_file(self, m=None, g=None, **kwargs):
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

    def build_signal_model(self, filepath, extra_args={}, **kwargs):
        if not filepath.endswith('.json'):
            raise NotImplementedError('Output file must be a json!')
        cs1_range = cs2_range = None
        for (label, space) in (self.config['analysis_space']):
            nbins = len(space)-1
            space_max = max(space)
            space_min = min(space)
            bbf_space = [str(nbins), str(space_min), str(space_max)]
            if label == 'cs1':
                cs1_range = ' '.join(bbf_space)
            elif label == 'cs2_bottom':
                cs2_range = ' '.join(bbf_space)
            else:
                raise NotImplementedError("%s is not a valid analysis space" % label)

        job = self.model_generator_command(filepath, cs1_range=cs1_range, cs2_range=cs2_range, **extra_args)
        log = filepath.replace('.json', '.log')
        submit_job(job, log=log, env="None", **kwargs)
        #print("Job submitted to build signal model. Wait for it to finish before proceeding")

    def model_generator_command(self, filepath, **kwargs):
        raise NotImplementedError

    @classmethod
    def limit_on_g(cls, rate_limit):
        return NotImplementedError

class ALP(Axion):
    g_scale = 1e-13
    # ALP is monoenergetic peak
    signal_generator_template = "source {dir}/setup_modelgenerator.sh\n" \
                                "python ~/bbf_model_generator.py --output {filepath} --prefix_dir {prefix_dir} " \
                                "--spectrum_tag=normal --tag=er_sr1_standard --axis 'cs1' {cs1_range} " \
                                "--axis 'cs2' {cs2_range} --normal_loc {mass} --normal_scale {sigma}\n " \
                                "python {dir}/normalize.py {filepath_full}"

    def build_config(self, signal_model_path=None):
        if signal_model_path is None:
            raise ValueError('path to signal model must be passed for the ALP')
        # add ALP source to baseclass config
        ALPs = {'color': 'red',
                'jsonname': signal_model_path,
                'extra_dont_hash_settings': er_ignore_parameters,
                'label': 'ALP %0.1f keV' % self.mass,
                'n_events_for_pdf': 20000000.0,
                'name': 'ALP',
                'recoil_type': 'er',
                'in_events_per_bin': True,
                # normalize the ALP histograms from bbf to be 1 event/ton/year, so need to scale by actual rate
                # self.rate() gives events/kg/day
                'histogram_multiplier': self.rate() * 365 * 1000}

        myconfig = super().build_config()
        myconfig['sources'].append(ALPs)
        self.sources = [source['name'] for source in myconfig['sources']]

        return myconfig

    def write_file(self, m=None, g=g_scale, **kwargs):
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
        # events/kg/day
        if mass is None:
            mass = self.mass
        if g is None:
            g = self.g
        return (1.29e19 / A) * mass * self.pe_xsec(mass) * g**2

    def flux(self, E, m=None, g=None):
        return NotImplementedError

    def total_events(self, mass=None, g=None):
        return self.rate(mass, g) * self.config['livetime_days'] * self.config['fiducial_mass']

    def model_generator_command(self, filepath, **kwargs):
        # need the following passed as kwargs
        required_kwargs = ['cs1_range', 'cs2_range']
        if not all([kw in kwargs for kw in required_kwargs]):
            raise ValueError('Missing one or more of the following kwargs to model_generator_command: %s' %
                             ', '.join(required_kwargs))
        if '/' in filepath:
            filename = filepath.split('/')[-1]
            prefix_dir = '/'.join(filepath.split('/')[:-1])
            filepath_full = os.path.join(prefix_dir, filename)
        else:
            filename = filepath_full = filepath
            prefix_dir='.'

        return self.signal_generator_template.format(dir=workdir,
                                                     filepath=filename,
                                                     prefix_dir=prefix_dir,
                                                     filepath_full=filepath_full,
                                                     cs1_range=kwargs.get('cs1_range'),
                                                     cs2_range=kwargs.get('cs2_range'),
                                                     mass=self.mass,
                                                     sigma=self.mass/1e10  # approximate delta function
                                                    )

    @classmethod
    def limit_on_g(cls, rate_limit):
        return np.sqrt(rate_limit) * cls.g_scale


class SolarAxion(Axion):
    '''Solar Axion class. Inherits from Axion'''
    g_scale = 1e-11

    signal_generator_template = "source {dir}/setup_modelgenerator.sh\n" \
                                "python ~/bbf_model_generator.py --output {filename} --prefix_dir {prefix_dir} " \
                                "--spectrum_tag=custom --tag=er_signal_sr1_standard --axis 'cs1' {cs1_range} " \
                                "--axis 'cs2' {cs2_range} --custom_filename {input_file}"
    def __init__(self, mass, g=None, **kwargs):
        self.data = pd.read_csv(full_path('data/solaraxion_flux.csv'),
                                usecols=[1,2])
        self.base_flux = interp1d(self.data['E'], self.data['flux'])
        g = self.g_scale if g is None else g
        self.data['flux'] = self.flux(self.data['E'], m=mass, g=g)
        super().__init__(mass, g, **kwargs)


    def build_config(self, signal_model_path=None):
        if signal_model_path is None:
            raise ValueError('path to signal model must be passed for the ALP')
        # add solar axion source to baseclass config
        solar_axion = {'color': 'red',
                       'jsonname': signal_model_path,
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

    def write_file(self, m=None, g=g_scale, **kwargs):
        if m is None:
            m = self.mass
        filename = kwargs.get('Espectrum', 'solar_axion_signal.json')
        #bi.data_reading.CACHE = dict()
        binning = [0.1, 10, 100]
        es = np.linspace(*binning)
        rates = self.dRdE(es).tolist()
        to_encode = {'coordinate_system': [["E", binning]],
                     'map': rates}
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(to_encode, f)


    def model_generator_command(self, filepath, **kwargs):
        # need the following passed as kwargs
        required_kwargs = ['cs1_range', 'cs2_range', 'input_file']
        if not all([kw in kwargs for kw in required_kwargs]):
            raise ValueError('Missing one or more of the following kwargs to model_generator_command: %s' %
                             ', '.join(required_kwargs))
        filename = filepath.split('/')[-1]
        prefix_dir = '/'.join(filepath.split('/')[:-1])
        return self.signal_generator_template.format(dir=workdir,
                                                     filename=filename,
                                                     prefix_dir=prefix_dir,
                                                     cs1_range=kwargs.get('cs1_range'),
                                                     cs2_range=kwargs.get('cs2_range'),
                                                     input_file=kwargs.get('input_file')
                                                    )

    @classmethod
    def limit_on_g(cls, rate_limit):
        return (rate_limit**0.25) * cls.g_scale

def full_path(local_path):
    """wrapper for finding files in the axion repo"""
    return os.path.join(workdir, local_path)