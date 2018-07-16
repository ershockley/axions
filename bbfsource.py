import os
import json
import numpy as np
import blueice as bi
import multihist as mh


class BBFSource(bi.HistogramPdfSource):
    """A source that draws cs1,cs2 from bbf output stored in a .json file 
    Configuration parameters:

        jsonname: root file to open.

        named_parameters: list of config setting names to pass to .format on jsonname.

        in_events_per_bin: if True, histogram is taken to be in events per day / bin,
                           if False or absent, taken to be events per day / bin volume

        histogram_multiplier: multiply histogram by this number

        Stupid convenience function for XENON:

        log10_bins: List of axis numbers.
                    If True, bin edges on this axis in the root file are log10() of the actual bin edges.

    """

    def build_histogram(self):
        format_dict = {k: self.config[k] for k in self.config.get('named_parameters', [])}
        jsonname = self.config['jsonname'].format(**format_dict)

        h = json_to_multihist(jsonname)

        # Check the binning
        for a_i, (_, expected_bin_edges) in enumerate(self.config['analysis_space']):
            if h.axis_names[a_i] in self.config.get('log10_bins', []):
                h.bin_edges[a_i] = 10 ** h.bin_edges[a_i]
            seen_bin_edges = h.bin_edges[a_i]
            try:
                assert len(seen_bin_edges) == len(expected_bin_edges)
            except AssertionError:
                print("Axis %s bin_edges don't match expected from config" % h.axis_names[a_i])
                print("Expect %d bin_edges from config" % len(expected_bin_edges))
                print("See %d bin_edges from json" % len(seen_bin_edges))
                print(expected_bin_edges)
                print(seen_bin_edges)
                raise

        self._bin_volumes = h.bin_volumes()  # TODO: make alias
        self._n_events_histogram = h.similar_blank_histogram()  # Shouldn't be in HistogramSource... anyway

        h *= self.config.get('histogram_multiplier', 1)

        # Convert h to density...
        if self.config.get('in_events_per_bin'):
            h.histogram /= h.bin_volumes()
        # bbf outputs histogram in events/ton/year
        self.events_per_day = (h.histogram * self._bin_volumes).sum() / 365

        # ... and finally to probability density
        h.histogram /= self.events_per_day
        self._pdf_histogram = h

    def simulate(self, n_events):
        dtype = [('cs1', float), ('cs2_bottom', float), ('source', int)]
        ret = np.zeros(n_events, dtype=dtype)
        t = self._pdf_histogram.get_random(n_events)
        ret['cs1'] = t[:, 0]
        ret['cs2_bottom'] = t[:, 1]
        return ret


def json_to_multihist(jsonname):
    """Converts a json output from bbf to a multihist object"""
    if not os.path.exists(jsonname):
        raise FileNotFoundError

    print('Opening %s' % jsonname)

    with open(jsonname) as f:
        data = json.load(f)

    axis_names = [data['binning'][0][0], data['binning'][1][0]]

    xbins, x0, x1 = data['binning'][0][1]
    ybins, y0, y1 = data['binning'][1][1]

    x_edges = np.linspace(x0, x1, xbins + 1)
    y_edges = np.linspace(y0, y1, ybins + 1)

    hist = np.array(data['hist'])

    out_hist = mh.Histdd.from_histogram(hist, axis_names=axis_names, bin_edges=[x_edges, y_edges])
    return out_hist