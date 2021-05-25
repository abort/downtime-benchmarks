#!/bin/python
import pandas as pd
import math
from datetime import datetime, timedelta
import itertools
import seaborn as sns
import matplotlib.ticker as tckr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib.dates import DateFormatter, date2num
from dataclasses import dataclass, field
import numpy as np
import os.path
from os import path
import sys
from glob import glob
from pathlib import Path


# Constants
phase_margin = 10  # we take a margin of X seconds for periods to draw them
plt.rcParams['figure.figsize'] = [25, 15]
procedure_names = ["NEWORD", "PAYMENT", "DELIVERY", "SLEV", "OSTAT"]
non_numbers = ['Property', 'Procedure']

matplotlib.use('Agg')
plt.ioff()
sns.set_theme()
sns.set_style('whitegrid')


def convert_to_numbers(df, exceptions = non_numbers):
    cols = [i for i in df.columns if i not in exceptions]
    for col in cols:
        df[col] = df[col].astype(float)
    return df

def read_percentiles(file):
    def annotate_procedure_names(data):
        data['Procedure'] = np.arange(data.shape[0])
        data['Procedure'] = data['Procedure'].apply(lambda x : procedure_names[x // 5])
        return data

    def convert_to_relative_time(data):
        timestamps = [ datetime.strptime(x, '%H:%M:%S') for x in list(data.columns.values[1:]) ]
        tuples = zip(timestamps[1:], timestamps)
        delta = [ int((a - b).total_seconds()) for a, b in tuples ]
        delta = [0] + delta
        timeline = list(itertools.accumulate(delta, lambda x, y: x + y))
        return data.rename(columns = dict(zip(data.columns.values[1:], timeline)))
        
    data = pd.read_csv(file, sep='\t', skiprows=1)
    data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
    data.columns.values[0] = 'Property'
    
    # remove segment rows
    data = data[~data['Property'].isin(procedure_names)]

    data = convert_to_numbers(annotate_procedure_names(convert_to_relative_time(data)))

    # drop trailing measurements that resulted in no value
    # while data[data.columns[-1]].isnull().values.all():
    #     data = data.drop(data.columns[-1], axis=1)

    return data
    
def read_throughputs(file):
    data = pd.read_csv(file, sep=',', index_col=0)
    return convert_to_numbers(data.drop(['system_time'], axis = 1))

def read_details(file):
    return pd.read_csv(file, sep=',', index_col=0)

def read_phases(file):
    return pd.read_csv(file, sep=',', index_col=0)

def get_throughput_matrices(df):
    data = df.copy()
    data.drop('total_tx', axis=1, inplace=True)
    return data

def get_latency_matrices(data):
    def get_procedure_data(p):
        filtered = data.loc[data['Procedure'] == p]
        return filtered[data.columns[~data.columns.isin(['Procedure'])]]
    
    def compute_latency_matrix(p):
        transposed = get_procedure_data(p).T
        transposed.columns = list(transposed.iloc[0])
        transposed = transposed.iloc[1:]
        transposed = convert_to_numbers(transposed)
        
        # check if last has NaN, remove whole row if so
        while transposed.iloc[-1].isna().any():
            transposed = transposed.iloc[:-1]

        transposed.index.name = 'ts'
        return transposed
    
    plots = dict()
    for c in procedure_names:
        plots[c] = compute_latency_matrix(c)
    return plots

def set_time_formatter(axis, second_dispersion):
    def time_ticks(x, pos):
        d = timedelta(seconds=x)
        return str(d)    
    
    axis.set_minor_locator(tckr.MultipleLocator(60))
    axis.set_major_formatter(tckr.FuncFormatter(time_ticks))
    axis.set_major_locator(plt.MultipleLocator(second_dispersion))

def plot_latencies(start_time, migration_start_time, migration_stop_time, procedure_name, matrix, min_y = None, max_y = None):
    def annot_max(xmax, ymax, ax = None):
        text= "{:.2e}µs".format(ymax)
        if not ax:
            ax = plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.20,0.96), **kw)

    ax = matrix.plot(x_compat=True, color = ['b', 'orange', 'r'], linestyle = 'solid')
    plt.yscale('log')
    ax.margins(0, 0)

    set_time_formatter(ax.xaxis, 300)
    ax.yaxis.set_major_locator(tckr.LogLocator())
    if min_y and max_y:
        plt.ylim(min_y, max_y)
        ax.yaxis.set_tick_params(which='minor', reset=False)
        # plt.yticks(np.arange(min_y, max_y, (max_y - min_y) // 3))

    plt.axvline(x = start_time, linewidth=1, color='r', linestyle='--')
    if migration_start_time:
        plt.axvline(x = migration_start_time, linewidth=1, color='g', linestyle='--')
    if migration_stop_time:
        plt.axvline(x = migration_stop_time, linewidth=1, color='b', linestyle='--')

    ax.set_ylabel("Latency in µs")
    ax.set_xlabel("Time passed in minutes")
    ax.grid(color='b', alpha=0.2, linestyle='dashed', linewidth=0.5, which='minor')

    # if 'MAX' in matrix:
    #     annot_max(matrix['MAX'].idxmax(), matrix['MAX'].max(), ax)
    plt.title('%s Latencies' % procedure_name)
    return plt

def plot_throughput(start_time, migration_start_time, migration_stop_time, matrix, max_y = None):
    ax = matrix.plot(x_compat = False)
    set_time_formatter(ax.xaxis, 300)
    ax.grid(color='b', alpha=0.2, linestyle='dashed', linewidth=0.5, which='minor')
    ax.margins(0.0, 0)
    if max_y:
        plt.ylim(top = max_y)
        ax.yaxis.set_tick_params(which='minor', reset=False)
        plt.yticks(np.arange(0, max_y, 5000))

    plt.axvline(x = start_time, linewidth=1, color='r', linestyle='--')
    if migration_start_time:    
        plt.axvline(x = migration_start_time, linewidth=1, color='g', linestyle='--')

    if migration_stop_time:        
        plt.axvline(x = migration_stop_time, linewidth=1, color='b', linestyle='--')

    # plt.tick_params(axis='y', direction='out', length=3, width=3)
    ax.set_ylabel("Throughput in TPM")
    ax.set_xlabel("Time passed in minutes")
    plt.title("Throughput")
    return plt

@dataclass
class Benchmark:
    """Class for keeping track of different details of a single benchmark run."""
    latencies: dict
    throughputs: pd.DataFrame
    throughput_counts: pd.DataFrame # perhaps change this to a dict too, this holds the Oracle TX count
    details: pd.DataFrame
    phases: pd.DataFrame
    rampup_time: int = field(init=False)
    start_time: int = field(init=False)
    migration_margin_start_time: int = field(init=False)
    migration_margin_stop_time: int = field(init=False)
    warehouses: int = field(init=False)
    virtual_users: int = field(init=False)
    has_migrations: bool = field(init=False)
    throughput_computed_rates: pd.DataFrame = field(init=False)

    # compute the delta for two transaction counts and divide by 2 (because each log statement is also a transaction)
    @staticmethod
    def __tx_delta__(x):
        return (x.iloc[-1] - x.iloc[0]) // 2

    def __post_init__(self):
        self.rampup_time = int(self.phases.loc['rampup'].ts)
        self.start_time = int(self.phases.loc['benchmark'].ts)

        self.warehouses = int(self.details.loc['warehouses'].value)
        self.virtual_users = int(self.details.loc['load_users'].value)

        mig_phase = pd.to_numeric(self.details.loc['has_migration_phase'].value)
        self.throughput_computed_rates = self.throughput_counts.rolling(60).apply(self.__tx_delta__)

        self.has_migrations = mig_phase == 1
        if self.has_migrations:
            num_ts = pd.to_numeric(self.phases["ts"])
            min_ts = int(num_ts.min())
            max_ts = int(num_ts.max())

            # take it with some margin if possible due to some possible accuracy deviations
            self.migration_start_time = self.phases.loc['premigration'].ts
            self.migration_margin_start_time = self.migration_start_time
            if self.migration_margin_start_time > min_ts + phase_margin:
                self.migration_margin_start_time -= phase_margin

            self.migration_stop_time = self.phases.loc['postmigration'].ts
            self.migration_margin_stop_time = self.migration_stop_time
            if self.migration_margin_stop_time < max_ts - phase_margin:
                self.migration_margin_stop_time += phase_margin
        else:
            self.migration_start_time = None
            self.migration_stop_time = None
            self.migration_margin_start_time = None
            self.migration_margin_stop_time = None
            
    def plot_latencies(self, procedure_name, simple = False, min_y = None, max_y = None):
        matrix = self.latencies[procedure_name]
        matrix = matrix[['P50', 'P99', 'MAX']] if simple else matrix

        return plot_latencies(self.start_time, self.migration_margin_start_time, self.migration_margin_stop_time, procedure_name, matrix, min_y, max_y)
    
    def plot_throughputs(self, max_y = None):
        return plot_throughput(self.start_time, self.migration_margin_start_time, self.migration_margin_stop_time, self.throughputs, max_y)

    def __get_for_phase__(self, df, phase):
        p = self.phases
        phase_ts = p.loc[phase].ts
        boundaries = p[p.ts >= phase_ts].ts.head(2)  # best effort select
        is_last_phase = len(boundaries) == 1
        min_bound = boundaries[0] - phase_margin  # add some secs margin

        closest_match = df.index.searchsorted(min_bound)
        # closest_match_ts = df.iloc[closest_match].name
        # assert(abs(closest_match_ts - min_bound) < 40)
        if is_last_phase:
            return df.iloc[closest_match:]
        else:
            max_bound = boundaries[1] + phase_margin  # add some secs margin
            closest_phase_end_match = df.index.searchsorted(max_bound)
            # closest_phase_end_match_ts = df.iloc[closest_phase_end_match].name
            # assert(abs(closest_phase_end_match_ts - max_bound) < 40)
            return df.iloc[closest_match:closest_phase_end_match]


    def get_throughput_rates_for_phase(self, procedure, phase):
        return self.__get_for_phase__(self.throughputs[procedure], phase)


    def get_latencies_for_phase(self, procedure, phase):
        return self.__get_for_phase__(self.latencies[procedure], phase)


def read_benchmark(base_dir, name):
    if not path.exists(base_dir):
        print("Base directory %s not found" % base_dir)
        return None


    directory = os.path.join(base_dir, name)
    if not path.exists(directory):
        print("Benchmark %s not found" % name)
        return None

    percentiles_f = os.path.join(directory, 'percentiles.csv')
    if not path.exists(percentiles_f):
        print("Latency measurements %s not found" % percentiles_f)
        return None

    throughput_f = os.path.join(directory, 'throughput.csv')
    if not path.exists(throughput_f):
        print("Throughput measurements %s not found" % throughput_f)
        return None

    details_f = os.path.join(directory, 'details.csv')
    if not path.exists(details_f):
        print("Measurement details/parameters not found" % details_f)
        return None

    phases_f = os.path.join(directory, 'phases.csv')
    if not path.exists(phases_f):
        print("Measurement phases not found" % phases_f)
        return None

    try:
        latency_matrices = get_latency_matrices(read_percentiles(percentiles_f))
        throughputs = read_throughputs(throughput_f)
        throughput_matrices = get_throughput_matrices(throughputs)
        throughput_count = throughputs[['total_tx']]
        details = read_details(details_f)
        phases = read_phases(phases_f)
        
        result = Benchmark(latency_matrices, throughput_matrices, throughput_count, details, phases)
        return result
    except Exception as e:
        print("Failed to retrieve data: ", e)
        return None

if __name__ == "__main__":
    benchmark_names = sys.argv[1:]
    base_dir = Path.cwd() / "outputs"
    if benchmark_names == []:
        print("No benchmark names provided, processing all in base directory %s" % base_dir) 

        benchmark_names = glob("%s/*/" % base_dir)
        benchmark_names = [ Path(b).relative_to(base_dir) for b in benchmark_names ]

    print("Processing %d benchmarks in %s" % (len(benchmark_names), base_dir))

    benchmark_data = dict()

    # create nenecessary directories
    for n in benchmark_names:
        full_path = base_dir / n
        if not full_path.exists() or not full_path.is_dir():
            print("%s is not a results directory, skipping %s" % (full_path, n))
            benchmark_data[n] = None
            continue

        charts_path = os.path.join(full_path, "charts")
        if not os.path.exists(charts_path):
            os.mkdir(charts_path)

            if not os.path.exists(charts_path):
                print("Could not create directory %s, skipping %s" % (charts_path, n))
                benchmark_data[n] = None
                continue

        benchmark = read_benchmark(str(base_dir.resolve()), n)
        if benchmark: 
            benchmark_data[n] = benchmark
        else:
            print("Skipping %s" % n)

    # find extremes to plot nicely
    lat_min = dict()
    lat_max = dict()
    thr_max = None
    for b in benchmark_data.values():
        for p in procedure_names:
            
            # remove first and last 10s if possible, to make sure we have a full window of measurements
            lat = b.latencies[p]
            if len(lat) > 2:
                lat = lat[2:-1]

            # we take the highest value
            mx = lat['MAX'].max()
            # we take the lowest value
            mn = lat['P50'].min()
            if p not in lat_min:
                lat_min[p] = mn
                lat_max[p] = mx
            else:
                if mn < lat_min[p]:
                   lat_min[p] = mn
                if mx > lat_max[p]:
                   lat_max[p] = mx

        # we plot all throughputs together, so we just want the minimum from the minima of each
        throughputs = b.throughputs

        mx = throughputs.max().max()
        if not thr_max or mx > thr_max:
            thr_max = mx

    # make sure that minimum and maximum are nicely logarithmic
    lat_min = { k: 10**int(math.log(v, 10)) for k, v in lat_min.items() }
    lat_max = { k: 10**int(math.ceil(math.log(v, 10))) for k, v in lat_max.items() }

    for n, b in benchmark_data.items():
        full_path = base_dir / n
        charts_path = os.path.join(full_path, "charts")

        tp = b.plot_throughputs(max_y = thr_max)
        tp.savefig("%s/throughputs.pdf" % charts_path, bbox_inches = 'tight', pad_inches = 0)
        tp.close()

        # baseline_throughputs = dict()
        # migration_throughputs = dict()
        # baseline_latencies = dict()
        # migration_latencies = dict()
        for t in procedure_names:
            p = b.plot_latencies(t, simple = True, min_y = lat_min[t], max_y = lat_max[t])
            p.savefig("%s/%s.pdf" % (charts_path, t.lower()), bbox_inches = 'tight', pad_inches = 0)

            # baseline_throughputs[t] = b.get_throughput_rates_for_phase(t, 'benchmark')
            # if b.has_migrations:
            #   migration_throughputs[t] = b.get_throughput_rates_for_phase(t, 'premigration')
            #   migration_latencies[t] = b.get_latencies_for_phase(t, 'premigration')

            # baseline_latencies[t] = b.get_latencies_for_phase(t, 'benchmark')
            p.close()


        # print("Baseline TPM (NEWORD) for %s" % n)
        # print(baseline_throughputs['NEWORD'].describe())
        # if b.has_migrations:
        #   print("Migration TPM (NEWORD) for %s" % n)
        #   print(migration_throughputs['NEWORD'].describe())
        # print("Baseline Latencies (NEWORD) for %s" % n)
        # print(baseline_latencies['NEWORD'].describe())
        # if b.has_migrations:
        #   print("Migration Latencies (NEWORD) for %s" % n)
        #   print(migration_latencies['NEWORD'].describe())