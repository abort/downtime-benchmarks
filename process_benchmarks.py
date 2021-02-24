#!/bin/python
import pandas as pd
from datetime import datetime, timedelta
import itertools
import seaborn as sns
import matplotlib.ticker as tckr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
plt.rcParams['figure.figsize'] = [25, 20]
procedure_names = ["NEWORD", "PAYMENT", "DELIVERY", "SLEV", "OSTAT"]
non_numbers = ['Property', 'Procedure']
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
        tuples = zip(timestamps[1:], timestamps) # assert len(list(tuples)) == len(timestamps) - 1
        delta = [ int((a - b).total_seconds()) for a, b in tuples ]
        delta = [0] + delta
        timeline = list(itertools.accumulate(delta, lambda x, y: x + y))
        data.columns.values[1:] = timeline
        return data
        
    data = pd.read_csv(file, sep='\t', skiprows=1)
    data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
    data.columns.values[0] = 'Property'
    
    # remove segment rows
    data = data[~data['Property'].isin(procedure_names)]
    data = convert_to_numbers(annotate_procedure_names(convert_to_relative_time(data)))
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
        if transposed.iloc[-1].isna().any():
            transposed = transposed.iloc[:-1]
        return transposed
    
    plots = dict()
    for c in procedure_names:
        plots[c] = compute_latency_matrix(c)
    return plots

def set_time_formatter(axis, second_dispersion):  
    def time_ticks(x, pos):
        d = timedelta(seconds=x)
        return str(d)    
    
    axis.set_major_formatter(tckr.FuncFormatter(time_ticks))
    axis.set_major_locator(plt.MultipleLocator(second_dispersion))

def plot_latencies(start_time, migration_start_time, migration_stop_time, procedure_name, matrix):
    def annot_max(xmax, ymax, ax = None):
        text= "{:.2e}µs".format(ymax)
        if not ax:
            ax=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.20,0.96), **kw)
    
    ax = matrix.plot(x_compat=True)
    plt.yscale('log')
    set_time_formatter(ax.xaxis, 120)
    ax.yaxis.set_major_locator(tckr.LogLocator())
    plt.axvline(x = start_time, linewidth=1, color='r', linestyle='--')
    if migration_start_time:
        plt.axvline(x = migration_start_time, linewidth=1, color='g', linestyle='--')
    if migration_stop_time:
        plt.axvline(x = migration_stop_time, linewidth=1, color='b', linestyle='--')

    ax.set_ylabel("Latency in µs")
    ax.set_xlabel("Time passed in minutes")
    ax.grid(color='b', alpha=0.2, linestyle='dashed', linewidth=0.5)

    annot_max(matrix['MAX'].idxmax(), matrix['MAX'].max(), ax)
    plt.title('%s Latencies' % procedure_name)
    return plt
    
def plot_throughput(start_time, migration_start_time, migration_stop_time, matrix):
    ax = matrix.plot(x_compat=True)
    set_time_formatter(ax.xaxis, 120)
    ax.grid(color='b', alpha=0.2, linestyle='dashed', linewidth=0.5)
    plt.axvline(x = start_time, linewidth=1, color='r', linestyle='--')
    if migration_start_time:    
        plt.axvline(x = migration_start_time, linewidth=1, color='g', linestyle='--')
        
    if migration_stop_time:        
        plt.axvline(x = migration_stop_time, linewidth=1, color='b', linestyle='--')

    ax.set_ylabel("Throughput in TPM")
    ax.set_xlabel("Time passed in minutes")
    plt.title("Throughput")
    return plt

@dataclass
class Benchmark:
    """Class for keeping track of different details of a single benchmark run."""
    latencies: dict
    throughputs: dict
    throughput_counts: pd.DataFrame # perhaps change this to a dict too, this holds the Oracle TX count
    details: pd.DataFrame
    phases: pd.DataFrame
    rampup_time: int = field(init=False)
    start_time: int = field(init=False)
    migration_start_time: int = field(init=False)
    migration_stop_time: int = field(init=False)
    warehouses: int = field(init=False)
    virtual_users: int = field(init=False)
    has_migrations: bool = field(init=False)
        
    def __post_init__(self):
        self.rampup_time = int(self.phases.loc['rampup'].ts)
        self.start_time = int(self.phases.loc['benchmark'].ts)

        self.warehouses = int(self.details.loc['warehouses'].value)
        self.virtual_users = int(self.details.loc['load_users'].value)

        mig_phase = pd.to_numeric(self.details.loc['has_migration_phase'].value)

        if mig_phase == 1:
            num_ts = pd.to_numeric(self.phases["ts"])
            min_ts = int(num_ts.min())
            max_ts = int(num_ts.max())

            # take it with some margin if possible
            self.migration_start_time = self.phases.loc['premigration'].ts
            if self.migration_start_time > min_ts + phase_margin:
                self.migration_start_time -= phase_margin

            self.migration_stop_time = self.phases.loc['postmigration'].ts
            if self.migration_stop_time < max_ts - phase_margin:
                self.migration_stop_time += phase_margin
        else:
            self.migration_start_time = None
            self.migration_stop_time = None
            
    def plot_latencies(self, procedure_name):
        return plot_latencies(self.start_time, self.migration_start_time, self.migration_stop_time, procedure_name, self.latencies[procedure_name])
    
    def plot_throughputs(self):
        return plot_throughput(self.start_time, self.migration_start_time, self.migration_stop_time, self.throughputs)


def read_benchmark(base_dir, name):
    if not path.exists(base_dir):
        print("Base directory %s not found" % base_dir)
        return


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


    latency_matrices = get_latency_matrices(read_percentiles(percentiles_f))
    throughputs = read_throughputs(throughput_f)
    throughput_matrices = get_throughput_matrices(throughputs)
    throughput_count = throughputs[['total_tx']]
    details = read_details(details_f)
    phases = read_phases(phases_f)
    
    result = Benchmark(latency_matrices, throughput_matrices, throughput_count, details, phases)
    return result

if __name__ == "__main__":
    benchmark_names = sys.argv[1:]
    base_dir = Path.cwd() / "outputs"
    if benchmark_names == []:
        print("No benchmark names provided, processing all in base directory %s" % base_dir) 

        benchmark_names = glob("%s/*/" % base_dir)
        benchmark_names = [ Path(b).relative_to(base_dir) for b in benchmark_names ]

    print("Processing %d benchmarks in %s" % (len(benchmark_names), base_dir))
    for n in benchmark_names:
        full_path = base_dir / n
        if not full_path.exists() or not full_path.is_dir():
            print("%s is not a results directory" % full_path)
            continue

        charts_path = os.path.join(full_path, "charts")
        if not os.path.exists(charts_path):
            os.mkdir(charts_path)

            if not os.path.exists(charts_path):
                print("Could not create directory %s" % charts_path)
                exit(1)


        print("Processing: %s" % n)
        b = read_benchmark(str(base_dir.resolve()), n)

        tp = b.plot_throughputs()
        tp.savefig("%s/throughputs.svg" % charts_path)

        for n in procedure_names:
            p = b.plot_latencies(n)
            print("Saving figure %s" % n)
            p.savefig("%s/%s" % (charts_path, n))