# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Epidemiological modeling for coronavirus, using the SEIR model."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
mpl.rcParams['figure.figsize'] = [25, 10]

# Number of individuals in the simulation. Used to generate a graph if none is passed.
N = 10000
# Properties of the connection graph. (2, 20, 0.2) makes top 0.1% = 15 * median.
# Minimal degree of nodes in the generated graph.
MIN_DEGREE = 2
# Mean degree of nodes in the generated graph.
MEAN_DEGREE = 20
# Parameter of degree distribution in generated graph.
# Higher gamma means a fatter tail of the degree distribution.
GAMMA = 0.2
# Number of simulation steps per one day of real life time.
STEPS_PER_DAY = 5
# Maximal number of steps in simulation. If this number is reached, simulation stops.
MAX_STEPS = 3000
# Number of individuals infected at t=0.
INITIAL_INFECTED_NUM = 10
# In a single interaction on a single day, what is the chance of an Infectious
# to infect a Susceptible. Fitted to produce doubling_time=3.13 For G with parameters
# (100000, 2, 20, 0.2). Though R0 might still be too small here: 2.0.
PROB_INFECT = 0.022
# In a single interaction on a single day, how much less likely is an Exposed,
# in his infectious period before becoming contagious,
# to infect a Susceptible than an Infected to infect a Susceptible.
# In other words, prob_infect_exposed = prob_infect * PROB_INFECT_EXPOSED_FACTOR.
# Set to 0 to make the Exposed non-infectious.
PROB_INFECT_EXPOSED_FACTOR = 0.5
# How many days before developing symptoms (becoming Infected) an Exposed
# individual is contagious. Questionable evidence, but 1-2 produces a reasonable R0.
DURATION_EXPOSED_INFECTS = 2
# Incubation period duration distribution mean. In days.
# Source: 3rd Imperial College paper: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Europe-estimates-and-NPI-impact-30-03-2020.pdf
INCUBATION_DURATION_MEAN = 5.1
# Incubation period duration distribution standard deviation. In days.
# Source: 3rd Imperial College paper: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Europe-estimates-and-NPI-impact-30-03-2020.pdf
INCUBATION_DURATION_STD = 4.38
# Probability to change from Infections to Recovered in a single day.
# Equals the inverse of the expected time to Recover.
# 3.5 = 0.3 * 0 + 0.56 * 5 + 0.1 * 5 + 0.04 * 5. Source: Pueyo spreadsheet: https://docs.google.com/spreadsheets/d/1uJHvBubps9Z2Iw_-a_xeEbr3-gci6c475t1_bBVkarc/edit#gid=0
PROB_RECOVER = 1 / 3.5
# Length of imposed quarantine, in days.
DAYS_IN_QUARANTINE = 14
# Probability that an Infected individual is tested and detected as sick in a single day.
# This encapsulates also the fact that many people are asymptomatic or will
# not be tested (or are not quarantined). They become Quarantined.
# Default: no tests at all.
PROB_INFECTED_DETECTED = 0
# Probability that a neighbor of an individual who tested positive is himself tested
# and detected as sick. This encapsulates the probability that they are traced
# as well. The testing of neighbors happens once, once the original individual
# is tested positive, so this is *not* a probability per day. They become Quarantined.
# Default - no neighbors are tested or detected.
PROB_NEIGHBOR_DETECTED = 0
# Probability that an Exposed individual is tested and detected as sick in a single day.
# This encapsulates the fact that they are presymptomatic so the entire population
# needs to be tested, and the test false negatives. They become Quarantined.
# Default - previous behavior, no exposed are detected.
PROB_EXPOSED_DETECTED = 0
# When an individual tests positive, whether their neighbors are quarantined as well.
QUARANTINE_NEIGHBORS = False
_EPSILON = 1e-10
# USA deaths data (worldometers.info): 11 -> 2220 from March 4th to March 28th.
DOUBLING_DAYS = float((28 - 4) / np.log2(2220 / 11))  # About 3.13
# Names of columns in SimulationResults DataFrame.
MAIN_GROUPS = ['susceptible', 'exposed', 'recovered', 'infected', 'quarantined']
# Column list out of the columns in the SimulationResults DataFrame.
ALL_COLUMNS = MAIN_GROUPS + ['test_rate']
# Plotting color conventions.
GROUP2COLOR = dict(susceptible='blue', exposed='orange', recovered='green',
                   quarantined='purple', infected='red', test_rate='brown')
# Attributes to print in SimulationResults summary.
SUMMARY_ATTRS = ['duration', 'fraction_infected', 'doubling_days', 'fraction_quarantine_time', 'peak_infected_time', 'peak_fraction_infected', 'fraction_tests', 'peak_test_rate']


class SimulationResults(object):
  """Simulation run results: metadata, aggregates and time series."""

  def __init__(self, results_df, G=None, **kwargs):
    """Initialize with simulation results, graph and hyper-parameters."""
    self.df = results_df
    self.hyperparams = kwargs
    for name, value in kwargs.items():
      setattr(self, name, value)
    if G is None:
      self.N = results_df[['susceptible', 'exposed', 'infected', 'recovered']].iloc[0].sum()
    else:
      self.N = len(G)
      # Extract graph parameters from its name.
      if G.name.startswith('power_law'):
        self.G_attrs = dict(zip(['gamma', 'min_degree', 'mean_degree'], map(float, G.name.split('_')[-3:])))
        self.G_attrs['N'] = self.N

    if not hasattr(self, 'steps_per_day'):
      self.steps_per_day = ((results_df['step'].iloc[1] - results_df['step'].iloc[0]) /
                            (results_df['day'].iloc[1] - results_df['day'].iloc[0]))

    self.analyze_results_df()

  def calculate_doubling_time(self):
    """Returns doubling time of initial spread (in days)."""
    results_df = self.df
    # Exponential regime before leveling off.
    idx_end = (results_df['exposed'] > results_df['exposed'].max() * 0.5).to_numpy().nonzero()[0][0]
    if self.peak_exposed_time < 3 or idx_end == 0:
      # 3 days peak time, or peak value is less than twice initial value - probably containment.
      return np.inf

    # Don't start from the very beginning, to avoid the initial dip in the
    # number of exposed, due to imperfect starting conditions.
    # Start when number of exposed > 2 * minimum (after passing the minimum).
    exposed_min = results_df['exposed'][:idx_end].min()
    idx_min = results_df['exposed'][:idx_end].idxmin()
    start_candidates = ((results_df.index >= idx_min) &
                        (results_df.index < idx_end) &
                        (results_df['exposed'] > exposed_min * 2)).to_numpy().nonzero()[0]
    if not start_candidates.size:
      # Empty - no candidates. Probably containment.
      return np.inf
    idx_start = start_candidates[0]

    # Linear regression to find doubling time: log2(exposed) ~ day + const
    try:
      X = sm.add_constant(results_df[idx_start:idx_end][['day']], prepend=False)
      log2_exposed = np.log2(results_df[idx_start:idx_end]['exposed'])
      regression_results = sm.OLS(log2_exposed, X).fit()
      # Days for doubling is the inverse of the doubling effect of one day.
      doubling_days = 1 / regression_results.params['day']
    except ValueError:
      doubling_days = None
    return doubling_days

  def calculate_halving_time(self):
    """Returns halving time of spread after peak (in days)."""
    results_df = self.df
    # Find peak.
    idx_peak = results_df['exposed'].idxmax()
    # Find end point for calculation, not right at the peak, but not at the noisy end.
    end_candidates = ((results_df.index >= idx_peak) &
                      (results_df['exposed'] < self.peak_exposed / 5) &
                      (results_df['exposed'] > 5)).to_numpy().nonzero()[0]
    if not end_candidates.size:
      # Halving is too short/noisy to calculate the halving time.
      return None

    idx_end = end_candidates[0]
    idx_start = idx_peak
    if idx_end - idx_start < 20:
      # Halving is too short to calculate the halving time.
      return None

    # Linear regression to find halving time: log2(exposed) ~ day + const
    try:
      X = sm.add_constant(results_df[idx_start:idx_end][['day']], prepend=False)
      log2_exposed = np.log2(results_df[idx_start:idx_end]['exposed'])
      regression_results = sm.OLS(log2_exposed, X).fit()
      # Days for halving is the inverse of the halving effect of one day.
      halving_days = -1 / regression_results.params['day']
    except ValueError:
      halving_days = None
    return halving_days

  def analyze_results_df(self):
    """Calculate various summary stats."""
    results_df = self.df
    self.duration = results_df['day'].iloc[-1]
    # Find peak infections.
    self.peak_infected_time = results_df['day'].iloc[results_df['infected'].idxmax()]
    self.peak_infected = results_df['infected'].max()
    self.peak_fraction_infected = results_df['infected'].max() / self.N
    self.peak_exposed_time = results_df['day'].iloc[results_df['exposed'].idxmax()]
    self.peak_exposed = results_df['exposed'].max()
    self.doubling_days = self.calculate_doubling_time()
    self.halving_days = self.calculate_halving_time()
    # Other result summary stats.
    self.fraction_infected = results_df['recovered'].iloc[-1] / self.N
    # Units: [steps] * [fraction of population]
    fraction_quarantine_steps = results_df['quarantined'].sum() / self.N
    # Units: [days] * [fraction of population]
    self.fraction_quarantine_time = fraction_quarantine_steps / self.steps_per_day
    total_tests = results_df['test_rate'].sum() / self.steps_per_day
    # Number of tests performed, as fraction of the population.
    self.fraction_tests = total_tests / self.N
    self.peak_test_rate = results_df['test_rate'].max() / self.N

  def plot_trends(self, fraction_of_population=True, hyperparams=True, G_attrs=False, columns=None):
    """Plots the time series of self.df, with the specified columns."""
    if columns is None:
      columns = MAIN_GROUPS
    title = ''
    if hyperparams:
      title += str({k: round(v, 3) for k, v in self.hyperparams.items()})
    if G_attrs:
      title += str(self.G_attrs)

    if fraction_of_population:
      scale = self.N
      ylabel = 'Fraction of population'
    else:
      scale = 1
      ylabel = 'Individuals'

    fig, ax_arr = plt.subplots(1, 2)
    fig.suptitle(title, fontsize=18)
    results_to_plot = self.df.drop('step', axis=1).set_index('day') / scale
    results_to_plot = results_to_plot[columns]
    for pane_ind, logy in enumerate([False, True]):
      ax = results_to_plot.plot(ax=ax_arr[pane_ind], logy=logy)
      ax.set_ylabel(ylabel)
    return ax_arr

  def summary(self, hyperparams=False, G_attrs=False, plot=False, **plot_kwargs):
    """Print several summary stats, and potentially plot the trends."""
    summary_dict = {attr_name: getattr(self, attr_name) for attr_name in SUMMARY_ATTRS}
    summary_series_list = [pd.Series(summary_dict)]
    if hyperparams:
      summary_series_list.append(pd.Series(self.hyperparams))
    if G_attrs:
      summary_series_list.append(pd.Series(self.G_attrs))
    print(pd.concat(summary_series_list))
    if plot:
      self.plot_trends(hyperparams=hyperparams, G_attrs=G_attrs, **plot_kwargs)


def get_gamma_distribution_params(mean, std):
  """Turn mean and std of Gamma distribution into parameters k and theta."""
  # mean = k * theta
  # var = std**2 = k * theta**2
  k = std**2 / mean
  theta = mean / k
  return k, theta


def edges2graph(edges, N=N):
  H = nx.MultiGraph()
  H.add_nodes_from(np.arange(N))
  H.add_edges_from(edges)
  return H


def generate_power_law_degrees(N=N, min_degree=MIN_DEGREE, mean_degree=MEAN_DEGREE, gamma=GAMMA):
  """Generate array of degrees according to a power law distribution.

  If U~U[0,1] then generate power law values as
  X = U^(-gamma) - 1
  and then stretch them linearly to provide mean and min as required.
  Then convert them to integers with probability of being rounded up equal to
  the fractional part of the number.
  """
  assert gamma >= 0
  power_law_values = 1 / np.random.random(N)**gamma - 1 + _EPSILON
  value_mean = np.mean(power_law_values)
  frac_degrees = min_degree + power_law_values / value_mean * (
      mean_degree - min_degree)
  int_degrees = frac_degrees.astype(int) + (
      np.random.random(N) < (frac_degrees % 1))
  return int_degrees


def generate_scale_free_graph(N=N, min_degree=MIN_DEGREE, mean_degree=MEAN_DEGREE, gamma=GAMMA):
  """Generates a graph with power-law degree distribution.

  - Draws random degrees according to a power-law distribution.
  - Creates one-sided edges for each node according to its degree.
  - Randomly matches these one-sided edges to another one-sided edge to create
    a single edge.
  - Genereates a graph from this edge list.

  Higher gamma means a fatter tail of the degree distribution.
  Name of returned graph is 'power_law_{gamma}_{min_degree}_{mean_degree}'
  """
  degrees = generate_power_law_degrees(N, min_degree, mean_degree, gamma)
  # pd.Series(degrees).value_counts().sort_index().to_frame().plot(loglog=True)
  nodes_multiple = np.concatenate([np.full(degree, i) for i, degree in enumerate(degrees)])
  np.random.shuffle(nodes_multiple)
  if nodes_multiple.shape[0] % 2 == 1:
    nodes_multiple = nodes_multiple[:-1]
  edges = nodes_multiple.reshape((nodes_multiple.shape[0] // 2, 2))
  H = edges2graph(edges, N)
  # In Graph name, include the parameters.
  H.name = f'power_law_{round(gamma, 3)}_{min_degree}_{mean_degree}'
  return H


def genereate_local_edges(N=N, n_divisions=4, mean_degree=MEAN_DEGREE):
  """Generates edges with high clustering coefficient and constant degree.

  Smaller n_divisions -> higher clustering.
  """
  all_nodes = np.arange(N)
  division_sizes = np.diff(np.linspace(0, mean_degree/2, n_divisions+1).astype(int))
  edges_groups = []
  for division_size in division_sizes:
    np.random.shuffle(all_nodes)
    for i in range(1, division_size + 1):
      # In the random ordering, create edges between each node to the one
      # that is `i` places later.
      edges_groups.append(np.stack([all_nodes[i:], all_nodes[:-i]], axis=1))

  edges = np.concatenate(edges_groups)
  return edges


def generate_local_graph(N=N, n_divisions=4, mean_degree=MEAN_DEGREE):
  """Generates a graph with a much higher clustering coefficient.

  Smaller n_divisions -> higher clustering.
  """
  edges = genereate_local_edges(N, n_divisions, mean_degree)

  H = edges2graph(edges, N)
  # In Graph name, include the parameters.
  H.name = f'local_graph_{n_divisions}_{mean_degree}'
  return H


def genereate_local_edges_from_degrees(degrees, n_divisions=4):
  """Generates edges with high clustering coefficient, using list of degrees.

  Doesn't exactly preserve this list of degrees. It becomes narrower.

  Smaller n_divisions -> higher clustering: clustering coefficient ~ 1/n_divisions
  """
  N = len(degrees)
  all_nodes = np.arange(N)
  divisions_per_node = [np.diff(np.linspace(0, degree, n_divisions+1).astype(int))
                        for degree in degrees]
  edges_groups = []
  for division_ind in range(n_divisions):
    np.random.shuffle(all_nodes)
    for j in range(N):
      node = all_nodes[j]
      n_neighbors = divisions_per_node[node][division_ind]
      # In the random ordering, create edges between each node to the ones
      # that are immediately after it.
      neighbors = all_nodes[j + 1:j + 1 + n_neighbors]
      node_edges = np.stack([np.full_like(neighbors, node), neighbors], axis=1)
      edges_groups.append(node_edges)

  edges = np.concatenate(edges_groups)
  # Drop half of edges, to make the average degree right.
  edges = edges[np.random.random(len(edges)) < 0.5]
  return edges


def generate_local_scale_free_graph(N=N, n_divisions=4, min_degree=MIN_DEGREE,
                                    mean_degree=MEAN_DEGREE, gamma=GAMMA):
  """Generates a scale-free graph with a much higher clustering coefficient.

  WARNING: Resulting graph distribution is not the same as generate_scale_free_graph
    with the sama parameters. This one has a thinner tail.
  Smaller n_divisions -> higher clustering: clustering coefficient ~ 1/n_divisions
  Graph name is 'local_power_law_{n_divisions}_{gamma}_{min_degree}_{mean_degree}'
  """
  degrees = generate_power_law_degrees(N, min_degree, mean_degree, gamma)
  edges = genereate_local_edges_from_degrees(degrees, n_divisions=n_divisions)

  H = edges2graph(edges, N)
  # In Graph name, include the parameters.
  H.name = f'local_power_law_{n_divisions}_{round(gamma, 3)}_{min_degree}_{mean_degree}'
  return H


def random_subset(indicator_arr, sample_prob):
  """In array of 0,1 leave subset of 1s with probability sample_prob.

  sample_prob can also be an array with same shape as indicator_arr, for
  individual heterogeneous sampling probabilities.
  """
  subset_arr = (np.random.random(indicator_arr.shape) < sample_prob) & indicator_arr
  return subset_arr


def init_states(N=N, initial_infected_num=INITIAL_INFECTED_NUM, incubation_duration_mean=INCUBATION_DURATION_MEAN, incubation_duration_std=INCUBATION_DURATION_STD, prob_recover=PROB_RECOVER):
  """Each group has an array of 0/1 indicators, and potentially days left array.

  Susceptible (S_arr), Exposed (E_arr, E_left), Infected (I_arr),
  Recovered (R_arr), Quarantined (Q_arr, Q_left), Tested Positive (TP_arr).
  Indicator value of 1 means the individual is a member of that group.
  *_left arrays are arrays of number of days left until transition to next state.

  Args:
    prob_recover: per day. Inverse of expected duration of being Infected.
  """
  all_nodes_arr = np.arange(N)
  S_arr = np.ones_like(all_nodes_arr)
  E_arr = np.zeros_like(S_arr)
  # Use expected ratio of exposed to infected.
  initial_exposed_num = int(initial_infected_num * incubation_duration_mean * prob_recover)
  E_arr[np.random.choice(all_nodes_arr, initial_exposed_num, replace=False)] = 1
  I_arr = np.zeros_like(S_arr)
  I_arr[np.random.choice(all_nodes_arr, int(initial_infected_num), replace=False)] = 1
  E_arr -= E_arr & I_arr  # E & I are mutually exclusive.
  S_arr -= (I_arr + E_arr)
  R_arr = np.zeros_like(S_arr)
  Q_arr = np.zeros_like(S_arr)
  # Time left to exit quarantine.
  Q_left = np.full_like(S_arr, -1, dtype='float')
  # Time left to become infected.
  E_left = np.full_like(S_arr, -1, dtype='float')
  # Seed with incubation periods.
  incubation_k, incubation_theta = get_gamma_distribution_params(incubation_duration_mean, incubation_duration_std)
  incubation_durations = np.random.gamma(incubation_k, incubation_theta, N)
  np.copyto(E_left, incubation_durations, where=(E_arr > 0), casting='safe')
  TP_arr = np.zeros_like(S_arr)
  return S_arr, E_arr, E_left, I_arr, R_arr, Q_arr, Q_left, TP_arr


def number_of_edges_to_group(group_arr, adj_mat):
  """Calculates number of edges from each node to nodes in group_arr.

  Returns array of integers representing number of edges for each node.
  Multiple edges are counted as their multiplication.

  Args:
    group_arr: Indicator array - 1 indicates group membership.
    adj_mat: Adjacency matrix of the graph.
  """
  # Adjacency matrix filtered down to infectious (not Quarantined)
  adj_mat_group = adj_mat.multiply(group_arr)
  # number of connections to infected (could be multiple connections from the same infected)
  connections_to_group = np.array(adj_mat_group.sum(axis=1))[:, 0]
  return connections_to_group


def individual_infection_probabilities(I_arr, Q_arr, prob_infect, adj_mat):
  """Individual probabilities of being infected by the infectious in I_arr.

  Counts the number of infectious neighbors, and calculates individual
  probability of being infected.
  Probability of being infected by each connection is prob_infect.
  Quarantined (Q_arr) are not infectious.
  """
  # Number of connections to infected (could be multiple connections from the same infected)
  connections_to_I = number_of_edges_to_group(I_arr & (1 - Q_arr), adj_mat)
  # Individual probability of not being infected: (1 - p_infect)**n_connections
  no_infection_probs = np.exp(np.log(1 - prob_infect) * connections_to_I)
  return 1 - no_infection_probs


def infection_step(S_arr, E_arr, E_left, I_arr, Q_arr, adj_mat,
                   incubation_k, incubation_theta, prob_infect, prob_infect_exposed_factor=PROB_INFECT_EXPOSED_FACTOR, duration_exposed_infects=DURATION_EXPOSED_INFECTS):
  """S -> E. Infectious individuals infect their neighbors with some probability.

  Individuals who are infectious are those who are in I but not in Quarantine
  and those who are Exposed (during final infectiousness duration) but not in Quarantine.
  Infectious individuals interact with each of their neighbors once (or multiple
  times if there are multiple edges between them).
  Each interaction with a neighbor infects that neighbor with probability
  prob_infect (for Infected) and prob_infect * prob_infect_exposed_factor (for Exposed),
  but only if that neighbor is Susceptible and not Quarantined.
  These infected individuals become Exposed, and are given an incubation time.

  Args:
    prob_infect: Probability of infection *per step* (not per day).
    incubation_k, incubation_theta: parameters of incubation period length
      in *days*, not steps.
  """
  # TODO: can optimize 2x by creating a composite vector of
  #   I_arr * (1-Q_arr) * np.log(1-prob_infect) + E_infections_arr * (1-Q_arr) * np.log(1-prob_infect_exposed)
  #   and multiply the matrix only once
  # Individual probability being infected by the Infected.
  infection_probs_infected = individual_infection_probabilities(
      I_arr, Q_arr, prob_infect, adj_mat)
  # Same for Exposed who are infectious, though with different probability.
  # Only infectious a few days before developing symptoms.
  E_infectious_arr = E_arr & (E_left <= duration_exposed_infects)
  prob_infect_exposed = prob_infect * prob_infect_exposed_factor
  # Individual probability being infected by the Exposed.
  infection_probs_exposed = individual_infection_probabilities(E_infectious_arr, Q_arr, prob_infect_exposed, adj_mat)
  # Newly infected according to probabilities,
  # but only Susceptible individuals who are not in Quarantine.
  new_infected = random_subset(S_arr & (1 - Q_arr), infection_probs_infected)
  new_infected |= random_subset(S_arr & (1 - Q_arr), infection_probs_exposed)

  # new_infected are now Exposed.
  E_arr += new_infected
  incubation_durations = np.random.gamma(incubation_k, incubation_theta, S_arr.shape[0])
  np.copyto(E_left, incubation_durations, where=(new_infected > 0), casting='safe')
  S_arr -= new_infected
  return S_arr, E_arr


def incubation_step(E_arr, E_left, I_arr, steps_per_day=STEPS_PER_DAY):
  """E -> I. Incubation period ends for some individuals. They become Infected.

  Args:
    E_left: Array of days left in incubation. When reaches 0, become infectious.
  """
  E_left -= 1 / steps_per_day  # Decrease time left in incubation.
  # Those whose incubation period ends.
  become_infected = E_arr & (E_left < _EPSILON)
  E_arr -= become_infected
  I_arr += become_infected
  return E_arr, E_left, I_arr


def recovery_step(I_arr, R_arr, prob_recover):
  """I -> R. Infected individuals Recover with given step-wise probability."""
  new_recovered = (np.random.random(I_arr.shape) < prob_recover) & I_arr
  R_arr += new_recovered
  I_arr -= new_recovered
  return I_arr, R_arr


def testing_step(E_arr, I_arr, TP_arr,
                 prob_infected_detected,
                 prob_neighbor_detected,
                 prob_exposed_detected,
                 adj_mat=None):
  """Testing of subsets of Infected, neighbors, general population.

  A test is positive if a node is either Exposed or Infectious.
  Test symptomatics: a random subset of Infected people are tested (and found positive).
  Mass-testing: a random subset of Exposed & Infected people test positive.
  Contact tracing: a random subset of neighbors of those who tested positive, are also tested.
  Counts number of tests performed, assuming 0 negative tests.
  Known positives are not retested.

  Args:
    TP_arr: Those who tested positive previously. They aren't retested or double-counted.
    prob_neighbor_detected: absolute probability, not per day / step.
    prob_infected_detected, prob_exposed_detected: probability *per step*, not per day.
    adj_mat: not required if prob_neighbor_detected=0.
  """
  # TODO: Testing for other populations (high degree nodes).
  # Infected group is tested, detected with some probability.
  new_TP_arr = random_subset(I_arr & (1 - TP_arr), prob_infected_detected)
  # Assumes 0 negatives (false & true).
  n_infected_tested = new_TP_arr.sum()

  # Random subset of entire population tested, carriers (Exposed or Infected)
  # detected with some probability.
  carrier_arr = E_arr | I_arr
  new_TP_arr |= random_subset(carrier_arr & (1 - TP_arr), prob_exposed_detected)
  N = len(E_arr)
  # N, since entire population was tested, other than known positives.
  # Assumes 0 negatives (false & true). Divide by P(test=positive) for more realistic estimate.
  n_general_tested = (N - TP_arr.sum()) * prob_exposed_detected

  # Neighbors of those who tested positive are themselves tested.
  # Check if neighbors are tested, to avoid matrix multiplication if not required.
  if prob_neighbor_detected > 0:
    connections_to_positive = number_of_edges_to_group(new_TP_arr, adj_mat)
    neighbors_tested = random_subset(connections_to_positive > 0, prob_neighbor_detected).astype(int)
    # Don't double count those who are already known positives, they aren't retested.
    neighbors_tested &= (1 - (new_TP_arr | TP_arr))
    # Assumes 0 negatives (false & true). Divide by P(test=positive) for more realistic estimate.
    n_neighbors_tested = neighbors_tested.sum()
    # Of those who were tested, the ones who tested positive.
    neighbors_detected = neighbors_tested & carrier_arr
    new_TP_arr |= neighbors_detected
  else:
    n_neighbors_tested = 0

  TP_arr |= new_TP_arr

  return TP_arr, n_infected_tested, n_neighbors_tested, n_general_tested


def quarantine_step(E_arr, I_arr, Q_arr, Q_left, R_arr, TP_arr, adj_mat=None,
                    quarantine_neighbors=QUARANTINE_NEIGHBORS,
                    days_in_quarantine=DAYS_IN_QUARANTINE,
                    steps_per_day=STEPS_PER_DAY):
  """Individuals exit quarantine. Those who tested positive enter quarantine.

  Quarantined people whose time is up exit Quarantine.
  Those who tested positive and haven't recovered stay in Quarantine.
  Those who tested positive and have recovered exit Quarantine, even if their time isn't up.
  Those who tested positive enter Quarantine.
  Optionally quarantine extends to neighbors of those who test positive.
  Quarantined individuals can neither infect nor be infected.

  Args:
    Q_left: Array of days left in quarantine. When reaches 0, quarantine ends.
    TP_arr: array indicating those who tested positive.
    quarantine_neighbors: an individual's neighbors are quarantined as well.
  """
  Q_left -= 1 / steps_per_day  # Decrease time left in Quarantine.
  # Known positives who haven't recovered.
  known_positives = TP_arr & (1 - R_arr)
  known_recovered = TP_arr & R_arr
  # Release those whose Quarantine time is up.
  exit_quarantine = Q_arr & (Q_left < _EPSILON)
  # Don't release known positives who haven't recovered.
  exit_quarantine &= (1 - known_positives)
  # Release known positives who have recovered (even if their 14 days aren't up).
  exit_quarantine |= known_recovered & Q_arr
  Q_arr -= exit_quarantine

  new_in_quarantine = known_positives
  if quarantine_neighbors and (known_positives.sum() > 0):
    # Number of connections to those who tested positive.
    connections_to_positive = number_of_edges_to_group(known_positives, adj_mat)
    new_in_quarantine |= (connections_to_positive > 0)

  Q_arr |= new_in_quarantine
  # Quarantine restarts even for those who are already in Quarantine if they
  # test positive or are neighbors of a new positive.
  np.copyto(Q_left, days_in_quarantine, where=(new_in_quarantine > 0), casting='safe')
  return Q_arr, Q_left


def create_counter(I_arr, E_arr, R_arr, Q_arr, S_arr, n_infected_tested, n_neighbors_tested, n_general_tested):
  """Counts number of individuals in each group.

  Returns a dict from group name to sum of its indicator array.
  """
  return dict(infected=I_arr.sum(),
              exposed=E_arr.sum(),
              recovered=R_arr.sum(),
              quarantined=Q_arr.sum(),
              susceptible=S_arr.sum(),
              n_infected_tested=n_infected_tested,
              n_neighbors_tested=n_neighbors_tested,
              n_general_tested=n_general_tested)


def simulation(N=N,
               G=None,
               initial_infected_num=INITIAL_INFECTED_NUM,
               prob_infect=PROB_INFECT,
               incubation_duration_mean=INCUBATION_DURATION_MEAN,
               incubation_duration_std=INCUBATION_DURATION_STD,
               prob_recover=PROB_RECOVER,
               quarantine_neighbors=QUARANTINE_NEIGHBORS,
               prob_infected_detected=PROB_INFECTED_DETECTED,
               prob_neighbor_detected=PROB_NEIGHBOR_DETECTED,
               prob_exposed_detected=PROB_EXPOSED_DETECTED,
               days_in_quarantine=DAYS_IN_QUARANTINE,
               prob_infect_exposed_factor=PROB_INFECT_EXPOSED_FACTOR,
               duration_exposed_infects=DURATION_EXPOSED_INFECTS,
               steps_per_day=STEPS_PER_DAY,
               min_degree=MIN_DEGREE,
               mean_degree=MEAN_DEGREE,
               gamma=GAMMA,
               max_steps=MAX_STEPS,
               verbose=False):
  """Simulates SEQIR model on a specified network. Includes Quarantine group.

  All probabilities in args are *daily* probabilities.
  """
  # Process input.
  if G is None:
    if verbose:
      print('Generating graph...')
    G = generate_scale_free_graph(N, min_degree=min_degree, mean_degree=mean_degree, gamma=gamma)
    if verbose:
      print('Done!')
  else:
    N = G.number_of_nodes()
  adj_mat = nx.adjacency_matrix(G)

  initial_infected_num = int(initial_infected_num)  # Make sure it's an integer.
  incubation_k, incubation_theta = get_gamma_distribution_params(incubation_duration_mean, incubation_duration_std)

  S_arr, E_arr, E_left, I_arr, R_arr, Q_arr, Q_left, TP_arr = init_states(
      N, initial_infected_num=initial_infected_num, incubation_duration_mean=incubation_duration_mean,
      incubation_duration_std=incubation_duration_std, prob_recover=prob_recover)
  n_infected_tested = n_neighbors_tested = n_general_tested = 0
  # Step-wise probabilities.
  prob_infect_per_step = prob_infect / steps_per_day
  prob_infected_detected_per_step = prob_infected_detected / steps_per_day
  prob_exposed_detected_per_step = prob_exposed_detected / steps_per_day
  prob_recover_per_step = prob_recover / steps_per_day

  max_step_wise_prob = max(prob_infect_per_step, prob_infected_detected_per_step,
                           prob_exposed_detected_per_step, prob_recover_per_step)
  if max_step_wise_prob > 0.5:
    print('WARNING: steps_per_day too small? Maximal step-wise probability =', max_step_wise_prob)

  counters = []

  # Main loop.
  for step_num in range(max_steps):
    # Save metrics.
    counters.append(create_counter(I_arr, E_arr, R_arr, Q_arr, S_arr,
                                   n_infected_tested, n_neighbors_tested, n_general_tested))

    # Infection step. S -> E
    S_arr, E_arr = infection_step(S_arr, E_arr, E_left, I_arr, Q_arr, adj_mat,
                                  incubation_k=incubation_k,
                                  incubation_theta=incubation_theta,
                                  prob_infect=prob_infect_per_step,
                                  prob_infect_exposed_factor=prob_infect_exposed_factor,
                                  duration_exposed_infects=duration_exposed_infects)

    # Incubation step. E -> I
    E_arr, E_left, I_arr = incubation_step(E_arr, E_left, I_arr,
                                           steps_per_day=steps_per_day)

    # Testing step. Update tested_positive.
    TP_arr, n_infected_tested, n_neighbors_tested, n_general_tested = testing_step(
        E_arr, I_arr, TP_arr,
        prob_infected_detected=prob_infected_detected_per_step,
        prob_neighbor_detected=prob_neighbor_detected,
        prob_exposed_detected=prob_exposed_detected_per_step,
        adj_mat=adj_mat)

    # Quarantine step. Update Q
    Q_arr, Q_left = quarantine_step(
        E_arr, I_arr, Q_arr, Q_left, R_arr, TP_arr, adj_mat,
        quarantine_neighbors=quarantine_neighbors,
        days_in_quarantine=days_in_quarantine,
        steps_per_day=steps_per_day)

    # Recovery step. I -> R
    I_arr, R_arr = recovery_step(I_arr, R_arr, prob_recover_per_step)

    # Stop condition. No more Infected or Exposed.
    if I_arr.sum() == 0 and E_arr.sum() == 0:
      break
  else:
    print('WARNING: max number of steps reached.')

  # Add final counters.
  counters.append(create_counter(I_arr, E_arr, R_arr, Q_arr, S_arr,
                                 n_infected_tested, n_neighbors_tested, n_general_tested))

  # Convert to DataFrame.
  results_df = pd.DataFrame(counters)
  tested = results_df.filter(regex='_tested').sum(axis=1)
  # Test rate: number of tests per day. Smoothe using day-length sliding window.
  results_df['test_rate'] = np.convolve(tested, np.ones((steps_per_day,)), mode='same')
  results_df['step'] = np.arange(len(results_df))
  results_df['day'] = results_df['step'] / steps_per_day
  results = SimulationResults(results_df, G, prob_infect=prob_infect,
                              prob_infected_detected=prob_infected_detected,
                              prob_neighbor_detected=prob_neighbor_detected,
                              prob_exposed_detected=prob_exposed_detected,
                              quarantine_neighbors=quarantine_neighbors,
                              steps_per_day=steps_per_day)
  if verbose:
    print('Simulation finished!')
  return results
