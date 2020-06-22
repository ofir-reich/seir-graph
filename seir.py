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
# to infect a Susceptible. Fitted to produce doubling_time=3.1 For G with parameters
# (100000, 2, 20, 0.2) and 40% asymptomatic, along with all other defaults.
PROB_INFECT = 0.027
# Ratio of infectiousness between Exposed (in the final days before becoming
# Infected) and Infected. In other words,
# prob_infect_exposed = prob_infect * prob_infect_exposed_factor.
# Set this to 0 to make the Exposed non-infectious.
PROB_INFECT_EXPOSED_FACTOR = 0.5
# Ratio of infectiousness between Asymptomatic and symptomatic. In other words,
# prob_infect_asymptomatic = prob_infect * relative_infectiousness_asymptomatic.
# The same ratio holds for the final contagious part of the incubation period.
# Set this to 0 to make the Asymptomatic non-infectious.
# Source: Assumed in a few places. This paper might claim 0.5: https://science.sciencemag.org/content/368/6490/489
RELATIVE_INFECTIOUSNESS_ASYMPTOMATIC = 0.5
# How many days before developing symptoms (becoming Infected) an Exposed
# individual is contagious. Questionable evidence, but 1-2 produces a reasonable R0.
DURATION_EXPOSED_INFECTS = 2
# Incubation period duration distribution mean. In days.
# Source: 3rd Imperial College paper: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Europe-estimates-and-NPI-impact-30-03-2020.pdf
INCUBATION_DURATION_MEAN = 5.1
# Incubation period duration distribution standard deviation. In days.
# Source: 3rd Imperial College paper: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Europe-estimates-and-NPI-impact-30-03-2020.pdf
INCUBATION_DURATION_STD = 4.38
# Share of population who would not develop symptoms even when sick. They are
# not detected along with the symptomatic Infected in the testing step, but are
# detected by direct tests (mass testing or when neighbors are tested).
# Source for default: Vo, Italy study and others (See Eric Topol paper): https://www.scripps.edu/science-and-medicine/translational-institute/about/news/sarc-cov-2-infection/index.html
PROB_ASYMPTOMATIC = 0.40
# Probability to change from Infections to Recovered in a single day.
# Equals the inverse of the expected time to Recover.
# 3.5 = 0.3 * 0 + 0.56 * 5 + 0.1 * 5 + 0.04 * 5. Source: Pueyo spreadsheet: https://docs.google.com/spreadsheets/d/1uJHvBubps9Z2Iw_-a_xeEbr3-gci6c475t1_bBVkarc/edit#gid=0
PROB_RECOVER = 1 / 3.5
# Length of imposed quarantine, in days.
DAYS_IN_QUARANTINE = 14
# Probability that an Infected individual is tested and detected as a carrier in a single day.
# This encapsulates also the fact that many people are mildly symptomatic or will
# not be tested (or quarantined).
# Default: no tests at all.
PROB_INFECTED_DETECTED = 0
# Probability that a neighbor of an individual who tested positive is himself traced,
# and then tested / quarantined. The testing of neighbors happens once, once the
# original individual is tested positive, so this is *not* a probability per day.
# Default - no neighbors are traced.
PROB_NEIGHBOR_TRACED = 0
# Share of the general population which is tested on a single day. Exposed and
# Infected individuals who are tested are detected as carriers.
# Default - previous behavior, no exposed are detected.
PROB_EXPOSED_DETECTED = 0
# When an individual tests positive, whether their traced neighbors are quarantined.
QUARANTINE_NEIGHBORS = False
# When an individual tests positive, whether their traced neighbors are tested.
TEST_NEIGHBORS = False
# Delay between the time a test is performed and the time results become known
# to the subject, and contact tracing is complete. Days.
TEST_DELAY_TIME = 0

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

  def plot_trends(self, fraction_of_population=True, hyperparams=True, G_attrs=False, columns=None, vertical=False):
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

    if vertical:
      fig, ax_arr = plt.subplots(2, 1, figsize=(10, 20))
      ax_arr[0].set_title('Epidemic Simulation')
      ax_arr[1].set_title('log scale')
    else:
      fig, ax_arr = plt.subplots(1, 2)
    fig.suptitle(title, fontsize=18)
    results_to_plot = self.df.drop('step', axis=1).set_index('day') / scale
    results_to_plot = results_to_plot[columns]
    for pane_ind, logy in enumerate([False, True]):
      ax = results_to_plot.plot(ax=ax_arr[pane_ind], logy=logy)
      ax.set_ylabel(ylabel)
      if logy:
        ax.get_legend().remove()
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
  theta = std**2 / mean
  k = mean / theta
  return k, theta


def edges2graph(edges, N=N):
  """Creates MultiGraph from list of edges."""
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


def random_subset(indicator_arr, sample_probs):
  """In array of 0,1 leave subset of 1s with probabilities sample_probs.

  Args:
    indicator_arr: An array of 0,1 values.
    sample_probs: A number or an array with same shape as indicator_arr, for
      individual heterogeneous sampling probabilities.
  """
  subset_arr = (np.random.random(indicator_arr.shape) < sample_probs) & indicator_arr
  return subset_arr


def init_states(N=N,
                initial_infected_num=INITIAL_INFECTED_NUM,
                incubation_duration_mean=INCUBATION_DURATION_MEAN,
                incubation_duration_std=INCUBATION_DURATION_STD,
                prob_recover=PROB_RECOVER,
                prob_asymptomatic=PROB_ASYMPTOMATIC):
  """Init the state variables of the simulation.

  Each group has an array of 0/1 indicators, and potentially days left array.
  Susceptible (S_arr), Exposed (E_arr, E_left), Infected (I_arr), Asymptomatic (A_arr),
  Recovered (R_arr), Quarantined (Q_arr, Q_left), Tested Positive (TP_arr),
  Time until test results return (T_result_left), what the test results wil be
  when they return (T_result_positive_arr).
  i-th indicator value of 1 means the i-th node is a member of that group.
  *_left are arrays of number of days left until transition to next state. A
  value of -1 or less means Not Available. For example an individual not in the
  E group (i.e. has E_arr[i]==0) does not have a meaningful E_left[i], so it
  will be -1 or less.

  Args:
    prob_recover: per day. Inverse of expected duration of being Infected.
    prob_asymptomatic: Share of population who would not develop symptoms even
      when sick. They are not detected along with the symptomatic Infected in
      the testing step, but are detected by direct tests (mass testing or when
      neighbors are tested).
  """
  all_nodes_arr = np.arange(N)
  S_arr = np.ones_like(all_nodes_arr)
  # Use expected ratio of exposed to infected.
  initial_exposed_num = int(initial_infected_num * incubation_duration_mean * prob_recover)
  initial_infected_num = int(initial_infected_num)
  # Draw E & I, mutually exclusive.
  exposed_and_infected_inds = np.random.choice(all_nodes_arr, initial_exposed_num + initial_infected_num, replace=False)
  exposed_inds = exposed_and_infected_inds[:initial_exposed_num]
  infected_inds = exposed_and_infected_inds[initial_exposed_num:]
  E_arr = np.zeros_like(S_arr)
  E_arr[exposed_inds] = 1
  I_arr = np.zeros_like(S_arr)
  I_arr[infected_inds] = 1
  A_arr = random_subset(np.ones_like(S_arr), prob_asymptomatic)
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
  # Time left until test results come back.
  T_result_left = np.full_like(S_arr, -1)
  # What the test results will be, when they come back.
  T_result_positive_arr = np.zeros_like(S_arr)
  return S_arr, E_arr, E_left, I_arr, A_arr, R_arr, Q_arr, Q_left, TP_arr, T_result_left, T_result_positive_arr


def number_of_edges_to_group(group_arr, adj_mat):
  """Calculates number of edges from each node to nodes in group_arr.

  Returns array of integers representing number of edges for each node.
  Multiple edges are counted as their multiplication.

  Args:
    group_arr: Indicator array - 1 indicates group membership.
    adj_mat: Adjacency matrix of the graph.

  Returns:
    Array indicating for each member of the population the number of neighbors
    they have in the group indicated by group_arr.
  """
  if (group_arr == 0).all():
    # Performance optimization. If the group is empty, don't count edges.
    return np.zeros_like(group_arr)
  # Adjacency matrix filtered down to neighbors of group.
  adj_mat_group = adj_mat.multiply(group_arr)
  # number of connections to infected (could be multiple connections from the same infected)
  connections_to_group = np.array(adj_mat_group.sum(axis=1))[:, 0]
  return connections_to_group


def spread_to_neighbors(from_group_arr, prob_spread, adj_mat, to_group=None):
  """Neighbors of the group are marked with a certain probability for each edge.

  Each member of the group denoted by from_group_arr spreads to each one of its
  neighbors with probability prob_spread. This can be used for infection,
  tracing or anything else.

  Args:
    from_group_arr: Indicator array of seed group, from which the spread emanates.
    prob_spread: Probability that each member of the group "marks" or spreads
      to each of its neighbors.
    adj_mat: Adjacency matrix of the graph.
    to_group: Indicator array. Restrict spread to members of this group. Default
      is no restriction - the entire population.

  Returns:
    Indicator array of 1s for those who were marked.
  """
  spread_probs = individual_infection_probabilities(from_group_arr, prob_spread, adj_mat)
  if to_group is None:
    to_group = np.ones_like(from_group_arr)
  return random_subset(to_group, spread_probs)


def individual_infection_probabilities(I_arr, prob_infect, adj_mat):
  """Individual probabilities of being infected by the infectious in I_arr.

  Counts the number of infectious neighbors, and calculates individual
  probability of being infected.
  Probability of being infected by each connection is prob_infect.

  Args:
    prob_infect: Probability of infecting a single neighbor.
    adj_mat: Adjacency matrix of the graph.
  """
  if prob_infect == 0:
    # Performance optimization - Don't count edges when no chance of infection.
    return np.zeros_like(I_arr)
  # Number of connections to infected (could be multiple connections from the same infected)
  connections_to_I = number_of_edges_to_group(I_arr, adj_mat)
  # Individual probability of not being infected: (1 - p_infect)**n_connections
  no_infection_probs = np.exp(np.log(1 - prob_infect) * connections_to_I)
  return 1 - no_infection_probs


def infection_step(S_arr, E_arr, E_left, I_arr, A_arr, Q_arr, adj_mat,
                   incubation_k, incubation_theta, prob_infect,
                   prob_infect_exposed_factor=PROB_INFECT_EXPOSED_FACTOR,
                   relative_infectiousness_asymptomatic=RELATIVE_INFECTIOUSNESS_ASYMPTOMATIC,
                   duration_exposed_infects=DURATION_EXPOSED_INFECTS):
  """S -> E. Infectious individuals infect their neighbors with some probability.

  Individuals who are infectious are those who are in I but not in Quarantine
  and those who are Exposed (during final infectiousness duration) but not in Quarantine.
  Infectious individuals interact with each of their neighbors once (or multiple
  times if there are multiple edges between them).
  Each interaction with a neighbor infects that neighbor with probability
  prob_infect (for Infected) and prob_infect * prob_infect_exposed_factor (for Exposed),
  but only if that neighbor is Susceptible and not Quarantined.
  For Asymptomatics, their probabilities of infecting (in both Infected and Exposed)
  are multiplied by relative_infectiousness_asymptomatic.

  All infected individuals become Exposed, and are given an incubation time.

  Args:
    prob_infect: Probability of infection *per step* (not per day).
    incubation_k, incubation_theta: parameters of incubation period length
      in *days*, not steps.
    prob_infect_exposed_factor: Ratio of infectiousness between Exposed (in the
      final days before becoming Infected) and Infected. In other words,
      prob_infect_exposed = prob_infect * prob_infect_exposed_factor.
      Set this to 0 to make the Exposed non-infectious.
    relative_infectiousness_asymptomatic: Ratio of infectiousness between
      Asymptomatic and symptomatic. In other words,
      prob_infect_asymptomatic = prob_infect * relative_infectiousness_asymptomatic.
      The same ratio holds for the final contagious part of the incubation period.
      Set this to 0 to make the Asymptomatic non-infectious.
    duration_exposed_infects: How many days before developing symptoms (becoming
      Infected) an Exposed individual is contagious.
  """
  # TODO: Can unite these 4 matrix multiplications into 1, to optimize.
  # Those who can be infected - Quarantined cannot be.
  susceptible_not_quarantined = S_arr & (1 - Q_arr)

  # Newly infected by symptomatic Infected who are not Quarantined.
  new_infected = spread_to_neighbors(
      I_arr & (1 - A_arr) & (1 - Q_arr), prob_infect, adj_mat, susceptible_not_quarantined)

  # Newly infected by Asymptomatic Infected who are not Quarantined.
  prob_infect_asymptomatic = prob_infect * relative_infectiousness_asymptomatic
  new_infected |= spread_to_neighbors(
      I_arr & A_arr & (1 - Q_arr), prob_infect_asymptomatic, adj_mat, susceptible_not_quarantined)

  # Newly infected by the Exposed who are infectious, and not Quarantined.
  # Only infectious a few days before developing symptoms.
  E_infectious_arr = E_arr & (E_left <= duration_exposed_infects)
  # Exposed are less infectious.
  prob_infect_exposed = prob_infect * prob_infect_exposed_factor
  # Infections by Exposed who will become symptomatic.
  new_infected |= spread_to_neighbors(
      E_infectious_arr & (1 - A_arr) & (1 - Q_arr), prob_infect_exposed,
      adj_mat, susceptible_not_quarantined)
  # Infections by Exposed who will become Asymptomatic.
  prob_infect_exposed_asymptomatic = prob_infect_exposed * relative_infectiousness_asymptomatic
  new_infected |= spread_to_neighbors(
      E_infectious_arr & A_arr & (1 - Q_arr), prob_infect_exposed_asymptomatic,
      adj_mat, susceptible_not_quarantined)

  # new_infected are now Exposed.
  E_arr = E_arr | new_infected
  incubation_durations = np.random.gamma(incubation_k, incubation_theta, S_arr.shape[0])
  np.copyto(E_left, incubation_durations, where=(new_infected > 0), casting='safe')
  S_arr = S_arr & (1 - new_infected)
  return S_arr, E_arr


def incubation_step(E_arr, E_left, I_arr, steps_per_day=STEPS_PER_DAY):
  """E -> I. Incubation period ends for some individuals. They become Infected.

  Args:
    E_left: Array of days left in incubation. When reaches 0, become infectious.
  """
  E_left = E_left - 1 / steps_per_day  # Decrease time left in incubation.
  # Those whose incubation period ends.
  become_infected = E_arr & (E_left < _EPSILON)
  E_arr = E_arr - become_infected
  I_arr = I_arr + become_infected
  return E_arr, E_left, I_arr


def recovery_step(I_arr, R_arr, prob_recover):
  """I -> R. Infected individuals Recover with given step-wise probability."""
  new_recovered = (np.random.random(I_arr.shape) < prob_recover) & I_arr
  R_arr = R_arr + new_recovered
  I_arr = I_arr - new_recovered
  return I_arr, R_arr


def tests_analyzed(T_result_left, T_result_positive_arr, steps_per_day):
  """Time left for results to come back decreased, some results come back.

  Args:
    T_result_left: Array. For each agent, number days left until test result comes back.
    T_result_positive_arr: Whether test results (When they come back) will be positive.

  Returns:
    new_TP_arr: New positives.
    T_result_left: Updated, with less time left.
  """
  # Decrease time left for test results to come back.
  T_result_left = T_result_left - 1. / steps_per_day
  # Test results come back for some agents.
  test_comes_back_arr = ((T_result_left < _EPSILON) & (T_result_left > -1)).astype(int)
  new_TP_arr = test_comes_back_arr & T_result_positive_arr
  # No longer waiting for test results.
  T_result_left = np.where(test_comes_back_arr, -1, T_result_left)
  return new_TP_arr, T_result_left


def test_group(tested_arr, carrier_arr, T_result_left, T_result_positive_arr, test_delay_time):
  """Tests a group, returning test results, results time, number tested.

  Args:
    tested_arr: Indicator array for those to be tested.
    carrier_arr: Of tested, those who will test positive. Only its logical and
      with tested_arr is used.
    T_result_left: Array. For each agent, number days left until test result comes back.
    T_result_positive_arr: Whether test results (When they come back) will be positive.
    test_delay_time: Delay between the time a test is performed and the time
      results return and contact-traced neighbors are tested/quarantined. Days.
  """
  # Record positive results for the positives (when they come back)
  T_result_positive_arr = T_result_positive_arr | (tested_arr & carrier_arr)
  # Start the clock on new tests being analyzed.
  T_result_left = np.where(tested_arr, test_delay_time, T_result_left)
  # Assumes 0 negatives (false & true). Divide by P(test=positive) for more realistic estimate.
  n_tested = tested_arr.sum()

  return T_result_left, T_result_positive_arr, n_tested


def symptomatics_tested(I_arr, A_arr, TP_arr, new_TP_arr, T_result_left, T_result_positive_arr, prob_infected_detected, test_delay_time):
  """Some fraction of symptomatics is tested. Asymptomatic patients aren't tested.

  Args:
    I_arr: Indicator array of Infected.
    A_arr: Asymptomatics. Those who would not develop symptoms even if infected.
    TP_arr: Those who tested positive previously. They aren't retested.
    new_TP_arr: Those whose positive test results came back in this step.
    T_result_left: Array. For each agent, number days left until test result comes back.
    T_result_positive_arr: Whether test results (When they come back) will be positive.
    prob_infected_detected: probability *per step*, not per day.
    test_delay_time: Delay between the time a test is performed and the time
      results return and contact-traced neighbors are tested/quarantined. Days.

  Returns:
    T_result_left, T_result_positive_arr, n_infected_tested.
  """
  # Infected group is tested with some probability. Except Asymptomatic.
  new_symptomatic_tested_arr = random_subset(I_arr & (1 - A_arr), prob_infected_detected)
  # Known positives and those pending test results aren't retested.
  pending_test_results_arr = (T_result_left > 0).astype(int)
  new_symptomatic_tested_arr &= (1 - (TP_arr | new_TP_arr | pending_test_results_arr))

  T_result_left, T_result_positive_arr, n_infected_tested = test_group(
      new_symptomatic_tested_arr, I_arr, T_result_left, T_result_positive_arr, test_delay_time)

  return T_result_left, T_result_positive_arr, n_infected_tested, new_symptomatic_tested_arr


def general_population_tested(E_arr, I_arr, TP_arr, new_TP_arr, T_result_left, T_result_positive_arr, prob_exposed_detected, test_delay_time):
  """Some fraction of the general population is tested.

  Args:
    E_arr: Indicator array of Exposed.
    I_arr: Indicator array of Infected.
    TP_arr: Those who tested positive previously. They aren't retested.
    new_TP_arr: Those whose positive test results came back in this step.
    T_result_left: Array. For each agent, number days left until test result comes back.
    T_result_positive_arr: Whether test results (When they come back) will be positive.
    prob_exposed_detected: Share of general population which is tested, *per step*.
    test_delay_time: Delay between the time a test is performed and the time
      results return and contact-traced neighbors are tested/quarantined. Days.

  Returns:
    T_result_left, T_result_positive_arr, n_infected_tested.
  """
  # Known positives and those pending test results aren't retested.
  pending_test_results_arr = (T_result_left > 0).astype(int)
  # Random subset of entire population tested,
  # except known positives and those pending test results.
  new_genpop_tested_arr = random_subset(1 - (TP_arr | new_TP_arr | pending_test_results_arr),
                                        prob_exposed_detected)

  # Carriers (Exposed or Infected) will have positive test results (when back).
  carrier_arr = E_arr | I_arr
  T_result_left, T_result_positive_arr, n_general_tested = test_group(
      new_genpop_tested_arr, carrier_arr, T_result_left, T_result_positive_arr, test_delay_time)

  return T_result_left, T_result_positive_arr, n_general_tested


def contact_tracing(E_arr, I_arr, TP_arr, new_TP_arr, T_result_left,
                    T_result_positive_arr, new_Q_arr, prob_neighbor_traced,
                    quarantine_neighbors, test_neighbors, test_delay_time, adj_mat):
  """Contact tracing. Neighbors of positives are traced and tested / quarantined.

  For each positive, each neighbor is traced with probability prob_neighbor_traced.

  Args:
    quarantine_neighbors: When an individual tests positive, whether their
      traced neighbors are quarantined.
    test_neighbors: When an individual tests positive, whether their traced
      neighbors are tested.

  Returns:
    new_Q_arr, T_result_left, T_result_positive_arr, n_neighbors_traced, n_neighbors_tested.
  """
  # Check if no neighbors are tested, to avoid matrix multiplication if not required.
  if prob_neighbor_traced == 0:
    n_neighbors_tested = 0
    n_neighbors_traced = 0
    return new_Q_arr, T_result_left, T_result_positive_arr, n_neighbors_traced, n_neighbors_tested

  neighbors_traced = spread_to_neighbors(new_TP_arr, prob_neighbor_traced, adj_mat)
  n_neighbors_traced = neighbors_traced.sum()
  n_neighbors_tested = 0

  if quarantine_neighbors:
    # Quarantine traced neighbors.
    # Known positives aren't newly quarantined (either covered elsewhere or recovered).
    new_Q_arr = new_Q_arr | (neighbors_traced & (1 - (new_TP_arr | TP_arr)))

  if test_neighbors:
    # Test the traced neighbors.
    # Carriers (Exposed or Infected) will have positive test results (when back).
    carrier_arr = E_arr | I_arr
    # Known positives and those pending test results aren't retested.
    pending_test_results_arr = (T_result_left > 0).astype(int)
    neighbors_tested = neighbors_traced & (1 - (TP_arr | new_TP_arr | pending_test_results_arr))
    T_result_left, T_result_positive_arr, n_neighbors_tested = test_group(
        neighbors_tested, carrier_arr, T_result_left, T_result_positive_arr, test_delay_time)

  return new_Q_arr, T_result_left, T_result_positive_arr, n_neighbors_traced, n_neighbors_tested


def testing_step(E_arr, I_arr, A_arr, TP_arr, T_result_left, T_result_positive_arr,
                 prob_infected_detected,
                 prob_neighbor_traced,
                 prob_exposed_detected,
                 quarantine_neighbors,
                 test_neighbors,
                 test_delay_time,
                 adj_mat=None,
                 steps_per_day=STEPS_PER_DAY):
  """Testing of subsets of Infected, the general population. Contact tracing.

  A test comes out positive if a node is either Exposed or Infected.
  Test symptomatics: a random subset of Infected people are tested (and found
    positive). Symptomatics are quarantined even before test results return.
  Mass-testing: a random subset of the population is tested (Exposed & Infected
    people test positive.)
  Contact tracing: a random subset of neighbors of those who tested positive,
  are traced, and then either tested or quarantined without being tested.
  Counts number of tests performed, assuming 0 false negative tests.
  Known positives and those pending test results are not retested.

  Args:
    A_arr: Asymptomatics. Those who would not develop symptoms even if infected.
    TP_arr: Those who tested positive previously. They aren't retested.
    T_result_left: Array. For each agent, number days left until test result comes back.
    T_result_positive_arr: Whether test results (When they come back) will be positive.
    prob_infected_detected: probability *per step*, not per day.
    prob_neighbor_traced: absolute probability, not per day / step.
    prob_exposed_detected: Share of general population which is tested, *per step*.
    quarantine_neighbors: When an individual tests positive, whether their
      traced neighbors are quarantined.
    test_neighbors: When an individual tests positive, whether their traced
      neighbors are tested.
    test_delay_time: Delay between the time a test is performed and the time
      results return and contact-traced neighbors are tested/quarantined. Days.
    adj_mat: not required if prob_neighbor_traced=0.

  Returns:
    TP_arr, new_Q_arr, T_result_left, T_result_positive_arr, n_infected_tested,
    n_neighbors_traced, n_neighbors_tested, n_general_tested
  """
  # TODO: Testing for other populations (high degree nodes).
  new_TP_arr, T_result_left = tests_analyzed(
      T_result_left, T_result_positive_arr, steps_per_day)

  (T_result_left, T_result_positive_arr, n_infected_tested,
   new_symptomatic_tested_arr) = symptomatics_tested(
       I_arr, A_arr, TP_arr, new_TP_arr, T_result_left, T_result_positive_arr,
       prob_infected_detected, test_delay_time)

  T_result_left, T_result_positive_arr, n_general_tested = general_population_tested(
      E_arr, I_arr, TP_arr, new_TP_arr, T_result_left, T_result_positive_arr,
      prob_exposed_detected, test_delay_time)

  # Indicator for those who need to enter Quarantine. New positives and symptomatic.
  new_Q_arr = new_TP_arr | new_symptomatic_tested_arr
  # TODO: Add non-covid symptomatics who are quarantined, and released on negative test results.

  (new_Q_arr, T_result_left, T_result_positive_arr, n_neighbors_traced,
   n_neighbors_tested) = contact_tracing(E_arr, I_arr, TP_arr, new_TP_arr,
                                         T_result_left, T_result_positive_arr,
                                         new_Q_arr, prob_neighbor_traced,
                                         quarantine_neighbors, test_neighbors,
                                         test_delay_time, adj_mat)
  TP_arr = TP_arr | new_TP_arr
  return TP_arr, new_Q_arr, T_result_left, T_result_positive_arr, n_infected_tested, n_neighbors_traced, n_neighbors_tested, n_general_tested


def quarantine_step(Q_arr, Q_left, R_arr, TP_arr, new_Q_arr,
                    days_in_quarantine=DAYS_IN_QUARANTINE,
                    steps_per_day=STEPS_PER_DAY):
  """Individuals exit quarantine. Those who tested positive enter quarantine.

  Quarantined people whose time is up exit Quarantine.
  Those who tested positive and haven't recovered stay in Quarantine.
  Those who tested positive and have recovered exit Quarantine, even if their time isn't up.
  Those who should (tested positive or traced neighbors) enter Quarantine.
  Quarantined individuals can neither infect nor be infected.

  Args:
    Q_left: Array of days left in quarantine. When reaches 0, quarantine ends.
    TP_arr: array indicating those who tested positive.
    new_Q_arr: array indicating those who should enter Quarantine in this step.
  """
  # Exiting quarantine
  # ------------------
  Q_left = Q_left - 1. / steps_per_day  # Decrease time left in Quarantine.
  # Known positives who haven't recovered.
  known_positives = TP_arr & (1 - R_arr)
  known_recovered = TP_arr & R_arr
  # Release those whose Quarantine time is up.
  exit_quarantine = Q_arr & (Q_left < _EPSILON)
  # Don't release known positives who haven't recovered.
  exit_quarantine &= (1 - known_positives)
  # Release known positives who have recovered (even if their 14 days aren't up).
  exit_quarantine |= known_recovered & Q_arr
  # Apply exiting quarantine.
  Q_arr &= (1 - exit_quarantine)
  # Entering quarantine
  # ------------------
  new_in_quarantine = new_Q_arr & (1 - Q_arr)
  Q_arr |= new_in_quarantine
  # Start the clock on time in Quarantine.
  np.copyto(Q_left, days_in_quarantine, where=(new_in_quarantine > 0), casting='safe')
  return Q_arr, Q_left


def create_counter(I_arr, E_arr, R_arr, Q_arr, S_arr, n_infected_tested, n_neighbors_traced, n_neighbors_tested, n_general_tested):
  """Counts number of individuals in each group.

  Returns a dict from group name to sum of its indicator array.
  """
  return dict(infected=I_arr.sum(),
              exposed=E_arr.sum(),
              recovered=R_arr.sum(),
              quarantined=Q_arr.sum(),
              susceptible=S_arr.sum(),
              n_infected_tested=n_infected_tested,
              n_neighbors_traced=n_neighbors_traced,
              n_neighbors_tested=n_neighbors_tested,
              n_general_tested=n_general_tested)


def simulation(N=N,
               G=None,
               initial_infected_num=INITIAL_INFECTED_NUM,
               prob_infect=PROB_INFECT,
               incubation_duration_mean=INCUBATION_DURATION_MEAN,
               incubation_duration_std=INCUBATION_DURATION_STD,
               prob_asymptomatic=PROB_ASYMPTOMATIC,
               prob_recover=PROB_RECOVER,
               quarantine_neighbors=QUARANTINE_NEIGHBORS,
               test_neighbors=TEST_NEIGHBORS,
               prob_infected_detected=PROB_INFECTED_DETECTED,
               prob_neighbor_traced=PROB_NEIGHBOR_TRACED,
               prob_exposed_detected=PROB_EXPOSED_DETECTED,
               test_delay_time=TEST_DELAY_TIME,
               days_in_quarantine=DAYS_IN_QUARANTINE,
               prob_infect_exposed_factor=PROB_INFECT_EXPOSED_FACTOR,
               relative_infectiousness_asymptomatic=RELATIVE_INFECTIOUSNESS_ASYMPTOMATIC,
               duration_exposed_infects=DURATION_EXPOSED_INFECTS,
               steps_per_day=STEPS_PER_DAY,
               min_degree=MIN_DEGREE,
               mean_degree=MEAN_DEGREE,
               gamma=GAMMA,
               max_steps=MAX_STEPS,
               verbose=False):
  """Simulates SEQIR model on a network.

  A Susceptible-Exposed-Infected-Recovered (SEIR) model for disease progression.
  This is an agent-based model, where each agent (person) is represented as a
  node in a graph, and infection happens between contacts, represented by graph
  edges. It also models testing and a Quarantine group of those who test positive,
  and contact tracing of their neighbors in the graph.
  At any given step, each node belongs to exactly one of the following groups:

  Susceptible (S): Nodes which weren’t infected yet. These are the only nodes
    which can be infected. When infected, they become Exposed.

  Exposed (E): Nodes which were infected and are now in their pre-symptomatic
    incubation period. They are infectious starting 2 days before the end of
    their (Gamma distributed) incubation period, at which point they develop
    symptoms (and become Infected).

  Infected (I): Nodes which are now symptomatic. More infectious than during the
    final Exposed period. When their (exponentially distributed) Infected period
    ends, they become Recovered.

  Recovered (R): Nodes who have recovered (or died). They are no longer
    infectious and cannot be infected.

  In addition to those groups, a node can also belong to additional groups:

  Tested Positive (TP): Nodes who tested positive for the disease. At each
    simulation step, a certain share of Infected nodes are tested and a certain
    share of all nodes are tested. Nodes which are Exposed or Infected test
    positive. They become Quarantined, and potentially some share of their
    neighbors is traced. This does not model false positives.

  Quarantined (Q): Nodes which are in quarantine. They can neither infect nor
    get infected. Quarantine ends after a set period. Their disease progression
    continues irrespective of Quarantine.

  The simulation keeps a state in the form of indicator arrays, which describe
  for each node which groups they are a member of: S_arr, E_arr, I_arr, R_arr,
  Q_arr, TP_arr. The time left before transition to the next state is kept in
  additional arrays Q_left (time left to exit Quarantine) and E_left (incubation
  time left before becoming Infected).

  All probabilities in args are *daily* probabilities, unless stated otherwise.

  Args:
    N: Number of nodes in the simulation.
    G: Graph on which to run the simulation. If None, generated using N and the
      args of graph attribute below.
    initial_infected_num: Number of individuals infected at t=0.
    prob_infect: In a single interaction on a single day, what is the chance of
     an Infectious to infect a Susceptible.
    incubation_duration_mean: Incubation period duration distribution mean. In
      days.
    incubation_duration_std: Incubation period duration distribution standard
      deviation. In days.
    prob_asymptomatic: Share of population who would not develop symptoms even
      when sick. They are not detected along with the symptomatic Infected in
      the testing step, but are detected by direct tests (mass testing or when
      neighbors are tested).
    prob_recover: Probability to change from Infections to Recovered in a single
      day.
    quarantine_neighbors: When an individual tests positive, whether their
      traced neighbors are quarantined.
    test_neighbors: When an individual tests positive, whether their traced
      neighbors are tested.
    prob_infected_detected: Probability that an Infected individual is tested
      and detected as a carrier in a single day. This encapsulates also the fact
      that many people are asymptomatic or will not be tested (or are not
      quarantined).
    prob_neighbor_traced: Probability that a neighbor of an individual who
      tested positive is himself traced and tested/quarantined. The testing of
      neighbors happens once, once the original individual is tested positive,
      so this is *not* a probability per day.
    prob_exposed_detected: Share of the general population which is tested on a
      single day. Exposed and Infected individuals who are tested are detected
      as carriers.
    test_delay_time: Delay between the time a test is performed and the time
      results return and contact-traced neighbors are tested/quarantined. Days.
    days_in_quarantine: Length of imposed quarantine, in days.
    prob_infect_exposed_factor: Ratio of infectiousness between Exposed (in the
      final days before becoming Infected) and Infected. In other words,
      prob_infect_exposed = prob_infect * prob_infect_exposed_factor.
      Set this to 0 to make the Exposed non-infectious.
    relative_infectiousness_asymptomatic: Ratio of infectiousness between
      Asymptomatic and symptomatic. In other words,
      prob_infect_asymptomatic = prob_infect * relative_infectiousness_asymptomatic.
      The same ratio holds for the final contagious part of the incubation period.
      Set this to 0 to make the Asymptomatic non-infectious.
    duration_exposed_infects: How many days before developing symptoms (becoming
      Infected) an Exposed individual is contagious. Questionable evidence, but
      1-2 produces a reasonable R0.
    steps_per_day: Number of simulation steps per one day of real life time.
    min_degree: Minimal degree of nodes in a generated graph.
    mean_degree: Mean degree of nodes in a generated graph.
    gamma: Parameter of degree distribution in generated graph.
      Higher gamma means a fatter tail of the degree distribution.
    max_steps: Maximal number of steps in simulation. If this number is reached,
      simulation stops.
    verbose: Whether to print some progress indicators.

  Returns:
    SimulationResults object with the results of the simulation. This holds the
    time series of total number of members in each group (Susceptible, Exposed
    etc.) as well as other variables of interest, such as number of tests
    performed.
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

  (S_arr, E_arr, E_left, I_arr, A_arr, R_arr, Q_arr, Q_left, TP_arr,
   T_result_left, T_result_positive_arr) = init_states(
       N, initial_infected_num=initial_infected_num, incubation_duration_mean=incubation_duration_mean,
       incubation_duration_std=incubation_duration_std, prob_recover=prob_recover,
       prob_asymptomatic=prob_asymptomatic)
  n_infected_tested = n_neighbors_traced = n_neighbors_tested = n_general_tested = 0
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
                                   n_infected_tested, n_neighbors_traced, n_neighbors_tested, n_general_tested))

    # Infection step. S -> E
    S_arr, E_arr = infection_step(S_arr, E_arr, E_left, I_arr, A_arr, Q_arr, adj_mat,
                                  incubation_k=incubation_k,
                                  incubation_theta=incubation_theta,
                                  prob_infect=prob_infect_per_step,
                                  prob_infect_exposed_factor=prob_infect_exposed_factor,
                                  relative_infectiousness_asymptomatic=relative_infectiousness_asymptomatic,
                                  duration_exposed_infects=duration_exposed_infects)

    # Incubation step. E -> I
    E_arr, E_left, I_arr = incubation_step(E_arr, E_left, I_arr,
                                           steps_per_day=steps_per_day)

    # Testing step. Update tested_positive and newly quarantined.
    (TP_arr, new_Q_arr, T_result_left, T_result_positive_arr, n_infected_tested,
     n_neighbors_traced, n_neighbors_tested, n_general_tested) = testing_step(
         E_arr, I_arr, A_arr, TP_arr, T_result_left, T_result_positive_arr,
         prob_infected_detected=prob_infected_detected_per_step,
         prob_neighbor_traced=prob_neighbor_traced,
         prob_exposed_detected=prob_exposed_detected_per_step,
         quarantine_neighbors=quarantine_neighbors,
         test_neighbors=test_neighbors,
         test_delay_time=test_delay_time,
         adj_mat=adj_mat,
         steps_per_day=steps_per_day)

    # Quarantine step. Update Q
    Q_arr, Q_left = quarantine_step(
        Q_arr, Q_left, R_arr, TP_arr, new_Q_arr,
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
                                 n_infected_tested, n_neighbors_traced, n_neighbors_tested, n_general_tested))

  # Convert to DataFrame.
  results_df = pd.DataFrame(counters)
  tested = results_df.filter(regex='_tested').sum(axis=1)
  # Test rate: number of tests per day. Smoothe using day-length sliding window.
  results_df['test_rate'] = np.convolve(tested, np.ones((steps_per_day,)), mode='same')
  results_df['step'] = np.arange(len(results_df))
  results_df['day'] = results_df['step'] / steps_per_day
  results = SimulationResults(results_df, G, prob_infect=prob_infect,
                              prob_infected_detected=prob_infected_detected,
                              prob_neighbor_traced=prob_neighbor_traced,
                              prob_exposed_detected=prob_exposed_detected,
                              quarantine_neighbors=quarantine_neighbors,
                              test_neighbors=test_neighbors,
                              steps_per_day=steps_per_day)
  if verbose:
    print('Simulation finished!')
  return results
