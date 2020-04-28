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

"""Specific experiments of the model."""

import glob
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from seir import *

OUTCOME_COLUMNS = ['Share of Population Infected', 'Peak Daily Test Rate', 'Quarantine Days per Person']

def quarantine_matters():
  """Results are different with/without quarantine."""
  # TODO: compare to:
  # - quarantine of similar amount of random individuals.
  # - quarantine of enough random individuals to attain the same results on infection.
  N = 10000
  G = generate_scale_free_graph(N, 2, 5, 0.4)
  for prob_detected_and_quarantined, quarantine_neighbors in [
      (0.05, True),  # Quarantine including neighbors - early containment. Less quarantine too!
      (0.05, False),  # Quarantine only of infected - 0.5 population infected
      (0.01, True),  # Quarantine including neighbors, but less testing - 0.5 infected. And even more quarantine than previous option!
      (0, False)  # No quarantine - 80% population infected.
  ]:
    prob_infect = 0.02
    prob_recover = 0.02
    prob_infect_exposed_factor = 0  # Exposed don't infect. Legacy behavior.
    results = simulation(N, G, prob_detected_and_quarantined=prob_detected_and_quarantined,
                         quarantine_neighbors=quarantine_neighbors, prob_infect=prob_infect,
                         prob_recover=prob_recover, prob_infect_exposed_factor=prob_infect_exposed_factor,
                         max_steps=5000)
    results.summary(hyperparams=True, plot=True)
    print()


def sensitive_to_gamma():
  """Parameter gamma of the network is important for spread of epidemic.

  Changes behavior from quick spread to finite fraction of the population
  to endemic behavior with around 1/250 of the population eventually infected.
  """
  N = 10000
  for gamma in [0.8, 0.4, 0.1, 0.05, 0]:
    G = generate_scale_free_graph(N, 2, 5, gamma)
    results = simulation(N, G, prob_infect=0.005, prob_recover=0.02, prob_detected_and_quarantined=0, quarantine_neighbors=False, max_steps=5000)
    results.summary(hyperparams=False, G_attrs=True, plot=True)
    print()


def average_results(result_lst, G=None, same_hyperparams=False):
  """Averages multiple SimulationResults objects.

  Args:
    result_lst: List of SimulationResults objects.
    same_hyperparams: if True, apply those hyperparams to the result.

  Returns:
    SimulationResults initialized with average of the result DataFrames.
  """
  # TODO: change to long form and wide form with ffill, like im plot_result_list?
  #   If implemented, add Q_arr.sum() == 0 to simulation stop condition, so that we don't ffill some constant nonzero value.
  all_res_df = pd.concat([results.df for results in result_lst])
  averaged_df = all_res_df.groupby(['day', 'step'])[ALL_COLUMNS].mean().reset_index()
  hyperparams = result_lst[0].hyperparams if same_hyperparams else {}
  averaged_results = SimulationResults(averaged_df, G, n_runs=len(result_lst), **hyperparams)
  if same_hyperparams:
    averaged_results.G_attrs = result_lst[0].G_attrs
  # averaged_results.summary()
  return averaged_results


def rerun_experiment(experiment_func, n_runs=20, aggregate=True):
  """Reruns experiment_func n_runs times and aggregates results.

  Args:
    aggregate: Whether to return a SimulationResults object of the average trend,
      or a list of SimulationResults objects.
  """
  result_lst = []
  for i in range(n_runs):
    results = experiment_func()
    results.df['run_idx'] = i
    result_lst.append(results)
  if aggregate:
    avg_results = average_results(result_lst, same_hyperparams=True)
    return avg_results
  else:
    return result_lst


def create_experiment_func(**kwargs):
  """Creates a function that runs a simulation, for rerun_experiment()."""
  def experiment_func():
    results = simulation(**kwargs)
    return results
  return experiment_func


def repeated_sim(n_runs=5, aggregate=True, **kwargs):
  """Runs a simulation with **kwargs, n_runs times. Returns all/aggregate results."""
  sim_func = create_experiment_func(**kwargs)
  avg_results = rerun_experiment(sim_func, n_runs=n_runs, aggregate=aggregate)
  return avg_results


def calibrate_doubling_time(n_runs=20, prob_min_bound=0.01, prob_max_bound=0.3,
                            stopping_ratio=1.1, prob_infected_detected=0, prob_exposed_detected=0,
                            target_doubling_days=DOUBLING_DAYS, **simulation_kwargs):
  """Fits prob_infect to produce the correct target doubling time.

  Uses binary search.
  """
  def get_doubling_days(prob, **kwargs):
    experiment_func = create_experiment_func(
        prob_infect=prob,
        prob_infected_detected=prob_infected_detected, **kwargs)
    avg_results = rerun_experiment(experiment_func, n_runs=n_runs)
    return avg_results.doubling_days

  prob2doubling_days = {}
  for prob in [prob_min_bound, prob_max_bound]:
    doubling_days = get_doubling_days(prob, **simulation_kwargs)
    prob2doubling_days[prob] = doubling_days
    print(f'{prob:.3f}, {doubling_days:.2f}')

  # If bounds are too narrow, expand them.
  while prob2doubling_days[prob_max_bound] > target_doubling_days:
    prob_max_bound = prob_max_bound * 2
    doubling_days = get_doubling_days(prob_max_bound, **simulation_kwargs)
    prob2doubling_days[prob_max_bound] = doubling_days
    print(f'{prob_max_bound:.3f}, {doubling_days:.2f}')

  while prob2doubling_days[prob_min_bound] < target_doubling_days:
    prob_min_bound = prob_min_bound / 2
    doubling_days = get_doubling_days(prob_min_bound, **simulation_kwargs)
    prob2doubling_days[prob_min_bound] = doubling_days
    print(f'{prob_min_bound:.3f}, {doubling_days:.2f}')

  while prob_max_bound / prob_min_bound > stopping_ratio:
    prob_mid = np.sqrt(prob_min_bound * prob_max_bound)
    doubling_days = get_doubling_days(prob_mid, **simulation_kwargs)
    prob2doubling_days[prob_mid] = doubling_days
    # Make sure doubling time is monotonic.
    doubling_days = np.clip(doubling_days,
                            prob2doubling_days[prob_max_bound],
                            prob2doubling_days[prob_min_bound])
    print(f'{prob_mid:.3f}, {doubling_days:.2f}')
    if doubling_days > target_doubling_days:
      # Increase probability of infection
      prob_min_bound = prob_mid
    else:  # doubling_days <= target_doubling_days:
      # Decrease probability of infection
      prob_max_bound = prob_mid

  return np.sqrt(prob_min_bound * prob_max_bound)


def true_cases_outnumber_known_cases(n_runs=5, prob_infected_detected=0, prob_exposed_detected=0):
  """In the early days of the epidemic, true cases far outnumber confirmed cases."""
  simple_sim = create_experiment_func(prob_infected_detected=prob_infected_detected)
  avg_results = rerun_experiment(simple_sim, n_runs=n_runs)
  avg_results.summary(plot=True)
  avg_results.df['confirmed_cases'] = avg_results.df[['infected', 'recovered']].sum(axis=1)
  avg_results.df['true_cases'] = avg_results.df[['exposed', 'infected', 'recovered']].sum(axis=1)
  # Before 5% of population is confirmed cases.
  early_days_end = (avg_results.df['confirmed_cases'] / N < 0.05).idxmin()
  avg_results.df = avg_results.df.iloc[:early_days_end]
  avg_results.plot_trends(columns=['confirmed_cases', 'true_cases'], hyperparams=False)
  plt.figure()
  ax = avg_results.df.set_index('day').eval('true_cases / confirmed_cases').plot()
  ax.set_title('ratio true cases / confirmed cases')


def draw_gamma_distribution(mean=INCUBATION_DURATION_MEAN,
                            std=INCUBATION_DURATION_STD,
                            n_samples=100000):
  """Draw values from Gamma distribution by mean & std."""
  k, theta = get_gamma_distribution_params(mean, std)
  values = np.random.gamma(k, theta, n_samples)
  return values


def estimate_effective_exposed_infectious_time(incubation_duration_mean=INCUBATION_DURATION_MEAN,
                                               incubation_duration_std=INCUBATION_DURATION_STD,
                                               duration_exposed_infects=DURATION_EXPOSED_INFECTS):
  """Mean number of days an exposed individual is infectious. Empirically estimated."""
  incubation_durations = draw_gamma_distribution(incubation_duration_mean,
                                                 incubation_duration_std)
  days_infectious_exposed = np.clip(incubation_durations, 0, duration_exposed_infects).mean()
  return days_infectious_exposed


def ballpark_r0(prob_infect=PROB_INFECT, prob_recover=PROB_RECOVER, incubation_duration_mean=INCUBATION_DURATION_MEAN, incubation_duration_std=INCUBATION_DURATION_STD, duration_exposed_infects=DURATION_EXPOSED_INFECTS, prob_infect_exposed_factor=PROB_INFECT_EXPOSED_FACTOR, mean_degree=MEAN_DEGREE, prob_infected_detected=0, prob_exposed_detected=0, **kwargs):
  """Naive back of the envelope calculation of R0 for Exposed, Infected, and total.

  Default is without quarantine. But can be adjusted using argument
  prob_infected_detected (default=0).
  """
  if prob_exposed_detected != 0:
    raise NotImplementedError(f"Currently can't handle prob_exposed_detected != 0. Received {prob_exposed_detected}")
  # Exposed stage.
  days_infectious_exposed = estimate_effective_exposed_infectious_time(
      incubation_duration_mean, incubation_duration_std, duration_exposed_infects)  # 3.58
  prob_infect_exposed = prob_infect * prob_infect_exposed_factor  # 0.02 (per day per neighbor)
  n_susceptible_neighbors = mean_degree - 1  # That node was already infected from a neighbor. So 7 susceptible neighbors.
  exposed_r0 = days_infectious_exposed * prob_infect_exposed * n_susceptible_neighbors  # 0.50

  # Infected stage.
  prob_end_infected = 1 - (1 - prob_recover) * (1 - min(prob_infected_detected, 1))
  days_infectious = 1 / prob_end_infected  # 3.5 days on average.
  # This term introduces slight non-linearity in prob_infect.
  n_susceptible_neighbors = mean_degree - 1 - exposed_r0  # 7 susceptible neighbors as before, but 0.5 was infected at the exposed stage.
  infected_r0 = days_infectious * prob_infect * n_susceptible_neighbors  # 0.91

  # This is almost linear in prob_infect and inversely linear in all the durations.
  # To make it work out to R0=2.4, we need all the durations to be about 2.4/1.4=1.7 times longer.
  total_r0 = exposed_r0 + infected_r0  # 1.41 - too low.
  return exposed_r0, infected_r0, total_r0


def how_doublings_scale(factor=2, n_runs=5):
  """Investigate the scaling of doubling time."""
  G = generate_scale_free_graph(N, 2, 8, 0.4)
  G_mean_degree = G.number_of_edges() * 2 / G.number_of_nodes()
  print('Base R0:', ballpark_r0())
  print('factor:', factor)

  # All durations are scaled by 'factor', prob_infect divided by 'factor'.
  # Everything is 'factor' times slower, so doubling time is multiplied by 'factor'.
  kwargs = dict(G=G, prob_infect=PROB_INFECT/factor, incubation_duration_mean=INCUBATION_DURATION_MEAN*factor, incubation_duration_std=INCUBATION_DURATION_STD*factor, prob_recover=PROB_RECOVER/factor, duration_exposed_infects=DURATION_EXPOSED_INFECTS*factor, prob_infected_detected=0, prob_exposed_detected=0)
  func = create_experiment_func(**kwargs)
  results = rerun_experiment(func, n_runs)
  print('All durations stretched by factor, prob_infect divided by factor.')
  print('Doubling days:', results.doubling_days)
  print('R0:', ballpark_r0(**kwargs, mean_degree=G_mean_degree))

  # prob_infect is not scaled. Doubling days are the same ~3 as originally.
  kwargs['prob_infect'] = PROB_INFECT
  func = create_experiment_func(**kwargs)
  results = rerun_experiment(func, n_runs)
  print('All durations stretched by factor, prob_infect unchanged.')
  print('Doubling days:', results.doubling_days)
  print('R0:', ballpark_r0(**kwargs, mean_degree=G_mean_degree))

  # Try more connections, but scale back prob_infect to offset it.
  # Doubling days stay the same, ~3.
  H = generate_scale_free_graph(N, 2, 20, 0.4)
  H_mean_degree = H.number_of_edges() * 2 / H.number_of_nodes()
  kwargs.update({'G': H, 'prob_infect': PROB_INFECT*(G_mean_degree-1)/(H_mean_degree-1)})
  func = create_experiment_func(**kwargs)
  results = rerun_experiment(func, n_runs)
  print('Graph with larger mean degree. prob_infect decreased to maintain R0.')
  print('Doubling days:', results.doubling_days)
  print('R0:', ballpark_r0(**kwargs, mean_degree=H_mean_degree))
  # results.summary(True, plot=True)

  # Without scaling prob_infect, doubling time sort of inversely proportional to mean degree - 1.
  H = generate_scale_free_graph(N, 2, 4, 0.4)
  H_mean_degree = H.number_of_edges() * 2 / H.number_of_nodes()
  kwargs.update({'G': H, 'prob_infect': PROB_INFECT})
  func = create_experiment_func(**kwargs)
  results = rerun_experiment(func, n_runs)
  print('Graph with larger mean degree. prob_infect stays the same.')
  print('Doubling days:', results.doubling_days)
  print('R0:', ballpark_r0(**kwargs, mean_degree=H_mean_degree))


def draw_infection_times(prob_infect_exposed_factor=PROB_INFECT_EXPOSED_FACTOR,
                         duration_exposed_infects=DURATION_EXPOSED_INFECTS,
                         incubation_duration_mean=INCUBATION_DURATION_MEAN,
                         incubation_duration_std=INCUBATION_DURATION_STD,
                         prob_recover=PROB_RECOVER,
                         prob_infect=PROB_INFECT):
  """Draw times of random infections (relative to first exposure)."""
  n_samples = 100000
  incubation_durations = draw_gamma_distribution(incubation_duration_mean,
                                                 incubation_duration_std,
                                                 n_samples)
  days_infectious_exposed = np.clip(incubation_durations, 0, duration_exposed_infects)
  infected_durations = np.random.exponential(scale=1/prob_recover, size=n_samples)
  prob_infect_exposed = prob_infect * prob_infect_exposed_factor
  # Stages of infectiousness
  # ------------------------
  # A) incubation_duration - days_infectious_exposed: 0 infectiousness
  # B) days_infectious_exposed: prob_infect * prob_infect_exposed_factor infectiousness
  # C) infected_duration: prob_infect infectiousness
  numeric_factor = 10  # Adjusts so that we don't always sample 0 or 1 infected.
  n_infections_exposed_float = prob_infect_exposed * days_infectious_exposed * numeric_factor
  n_infections_exposed_int = n_infections_exposed_float.astype(int) + (np.random.random(n_samples) < (n_infections_exposed_float % 1))
  n_infections_infected_float = prob_infect * infected_durations * numeric_factor
  n_infections_infected_int = n_infections_infected_float.astype(int) + (np.random.random(n_samples) < (n_infections_infected_float % 1))
  all_infection_times_exposed = np.concatenate([
      np.random.uniform(incubation_durations[i] - days_infectious_exposed[i],
                        incubation_durations[i],
                        n_infections_exposed_int[i]) for i in range(n_samples)])
  all_infection_times_infected = np.concatenate([
      np.random.uniform(incubation_durations[i],
                        incubation_durations[i] + infected_durations[i],
                        n_infections_infected_int[i]) for i in range(n_samples)])

  all_infection_times = np.concatenate([all_infection_times_exposed, all_infection_times_infected])
  # sns.distplot(all_infection_times)  # Not the same as Gamma distribution.
  return all_infection_times


def compare_serial_intervals(**kwargs):
  all_infection_times = draw_infection_times(**kwargs)
  m = all_infection_times.mean()
  s = all_infection_times.std()  # 6.5, 4.03 according to Imperial #3
  md = np.median(all_infection_times)
  print(f'mean: {m}, std: {s}, coef of variation: {s/m}, median = {md}')
  sns.distplot(all_infection_times)
  gamma_values = draw_gamma_distribution(m, s)
  sns.distplot(gamma_values)  # Not exactly the same.


def check_graph_params(min_degree=MIN_DEGREE, mean_degree=MEAN_DEGREE, gamma=GAMMA, N=N):
  degs = generate_power_law_degrees(N, min_degree, mean_degree, gamma)
  print(np.median(degs), np.sort(degs)[-int(0.001 * N)])  # (median, top 0.1%)


def show_r0_not_accurate():
  """Exponential growth despite R0 < 1."""
  G = generate_scale_free_graph(N, 2, 20, 0.2)
  G_mean_degree = G.number_of_edges() * 2 / G.number_of_nodes()
  prob_infect = 0.024  # calibrated for 3-day doubling w/o quarantine.
  prob_detected_and_quarantined = 0.8
  r0 = ballpark_r0(prob_infect, mean_degree=G_mean_degree, prob_detected_and_quarantined=prob_detected_and_quarantined)
  print('R0:', r0)  # Total < 1
  results = repeated_sim(prob_infect=prob_infect, G=G, prob_detected_and_quarantined=prob_detected_and_quarantined, n_runs=3)
  results.summary(plot=True)  # Exponential growth.

  prob_infect = 0.01
  prob_detected_and_quarantined = 0.0
  r0 = ballpark_r0(prob_infect, mean_degree=G_mean_degree, prob_detected_and_quarantined=prob_detected_and_quarantined)
  results = repeated_sim(prob_infect=prob_infect, G=G, prob_detected_and_quarantined=prob_detected_and_quarantined, n_runs=3)
  print('R0:', r0)
  results.summary(plot=True)


def plot_ratio_exposed_infected(results):
  print(results.df.set_index('day').eval('exposed / infected').loc[results.peak_time / 4:results.peak_time * 3 / 4].mean())
  return results.df.set_index('day').eval('exposed / infected').plot()


def plot_result_list(result_lst, columns=None):
  """Plots a list of SimulationResults on the same axes.

  Args:
    result_lst: As returned from rerun_experiment(..., aggregate=False).
      Assumed each SimulationResults attribute df has a column 'run_idx'.
    columns: The DataFrame columns to plot.
  """
  if columns is None:
    columns = MAIN_GROUPS
  df_long = pd.concat([result.df.drop(columns='step').set_index(['day', 'run_idx']) / result.N for result in result_lst]).reset_index()
  avg_result = average_results(result_lst, same_hyperparams=True)
  df_wide = df_long.pivot_table(index='day', columns='run_idx').ffill()
  avg_df_normed = avg_result.df.set_index('day') / avg_result.N
  alpha = 0.2
  fig, ax_arr = plt.subplots(1, 2)
  title = str({k: round(v, 3) for k, v in avg_result.hyperparams.items()})
  fig.suptitle(title, fontsize=18)
  for group in columns:
    for pane_ind, logy in enumerate([False, True]):
      color = GROUP2COLOR.get(group, 'grey')
      ax = df_wide[group].plot(c=color, alpha=alpha, logy=logy, legend=False, ax=ax_arr[pane_ind])
      ax = avg_df_normed[group].plot(c=color, linewidth=2, logy=logy, ax=ax)
      ax.set_ylabel('Fraction of population')
      if logy:
        ax.set_ylim(1 / avg_result.N)  # Below 1/N there's a lot of noise.


def clustering_coefficient_matters(mean_degree=MEAN_DEGREE):
  """Degree distribution and clustering coefficients change R0 for constant doubling."""
  G = generate_scale_free_graph(N, mean_degree=mean_degree, gamma=0.2)  # R0=2.2
  G0 = generate_scale_free_graph(N, mean_degree=mean_degree, gamma=0)  # Constant degree. R0=3.8
  H0 = generate_local_graph(N, n_divisions=4, mean_degree=mean_degree) # Constant degree, clustering 0.13. R0=4.7
  H1 = generate_local_graph(N, n_divisions=2, mean_degree=mean_degree) # Constant degree, clustering 0.32. R0=5.07

  kwargs = dict(n_runs=3, stopping_ratio=1.1, prob_min_bound=0.01, prob_max_bound=0.1)
  for graph in [G, G0, H0, H1]:
    print(graph.name)
    print('Clustering coef:', nx.average_clustering(nx.Graph(graph)))
    found_prob_infect = calibrate_doubling_time(G=graph, **kwargs)
    print('prob_infect:', found_prob_infect)
    r0 = ballpark_r0(prob_infect=found_prob_infect, mean_degree=mean_degree)
    print('R0: ', r0)


def get_graph_degrees(G):
  degs = G.degree()
  return list(dict(degs).values())


def scale_free_divisions_stats():
  for n_divisions in [10, 4, 2, 1]:
    print(n_divisions)
    G = generate_local_scale_free_graph(N, n_divisions, 2, 20, 0.2)
    print(nx.average_clustering(nx.Graph(G)))
    degs = G.degree()
    ldegs = list(dict(degs).values())
    print(np.mean(ldegs), np.median(ldegs))
    print(sorted(ldegs)[-10:])
    # plt.figure()
    sns.distplot(ldegs, hist=False)


def read_result(path):
  """Returns params, result saved in given path."""
  params, result = pickle.load(open(path, 'rb'))
  return params, result


def collect_results(paths):
  """Returns results from paths in a summary DataFrame.

  Args:
    paths: List of strings (full paths) or a single string - input to Glob.
  """
  if isinstance(paths, str):
    paths = sorted(glob.glob(paths))

  res_list = []
  for path in paths:
    params, result = read_result(path)
    # Reanalyze, if there were changes.
    result.analyze_results_df()
    params['result'] = result
    res_list.append(params)

  summary_table = pd.DataFrame(res_list)
  for attr in SUMMARY_ATTRS:
    summary_table[attr] = summary_table['result'].apply(lambda x: getattr(x, attr))
  return summary_table


def graph_parameters_sensitivity(all_paths):
  summary_table = collect_results(all_paths)
  summary_table['doublings_per_day'] = 1 / summary_table['doubling_days']
  summary_table['growth_rate'] = 2**summary_table['doublings_per_day']
  summary_table[['mean_degree', 'prob_infect', 'gamma','doubling_days', 'fraction_infected']].sort_values(['mean_degree', 'prob_infect', 'gamma'])
  summary_table[['mean_degree', 'prob_infect', 'gamma','doubling_days', 'fraction_infected']].sort_values(['mean_degree', 'prob_infect', 'gamma']).query('prob_infect==0.01')
  fig, ax = plt.subplots()
  # summary_table.query('prob_infect==0.01').plot.scatter(x='mean_degree', y='gamma', s=500, c='doubling_days', colormap='autumn', ax=ax)
  summary_table.query('prob_infect==0.01').plot.scatter(x='mean_degree', y='gamma', s=500, c='growth_rate', colormap='Reds', ax=ax)

  # Mean degree & prob_infect matter in the predictable way - through their product.
  # summary_table.query('gamma==0.2 and prob_infect < 0.4').pivot_table(index='mean_degree', columns='prob_infect', values='growth_rate').plot(loglog=True)
  summary_table['R0'] = summary_table.apply(lambda rec: ballpark_r0(prob_infect=rec['prob_infect'], mean_degree=rec['mean_degree'])[-1], axis=1)
  summary_table['expected_infected_neighbors'] = summary_table['prob_infect'] * summary_table['mean_degree']
  summary_table.query('gamma==0.2 and prob_infect < 0.4').plot.scatter(x='R0', y='growth_rate', loglog=True)

  # In one plot for many gamma values.
  summary_table['gamma (log)'] = np.log(summary_table['gamma'])
  fig, ax = plt.subplots()
  summary_table.query('R0 < 20 and growth_rate>1 and gamma<0.5').plot.scatter(x='R0', y='growth_rate', s=100, c='gamma (log)', alpha=0.5, colormap='cool', edgecolor='black', loglog=True, ax=ax)
  ax.set_ylabel('Daily Growth Rate')
  # summary_table.plot.scatter(x='R0', y='growth_rate', s=100, c='gamma', colormap='cool', loglog=True, ax=ax)

  def calc_moments(N, gamma=0.2, mean_degree=20, min_degree=2):
    degs = generate_power_law_degrees(int(N), min_degree, mean_degree, gamma)
    return degs.mean(), (degs**2).mean()

  def moment_ratio(N, gamma=0.2, mean_degree=20, min_degree=2):
    moments = calc_moments(N, gamma, mean_degree, min_degree)
    return moments[1] / moments[0]

  graph_attrs = summary_table[['N', 'gamma', 'min_degree', 'mean_degree']].drop_duplicates().copy()
  graph_attrs['moment_ratio'] = graph_attrs.apply(lambda rec: moment_ratio(rec['N'], rec['gamma'], rec['mean_degree'], rec['min_degree']), axis=1)

  summary_table = summary_table.merge(graph_attrs, on=['N', 'gamma', 'min_degree', 'mean_degree'])
  summary_table['infect_x_moment_ratio'] = summary_table.eval('prob_infect * moment_ratio')

  fig, ax = plt.subplots()
  summary_table.query('R0 < 20 and gamma<0.5 and growth_rate>1').plot.scatter(x='infect_x_moment_ratio', y='growth_rate', s=100, c='gamma (log)', alpha=0.5, colormap='cool', edgecolor='black', loglog=True, ax=ax)
  ax.set_xlabel('mu2 / mu1 * infection probability')
  ax.set_ylabel('Daily Growth Rate')
  # summary_table.query('R0 < 20').plot.scatter(x='infect_x_moment_ratio', y='growth_rate', s=100, c='gamma (log)', alpha=0.5, colormap='cool', edgecolor='black', loglog=True, ax=ax)
  # summary_table.query('infect_x_moment_ratio < 10 and gamma<0.3 and growth_rate>1').plot.scatter(x='infect_x_moment_ratio', y='growth_rate', s=100, c='gamma (log)', alpha=0.5, colormap='cool', edgecolor='black', loglog=True, ax=ax)
  # summary_table.query('infect_x_moment_ratio < 100 and gamma<0.5 and growth_rate>1').plot.scatter(x='infect_x_moment_ratio', y='growth_rate', s=100, c='gamma (log)', alpha=0.5, colormap='cool', edgecolor='black', loglog=True, ax=ax)

  return summary_table


def calibration_analysis():
  summary_table = collect_results('calibrate_doubling*')
  summary_table[['initial_infected_num', 'prob_infect', 'doubling_days']].sort_values(['initial_infected_num', 'prob_infect'])
  # Initial infected num = 10 show lower doubling times by a 0.15.
  tab = summary_table.pivot_table(index='prob_infect', columns='initial_infected_num', values='doubling_days')
  (tab[100] - tab[10]).describe()
  tab.plot()
  fig, ax = plt.subplots(figsize=(15,10))
  summary_table.query('initial_infected_num==100').set_index('prob_infect').sort_index()['doubling_days'].plot(ax=ax)
  ax.set_ylabel('Doubling Time (Days)')
  ax.set_xlabel('Probability of Infection')
  return summary_table


def my_relplot(df, x, y, s=None, c=None, marker=None, c_categorical=False, ax=None, **kwargs):
  """Plotting function to mimic some of the functionality of sns.relplot."""
  if (c is not None) and (c in df.columns):
    if c_categorical or not pd.api.types.is_numeric_dtype(df[c]):
      # Categorical color column
      df = df.copy()
      df[c + ' '] = df[c].rank()
      c = c + ' '

  MARKERS = 'o^s*XD2<>H|_'
  if ax is None:
    fig, ax = plt.subplots()
  if (marker is None) or (marker not in df.columns):
    ax = df.plot.scatter(x=x, y=y, s=s, c=c, marker=marker, ax=ax, **kwargs)
  else:
    # marker is a column in df
    gr = df.groupby(marker)
    colorbar = kwargs.pop('colorbar', True)
    for i, (_, df_filtered) in enumerate(gr):
      marker = MARKERS[i]
      ax = df_filtered.plot.scatter(x=x, y=y, s=s, c=c, marker=marker, ax=ax, colorbar=colorbar, **kwargs)
      colorbar = False  # Only draw one
  return ax


def policy_contour_lines_plot(table_no_tracing):
  # My very cool incomprehensible graph
  fig, ax = plt.subplots()
  for quarantine_neighbors in [True, False]:
    style = ['-', '--'][int(quarantine_neighbors)]
    tab = table_no_tracing.query('quarantine_neighbors == @quarantine_neighbors')
    tab.query('prob_exposed_detected==0').sort_values('prob_infected_detected').plot(x='Share of Population Infected', y='Peak Daily Test Rate', style=style, label='test infected', logx=True, ax=ax)
    tab.query('prob_infected_detected==0').sort_values('prob_exposed_detected').plot(x='Share of Population Infected', y='Peak Daily Test Rate', style=style, label='test general population', logx=True, ax=ax)
  ax.set_ylabel('Fraction eventually infected')


def policy_contour_lines_plot2(table_no_genpop):
  days_to_detect_infected_options = sorted(table_no_genpop['Days to Detect Infected'].unique())
  table_no_genpop.pivot_table(values='peak_test_rate', index='fraction_infected', columns='prob_neighbor_detected')
  fig, ax = plt.subplots()
  for days_to_detect_infected in days_to_detect_infected_options:
    tab = table_no_genpop[table_no_genpop['Days to Detect Infected'] == days_to_detect_infected].sort_values('fraction_infected')
    tab.plot(x='Share of Population Infected', y='peak_test_rate', label=days_to_detect_infected, marker='*', loglog=True, ax=ax)
  ax.set_ylabel('Peak Daily Test Rate')
  ax.legend(title='Days to Detect')

  # prob_neighbor_detected_options = sorted(table_no_genpop['prob_neighbor_detected'].unique())
  # fig, ax = plt.subplots()
  # for prob_neighbor_detected in prob_neighbor_detected_options:
  #   tab = table_no_genpop[table_no_genpop['prob_neighbor_detected'] == prob_neighbor_detected].sort_values('fraction_infected')
  #   tab.plot(x='fraction_infected', y='peak_test_rate', label=prob_neighbor_detected, marker='*', loglog=True, ax=ax)
  return ax

def outcome_bar_chart(tab_indexed, title='', logy=True, plot_columns=OUTCOME_COLUMNS, ylim=(0.0003, 10), **kwargs):
  ax = tab_indexed[plot_columns].plot.bar(logy=logy, title=title, **kwargs)
  ax.set_ylim(ylim)
  return ax


def test_symptomatic_vs_genpop_barcharts(table_filtered, quarantine_neighbors, logy=True, days_to_detect_infected=(0.2, 0.5, 1.0, 2.0, 10.0, 20.0)):
  table_no_tracing = table_filtered.query('prob_neighbor_detected==0').copy()
  tab = table_no_tracing.query('quarantine_neighbors == @quarantine_neighbors')

  # tab.query('prob_exposed_detected==0').sort_values('prob_infected_detected')[['prob_infected_detected'] + plot_columns + ['doubling_days']]
  # tab.query('prob_infected_detected==0').sort_values('prob_exposed_detected')[['prob_infected_detected', 'prob_exposed_detected'] + plot_columns + ['doubling_days']]

  title = 'Test Symptomatic, Quarantine Positives'
  if quarantine_neighbors:
    title += ' & Contacts'
  ax = outcome_bar_chart(tab.query('prob_exposed_detected==0').set_index('prob_infected_detected').sort_index(), logy=logy, title=title)
  ax.set_xlabel('Fraction detected per day')
  ax = outcome_bar_chart(tab.query('prob_exposed_detected==0').set_index('Days to Detect Infected').loc[days_to_detect_infected].sort_index(), logy=logy, title=title)

  title = 'Test General Population, Quarantine Positives'
  if quarantine_neighbors:
    title += ' & Contacts'
  ax = outcome_bar_chart(tab.query('prob_infected_detected==0').set_index('prob_exposed_detected').sort_index(ascending=False), logy=logy, title=title)
  ax.set_xlabel('Fraction tested per day')


def testing_quarantine_scenarios_analysis():
  """Produce charts comparing testing of symptomatic, tracing, and general population."""
  summary_table = collect_results('quarantine*')
  nu = summary_table.nunique()
  varying_cols = nu[(nu > 1) & (nu < 20)].index.tolist()
  summary_table.drop_duplicates(subset=varying_cols, keep='last', inplace=True)

  summary_table['Days to Detect Infected'] = np.round(1 / summary_table['prob_infected_detected'], 2)
  summary_table['Share of Population Infected'] = summary_table['fraction_infected']
  summary_table['Peak Daily Test Rate'] = summary_table['peak_test_rate']
  summary_table['Quarantine Days per Person'] = summary_table['fraction_quarantine_time']
  summary_table['Share of Contacts Tested'] = summary_table['prob_neighbor_detected']

  # Doubling in 3.1 days or ~5 days.
  for prob_infect in [0.022, 0.014]:
    table_filtered = summary_table.query('initial_infected_num==100 and prob_infect==@prob_infect').copy()
    table_no_tracing = table_filtered.query('prob_neighbor_detected==0')

    policy_contour_lines_plot(table_no_tracing)

    # Bar charts comparing results of testing the symptomatic vs. testing the general population.
    # Show that quarantining only the symptomatic, but immediately (prob=5) gives containment, though imperfect.
    test_symptomatic_vs_genpop_barcharts(table_filtered, quarantine_neighbors=False, days_to_detect_infected=[0.2, 0.5, 1.0, 2.0, 10.0])
    # With quarantining also neighbors, more leeway.
    test_symptomatic_vs_genpop_barcharts(table_filtered, quarantine_neighbors=True, days_to_detect_infected=[0.2, 1.0, 2.0, 10.0, 20.0])

    # Policy scatters: testing the infected vs. testing the general population.
    table_no_zeros = table_no_tracing.query('quarantine_neighbors and prob_infected_detected > 0 and prob_exposed_detected > 0')
    # These are the different testing policies.
    my_relplot(table_no_zeros, x='Days To Detect Infected', y='prob_exposed_detected', c='prob_exposed_detected', colormap='tab10', s=200, marker='Days To Detect Infected', c_categorical=True, colorbar=True, alpha=0.5, edgecolor='black', loglog=True)
    # And these are their accomplishments in the containment - testing plane (total tests & peak test rate).
    my_relplot(table_no_zeros, x='fraction_infected', y='peak_test_rate', c='prob_exposed_detected', colormap='tab10', s=200, marker='Days To Detect Infected', c_categorical=True, alpha=0.5, edgecolor='black', loglog=True)
    # my_relplot(table_no_zeros, x='fraction_infected', y='fraction_tests', c='prob_exposed_detected', colormap='tab10', s=200, marker='Days To Detect Infected', c_categorical=True, alpha=0.5, edgecolor='black', loglog=True)
    # And the containment-quarantine plane.
    my_relplot(table_no_zeros, x='fraction_infected', y='fraction_quarantine_time', c='prob_exposed_detected', colormap='tab10', s=200, marker='Days To Detect Infected', c_categorical=True, alpha=0.5, edgecolor='black', loglog=True)

    # Test & trace policy scatters.
    table_no_genpop = table_filtered.query('not quarantine_neighbors and prob_exposed_detected==0 and prob_infected_detected>0 and prob_neighbor_detected>0').copy()
    my_relplot(table_no_genpop, x='Days To Detect Infected', y='prob_neighbor_detected', c='prob_neighbor_detected', colormap='Set1', s=2000, marker='Days To Detect Infected', c_categorical=True, colorbar=True, alpha=0.5, edgecolor='black', loglog=True, title=f'Infection Probability = {prob_infect}')
    my_relplot(table_no_genpop, x='fraction_infected', y='peak_test_rate', c='prob_neighbor_detected', colormap='Set1', s=2000, marker='Days To Detect Infected', c_categorical=True, alpha=0.5, edgecolor='black', loglog=True, title=f'Infection Probability = {prob_infect}')
    # my_relplot(table_no_genpop, x='fraction_infected', y='fraction_tests', c='prob_neighbor_detected', colormap='Set1', s=2000, marker='Days To Detect Infected', c_categorical=True, alpha=0.5, edgecolor='black', loglog=True, title=f'Infection Probability = {prob_infect}')
    my_relplot(table_no_genpop, x='fraction_infected', y='fraction_quarantine_time', c='prob_neighbor_detected', colormap='Set1', s=2000, marker='Days To Detect Infected', c_categorical=True, alpha=0.5, edgecolor='black', loglog=True, title=f'Infection Probability = {prob_infect}')

    # Policy countour lines
    ax = policy_contour_lines_plot2(table_no_genpop)
    ax.set_title(f'Policy contour lines, infection probability={prob_infect}')
    # Bar chart for specific prob_infected_detected.
    outcome_bar_chart(table_no_genpop.query('prob_infected_detected==0.5').set_index('Share of Contacts Tested').sort_index(ascending=False), title=f'Contact tracing, infection probability={prob_infect}')
    outcome_bar_chart(table_no_genpop.query('prob_infected_detected==0.5').set_index('Share of Contacts Tested').sort_index(ascending=False), title='Contact tracing')

    # Mass testing with contact tracing.
    table_no_test_infected = table_filtered.query('prob_infected_detected==0')
    ax = outcome_bar_chart(table_no_test_infected.query('not quarantine_neighbors and prob_infected_detected==0 and prob_neighbor_detected==1').set_index('prob_exposed_detected').sort_index(ascending=False).sort_index(ascending=False), logy=True, title='Mass testing with perfect contact tracing')
    ax.set_xlabel('Fraction tested per day')


  return summary_table
