import os
import pandas as pd
from deflex import Scenario
import re

def get_var_costs(es):
    """
    Carbon emissions
        One aggregated value	Mio tonnes	% change
    """
    r = es.results['Main']
    p = es.results['Param']

    var_cost_df = pd.DataFrame()

    for i in r.keys():
        if (i[0].label.cat == 'source') & (i[0].label.tag == 'commodity'):
            var_cost_df.loc[i[0].label.subtag, 'var_cost [Eur/MWh]'] = (
                p[i]['scalars']['variable_costs'])
            var_cost_df.loc[i[0].label.subtag, 'summed_flow [MWh]'] = (
                r[i]['sequences']['flow'].sum())

    var_cost_df['summed_variable_costs [Eur]'] = (var_cost_df['var_cost [Eur/MWh]'] *
                                            var_cost_df['summed_flow [MWh]'])

    var_cost_df.sort_index(inplace=True)
    return var_cost_df


def get_installed_capacity(es):
    """
    Installed capacity
        Differentiated by technologies on next sheet	GW	absolute difference or % change
    """
    p = es.results['Param']
    r = es.results['Main']

    installed_chp = (i for i in p.keys()
                     if (i[1] is not None)
                     if (i[1].label.tag == 'chp'))
    installed_hp = (i for i in p.keys()
                    if (i[1] is not None)
                    if (i[0].label.tag == 'hp'))
    installed_pp = (i for i in p.keys()
                    if (i[1] is not None)
                    if (i[0].label.tag == 'pp'))
    installed_decentral_heat = (i for i in p.keys()
                                if (i[0].label.cat == 'trsf')
                                & (i[0].label.region == 'DE')
                                & (i[1] is not None))
    installed_source_ee = (i for i in p.keys()
                           if ((i[0].label.cat, i[0].label.tag) == ('source', 'ee'))
                           & (i[1] is not None))
    installed_line = (i for i in p.keys()
                      if (i[0].label.cat == 'line')
                      & (i[1] is not None))
    installed_phes = (i for i in p.keys()
                      if (i[0].label.subtag == 'phes'))
    installed_capacity = dict()

    for i in installed_chp:
        conversion_factors = p[(i[1], None)]['scalars']
        regex = re.compile(r'conversion_factors_bus_electricity_all')
        key_el = list(filter(regex.search, conversion_factors.keys()))[0]
        regex = re.compile(r'conversion_factors_bus_heat_district')
        key_heat = list(filter(regex.search, conversion_factors.keys()))[0]

        conversion_factor_el = conversion_factors[key_el]
        conversion_factor_heat = conversion_factors[key_heat]
        nominal_value_fuel = p[(i[0], i[1])]['scalars']['nominal_value']

        installed_capacity['electricity', i[1].label.tag, i[0].label.subtag, i[0].label.region, 'nominal_value'] = \
            conversion_factor_el * nominal_value_fuel
        installed_capacity['heat', i[1].label.tag, i[0].label.subtag, i[0].label.region, 'nominal_value'] = \
            conversion_factor_heat * nominal_value_fuel

    for i in installed_source_ee:
        installed_capacity['electricity', 'ee', i[0].label.subtag, i[0].label.region, 'nominal_value'] = \
            p[i]['scalars']['nominal_value']

    for i in installed_pp:
        installed_capacity['electricity', i[0].label.tag, i[0].label.subtag, i[0].label.region, 'nominal_value'] = \
            p[i]['scalars']['nominal_value']

    for i in installed_phes:
        if i[1] == None:
            installed_capacity['electricity', 'storage', i[0].label.subtag, i[0].label.region,  'nominal_capacity'] = \
                p[i]['scalars']['nominal_capacity']
        else:
            installed_capacity['electricity', 'storage', i[0].label.subtag, i[0].label.region,  'nominal_value'] = \
                p[i]['scalars']['nominal_value']

    for i in installed_line:
        installed_capacity['electricity', 'line', i[0].label.subtag, i[0].label.region,  'nominal_value'] = \
            p[i]['scalars']['nominal_value']

    for i in installed_hp:
        installed_capacity['heat', i[0].label.tag, i[0].label.subtag, i[0].label.region, 'nominal_value'] = \
            p[i]['scalars']['nominal_value']

    for i in installed_decentral_heat:
        installed_capacity['heat', 'decentral', i[0].label.subtag, i[0].label.region, 'nominal_value'] = \
            r[i]['sequences']['flow'].max()

    installed_capacity = pd.Series(installed_capacity)
    installed_capacity = installed_capacity.sort_index()
    installed_capacity = installed_capacity.sum(level=[0,1,2,4])
    installed_capacity.name = 'installed_capacity'
    return installed_capacity

def get_cap_costs(es):
    """
    Get capital cost for existing and invested capacity.
    """
    installed_capacity = get_installed_capacity(es).unstack()
    inv_cost = pd.read_csv('/home/jann/reegis/scenarios/deflex/2012/investment_costs.csv', header=0, index_col=[0,1,2])
    inv_cost = inv_cost.drop(('electricity','ee','wind'))
    cap_costs = inv_cost.append(installed_capacity, 0)
    cap_costs['capex'] = cap_costs['investment_cost'].multiply(cap_costs['nominal_value'])
    # print(cap_costs)
    return None


def get_lcoe(es):
    """
    TODO
    Cost
        Total system cost	eur	% change
        Average total system cost (LCOE)	eur/MWh	% change
        Capacity cost	eur	% change
        Variable cost	eur	% change
        Network cost	eur	% change
    """
    cap_cost_el = get_cap_costs(es) # This has to be split among heat and el
    cap_cost_el = get_var_costs(es) # This has to be split among heat and el
    nw_cost = 0
    consumed_electricity = 2
    lcoe = (cap_cost_el + cap_cost_el) * 1/consumed_electricity
    return lcoe


def fetch_cost_emission(es, with_chp=True):
    """
    Fetch the costs and emissions per electricity unit for all power plants
    and combined heat and power plants if with chp is True

    Returns
    -------
    pd.DataFrame
    """
    idx = pd.MultiIndex(levels=[[], [], []], labels=[[], [], []])
    parameter = pd.DataFrame(index=idx)
    p = es.results['Param']
    flows = [x for x in p if x[1] is not None]
    cs = [x for x in flows if (x[0].label.tag == 'commodity') and
          (x[0].label.cat == 'source')]

    # Loop over commodity sources
    for k in cs:
        fuel = k[0].label.subtag
        emission = p[k]['scalars'].emission
        var_costs = p[k]['scalars'].variable_costs
        # print(var_costs)

        # All power plants with the commodity source (cs)
        pps = [x[1] for x in flows if x[0] == k[1] and x[1].label.tag == 'pp']
        for pp in pps:
            region = pp.label.region
            key = 'conversion_factors_bus_electricity_all_{0}'.format(region)
            cf = p[pp, None]['scalars'][key]
            region = pp.label.region
            parameter.loc[(fuel, region, 'pp'), 'emission'] = emission / cf
            parameter.loc[(fuel, region, 'pp'), 'var_costs'] = var_costs / cf

        # All chp plants with the commodity source (cs)
        if with_chp is True:
            chps = [
                x[1] for x in flows if
                x[0] == k[1] and x[1].label.tag == 'chp']
            hps = [x[1] for x in flows if
                   x[0] == k[1] and x[1].label.tag == 'hp']
        else:
            chps = []
            hps = []

        for chp in chps:
            region = chp.label.region
            eta = {}
            hp = [x for x in hps if x.label.region == region]
            if len(hp) == 1:
                key = 'conversion_factors_bus_heat_district_{0}'.format(region)
                eta['heat_hp'] = p[hp[0], None]['scalars'][key]
                for o in chp.outputs:
                    key = 'conversion_factors_{0}'.format(o)
                    eta_key = o.label.tag
                    eta_val = p[chp, None]['scalars'][key]
                    eta[eta_key] = eta_val
                alternative_resource_usage = (
                        eta['heat'] / (eta['electricity'] * eta['heat_hp']))
                chp_resource_usage = 1 / eta['electricity']

                cf = 1 / (chp_resource_usage - alternative_resource_usage)
                parameter.loc[
                    (fuel, region, 'chp'), 'emission'] = emission / cf
                parameter.loc[
                    (fuel, region, 'chp'), 'var_costs'] = var_costs / cf

                # eta['heat_ref'] = 0.9
                # eta['elec_ref'] = 0.55
                #
                # pee = (1 / (eta['heat'] / eta['heat_ref']
                #             + eta['electricity'] / eta['elec_ref'])) *\
                #       (eta['electricity'] / eta['elec_ref'])
                # parameter.loc[
                #     (fuel, region, 'chp_fin'), 'emission'] = emission / pee
                # parameter.loc[
                #     (fuel, region, 'chp_fin'), 'var_costs'] = var_costs / pee
            elif len(hp) > 1:
                print('error')
            else:
                print('Missing hp: {0}'.format(str(chp)))
                parameter.loc[
                    (fuel, region, 'chp'), 'emission'] = float('nan')
                parameter.loc[
                    (fuel, region, 'chp'), 'var_costs'] = 0
    return parameter


def get_market_clearing_price(es, with_chp=False):
    parameter = fetch_cost_emission(es, with_chp=with_chp)
    my_results = es.results['main']

    # Filter all flows
    flows = [x for x in my_results if x[1] is not None]
    flows_to_elec = [x for x in flows if x[1].label.cat == 'bus' and
                     x[1].label.tag == 'electricity']

    # Filter tags and categories that should not(!) be considered
    tag_list = ['ee']
    cat_list = ['line', 'storage', 'shortage']
    if with_chp is False:
        tag_list.append('chp')
    flows_trsf = [x for x in flows_to_elec if x[0].label.cat not in cat_list
                  and x[0].label.tag not in tag_list]

    # Filter unused flows
    flows_not_null = [x for x in flows_trsf if sum(
        my_results[x]['sequences']['flow']) > 0]

    # Create merit order for each time step
    merit_order = pd.DataFrame()
    for flow in flows_not_null:
        seq = my_results[flow]['sequences']['flow']
        merit_order[flow[0]] = seq * 0
        lb = flow[0].label
        var_costs = parameter.loc[(lb.subtag, lb.region, lb.tag), 'var_costs']
        merit_order[flow[0]].loc[seq > 0] = var_costs
    merit_order['max'] = merit_order.max(axis=1)

    return merit_order['max']


def get_average_yearly_price(es):
    """
    (Wholesale) prices
        (Quantity-weighted?) average yearly price	eur/MWh	absolute difference or % change
    """
    market_clearing_price = get_market_clearing_price(es)
    average_yearly_price = market_clearing_price.mean()
    return average_yearly_price


def get_generation(es):
    """
    Yearly generation
        Differentiated by technologies on next sheet	TWh	absolute difference or % change
    	Storage cycles; only for electricity storage technologies	without unit	absolute difference or % change
    	Renewable curtailment	a) absolute (TWh), and/or b) relative to potential RES generation	absolute difference or % change
    """
    r = es.results['Main']

    generation_dict = {}
    gen = (i for i in r.keys() if i[1] is not None and i[1].label.cat == 'bus')
    for i in gen:
        if i[1].label.tag == 'heat':
            if i[1].label.region == 'DE':
                if i[0].label.cat == 'shortage':
                    generation_dict['heat', 'shortage', i[1].label.subtag, i[1].label.region] = \
                        r[i]['sequences']['flow']
                elif i[0].label.cat == 'trsf':
                    generation_dict['heat', 'decentral', i[1].label.subtag, i[1].label.region] = \
                        r[i]['sequences']['flow']
            else:
                if i[0].label.cat == 'shortage':
                    generation_dict['heat', 'shortage', i[0].label.subtag, i[0].label.region] = \
                        r[i]['sequences']['flow']
                else:
                    generation_dict['heat', i[0].label.tag, i[0].label.subtag, i[0].label.region] = \
                        r[i]['sequences']['flow']

        elif i[1].label.tag == 'electricity':
            if i[0].label.cat == 'shortage':
                generation_dict['electricity', i[0].label.cat, 'None', i[1].label.region] = \
                    r[i]['sequences']['flow']
            elif (i[0].label.cat == 'line') or (i[0].label.cat == 'storage'):
                # lines and storages
                pass
            else:
                generation_dict['electricity', i[0].label.tag, i[0].label.subtag, i[1].label.region] = \
                   r[i]['sequences']['flow']
        else:
            # commodity busses
            pass

    generation_df = pd.DataFrame(generation_dict)
    return generation_df


def get_yearly_generation(es):
    generation_df = get_generation(es)
    # spatial sum
    generation_df = generation_df.sum(axis=1, level=[0, 1, 2])
    generation_df = generation_df.sort_index(axis=1, level=[0, 1, 2])
    # temporal sum
    generation_df = generation_df.sum(axis=0)
    # total
    generation_df['electricity','total','total'] = generation_df['electricity'].drop('shortage').sum()
    generation_df['heat','total','total'] = generation_df['heat'].drop('shortage').sum()
    return generation_df


def get_shortage(generation_df):
    """
    Cross-border exchange
        Gross exports and imports or just net ex-/imports?	TWh	absolute difference or % change
    """
    shortage = generation_df[:,'shortage']
    return shortage


def get_emissions(es):
    """
    Carbon emissions
        One aggregated value	Mio tonnes	% change
    """
    r = es.results['Main']
    p = es.results['Param']

    emission_df = pd.DataFrame()

    for i in r.keys():
        if (i[0].label.cat == 'source') & (i[0].label.tag == 'commodity'):
            emission_df.loc[i[0].label.subtag, 'specific_emission [kgCO2/MWh]'] = (
                p[i]['scalars']['emission'])
            emission_df.loc[i[0].label.subtag, 'summed_flow [MWh]'] = (
                r[i]['sequences']['flow'].sum())

    emission_df['total_emission [kgCO2]'] = (emission_df['specific_emission [kgCO2/MWh]'] *
                                     emission_df['summed_flow [MWh]'])

    emission_df.sort_index(inplace=True)
    return emission_df


def get_start_ups(es):
    """
    Start-ups
        Yearly number of start-ups by technology	without unit	absolute difference or % change
    	Yearly start-up costs by technology	eur	% change
    """
    generation_df = get_generation(es)
    nonzero = generation_df.apply(lambda x: x != 0)

    # nonzero['electricity','shortage','None','DE01'][0] = True
    # print(nonzero[:-1].reset_index(drop=True))
    # print(nonzero[1:].reset_index(drop=True))

    startups = nonzero[:-1].reset_index(drop=True) < nonzero[1:].reset_index(drop=True)
    startups = startups.sum(axis=0)
    startups = startups.sort_index(level=[0, 1, 2])
    startups = startups.sum(level=[0, 1, 2])
    return startups


def get_demand(es):
    """
    Demand
        Yearly (final) electricity demand per country	TWh	-
    	Peak (hourly) demand per country	GW(h)	-
    """
    r = es.results['Main']

    demand_dict = {}
    gen = (i for i in r.keys() if i[1] is not None)
    for i in gen:
        if (i[1].label.cat == 'demand'):
            if (i[1].label.tag == 'electricity'):
                demand_dict[i[1].label.tag, i[1].label.region] = \
                    r[i]['sequences']['flow']

            if (i[1].label.tag == 'heat'):
                demand_dict[i[1].label.tag, i[1].label.subtag] = \
                    r[i]['sequences']['flow']

    demand_df = pd.DataFrame(demand_dict)

    demand_df['electricity', 'total'] = demand_df['electricity'].sum(axis=1)
    demand_df['heat', 'total'] = demand_df['heat'].sum(axis=1)
    demand_total = demand_df[[('electricity', 'total'), ('heat', 'total')]]
    demand_sum = demand_total.sum() * 1e-3
    demand_max = demand_total.max() * 1e-6
    demand = pd.concat([demand_sum, demand_max], axis=1, keys=['sum [MWh]', 'max [MW]'])
    return demand


def get_formatted_results(costs, installed_capacity, yearly_generation, cycles, emissions, average_yearly_price, startups, demand):
    r"""
    Gives back results in the standard output format as agreed upon with all
    model experiment participants.

    Returns
    -------
    formatted_results : pandas.DataFrame
    """
    abs_path = os.path.dirname(os.path.abspath(__file__))
    formatted_results = pd.read_csv(os.path.join(abs_path, 'ose_output_template_deflex.csv'))
    formatted_results['Model'] = 'deflex'
    formatted_results['Scenario'] = 'deflex'

    mapping = pd.read_csv(os.path.join(abs_path, 'mapping_results_to_output_template.csv'))
    for index, row in mapping.iterrows():
        to_variable = row['to_variable']
        from_table = row['from_table']
        key = row[['key_0', 'key_1', 'key_2', 'key_3']]
        print(locals()[from_table])
        formatted_results.loc[formatted_results['Variable'] == to_variable, 'Value'] = 1
    return formatted_results


def postprocess(es_filename, results_path):
    sc = Scenario()
    sc.restore_es(filename=es_filename)
    es = sc.es

    demand = get_demand(es)
    yearly_generation = get_yearly_generation(es)
    shortage = get_shortage(yearly_generation)
    startups = get_start_ups(es)
    emissions = get_emissions(es)
    var_costs = get_var_costs(es)
    average_yearly_price = get_average_yearly_price(es)
    installed_capacity = get_installed_capacity(es)
    cap_costs = get_cap_costs(es)
    # lcoe = get_lcoe(es)
    costs = pd.DataFrame()
    cycles = pd.DataFrame()
    formatted_results = get_formatted_results(costs,
                                              installed_capacity,
                                              yearly_generation,
                                              cycles,
                                              emissions,
                                              average_yearly_price,
                                              startups,
                                              demand)
    print(formatted_results[['Variable', 'Unit', 'Value']])

    # param = fetch_cost_emission(es)
    # print(param)
    # print(param.mean(level=0))

    # save
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    print(results_path)

    demand.to_csv(results_path + '/' + 'demand.csv')
    yearly_generation.to_csv(results_path + '/' + 'yearly_generation.csv')
    shortage.to_csv(results_path + '/' + 'shortage.csv')
    startups.to_csv(results_path + '/' + 'startups.csv')
    emissions.to_csv(results_path + '/' + 'emissions.csv')
    var_costs.to_csv(results_path + '/' + 'var_costs.csv')
    pd.Series(average_yearly_price).to_csv(results_path + '/' + 'average_yearly_price.csv')
    installed_capacity.to_csv(results_path + '/' + 'installed_capacity.csv')
    # cap_costs.to_csv('postproc_results/cap_costs.csv')
    costs.to_csv(results_path + '/' + 'costs.csv')
    cycles.to_csv(results_path + '/' + 'cycles.csv')
    formatted_results.to_csv(results_path + '/' + 'formatted_results.csv')

    # print('\n ### demand \n', demand)
    # print('\n ### yearly generation \n', yearly_generation)
    # print('\n ### shortage \n', shortage)
    # print('\n ### startups \n', startups)
    # print('\n ### emissions \n', emissions)
    # print('\n ### variable_cost \n', var_costs)
    # print('\n ### average yearly price \n', average_yearly_price)
    # print('\n ### installed_capacity \n', installed_capacity)
    # print(installed_capacity.drop([('electricity', 'line'), ('electricity', 'storage')]).sum(level=0))
    # print('\n ### average_yearly_price [Eur/MWh] \n', average_yearly_price)
    # print('\n ### cap_costs \n', cap_costs)

if __name__=='__main__':
    dpath = '/home/jann/reegis/scenarios/deflex/2012/results_cbc/'
    filename = 'deflex_2012_de02.esys'  # 'deflex_2012_de21.esys'
    es_filename = dpath + filename
    results_path = f'/home/jann/reegis/scenarios/deflex/2012/postproc_results_{re.split(".esys", filename)[0]}/'
    postprocess(es_filename, results_path)