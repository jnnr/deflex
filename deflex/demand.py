# -*- coding: utf-8 -*-

"""Processing a list of power plants in Germany.

Copyright (c) 2016-2018 Uwe Krien <uwe.krien@rl-institut.de>

SPDX-License-Identifier: GPL-3.0-or-later
"""
__copyright__ = "Uwe Krien <uwe.krien@rl-institut.de>"
__license__ = "GPLv3"


# Python libraries
import os
import logging

# External libraries
import pandas as pd
from workalendar.europe import Germany

# oemof libraries
from oemof.tools import logger
import demandlib.bdew as bdew
import demandlib.particular_profiles as profiles

# internal modules
import reegis_tools.config as cfg
import reegis_tools.entsoe
import reegis_tools.bmwi
import reegis_tools.geometries
import reegis_tools.openego
import reegis_tools.heat_demand

import de21.geometries
import de21.inhabitants


def renpass_demand_share():
    demand_share = os.path.join(cfg.get('paths', 'data_de21'),
                                cfg.get('static_sources',
                                        'renpass_demand_share'))
    return pd.read_csv(
            demand_share, index_col='region_code', squeeze=True)


def openego_demand_share():
    demand_reg = prepare_ego_demand()['sector_consumption_sum']
    print(demand_reg)
    demand_sum = demand_reg.sum()
    return demand_reg.div(demand_sum)


def de21_profile_from_entsoe(year, share, annual_demand=None, overwrite=False):
    load_file = os.path.join(cfg.get('paths', 'entsoe'),
                             cfg.get('entsoe', 'load_file'))

    if not os.path.isfile(load_file) or overwrite:
        reegis_tools.entsoe.split_timeseries_file(overwrite)

    # start = datetime.datetime(year, 1, 1, 0, 0)
    # end = datetime.datetime(year, 12, 31, 23, 0)

    entsoe = reegis_tools.entsoe.get_entsoe_load(year)
    # entsoe = entsoe.tz_localize('UTC').tz_convert('Europe/Berlin')
    de_load_profile = entsoe.DE_load_

    load_profile = pd.DataFrame(index=de_load_profile.index)
    for i in range(21):
        region = 'DE{:02.0f}'.format(i + 1)
        if region not in share:
            share[region] = 0
        load_profile[region] = de_load_profile.multiply(float(share[region]))

    if annual_demand is not None:
        load_profile = load_profile.div(load_profile.sum().sum()).multiply(
            annual_demand)
    return load_profile


def prepare_ego_demand(overwrite=False):
    egofile_de21 = os.path.join(cfg.get('paths', 'demand'),
                                cfg.get('demand', 'ego_file_de21'))

    if os.path.isfile(egofile_de21) and not overwrite:
        ego_demand_de21 = pd.read_hdf(egofile_de21, 'demand')
    else:
        ego_demand_df = reegis_tools.openego.get_ego_demand(overwrite=False)
        # Create GeoDataFrame from ego demand file.
        ego_demand = reegis_tools.geometries.Geometry(name='ego demand',
                                                      df=ego_demand_df)

        ego_demand.create_geo_df()

        # Load region polygons
        de21_regions = de21.geometries.de21_regions()

        # Add column with region id
        ego_demand.gdf = reegis_tools.geometries.spatial_join_with_buffer(
            ego_demand, de21_regions)

        # Overwrite Geometry object with its DataFrame, because it is not
        # needed anymore.
        ego_demand_de21 = pd.DataFrame(ego_demand.gdf)

        # Delete the geometry column, because spatial grouping will be done
        # only with the region column.
        del ego_demand_de21['geometry']

        # Write out file (hdf-format).
        ego_demand_de21.to_hdf(egofile_de21, 'demand')

    return ego_demand_de21.groupby('de21_region').sum()


def create_de21_slp_profile(year, outfile):
    demand_de21 = prepare_ego_demand()

    cal = Germany()
    holidays = dict(cal.holidays(year))

    de21_profile = pd.DataFrame()

    for region in demand_de21.index:
        annual_demand = demand_de21.loc[region]

        annual_electrical_demand_per_sector = {
            'g0': annual_demand.sector_consumption_retail,
            'h0': annual_demand.sector_consumption_residential,
            'l0': annual_demand.sector_consumption_agricultural,
            'i0': annual_demand.sector_consumption_industrial}
        e_slp = bdew.ElecSlp(year, holidays=holidays)

        elec_demand = e_slp.get_profile(annual_electrical_demand_per_sector)

        # Add the slp for the industrial group
        ilp = profiles.IndustrialLoadProfile(e_slp.date_time_index,
                                             holidays=holidays)

        elec_demand['i0'] = ilp.simple_profile(
            annual_electrical_demand_per_sector['i0'])

        de21_profile[region] = elec_demand.sum(1).resample('H').mean()
    de21_profile.to_csv(outfile)


def get_de21_slp_profile(year, annual_demand=None, overwrite=False):
    outfile = os.path.join(
        cfg.get('paths', 'demand'),
        cfg.get('demand', 'ego_profile_pattern').format(year=year))
    if not os.path.isfile(outfile) or overwrite:
        create_de21_slp_profile(year, outfile)

    de21_profile = pd.read_csv(
        outfile, index_col=[0], parse_dates=True).multiply(1000)

    if annual_demand is not None:
        de21_profile = de21_profile.div(de21_profile.sum().sum()).multiply(
            annual_demand)

    return de21_profile


def get_de21_profile(year, kind, annual_demand=None, overwrite=False):
    """

    Parameters
    ----------
    year : int
        The year of the profile. The year is passed to the chosen function.
        Make sure the function can handle the given year.
    kind : str
        Name of the method to create the profile
    annual_demand : float
        The annual demand for the profile. By default the original annual
        demand is used.
    overwrite :
        Be aware that some results are stored to speed up the calculations. Set
        overwrite to True or remove the stored files if you are not sure.

    Returns
    -------

    """
    # Use the openEgo proposal to calculate annual demand and standardised
    # load profiles to create profiles.
    if kind == 'openego':
        return get_de21_slp_profile(year, annual_demand, overwrite)

    # Use the renpass demand share values to divide the national entsoe profile
    # into 18 regional profiles.
    elif kind == 'renpass':
        return de21_profile_from_entsoe(year, renpass_demand_share(),
                                        annual_demand, overwrite)

    # Use the openEgo proposal to calculate the demand share values and use
    # them to divide the national entsoe profile.
    elif kind == 'openego_entsoe':
        return de21_profile_from_entsoe(year, openego_demand_share(),
                                        annual_demand, overwrite)

    else:
        logging.error('Method "{0}" not found.'.format(kind))


def elec_demand_tester(year):
    oe = get_de21_profile(year, 'openego') * 1000000
    rp = get_de21_profile(year, 'renpass') * 1000000
    ege = get_de21_profile(year, 'openego_entsoe') * 1000000

    netto = reegis_tools.bmwi.get_annual_electricity_demand_bmwi(year)

    oe_s = get_de21_profile(year, 'openego', annual_demand=netto)
    rp_s = get_de21_profile(year, 'renpass', annual_demand=netto)
    ege_s = get_de21_profile(year, 'openego_entsoe', annual_demand=netto)

    print('[TWh] original    scaled (BMWI)')
    print(' oe:  ', int(oe.sum().sum() / 1e+12), '       ',
          int(oe_s.sum().sum()))
    print(' rp:  ', int(rp.sum().sum() / 1e+12), '       ',
          int(rp_s.sum().sum()))
    print('ege:  ', int(ege.sum().sum() / 1e+12), '       ',
          int(ege_s.sum().sum()))
    print(ege_s)


def get_heat_profiles_de21(year, time_index=None, keep_unit=False):
    heat_demand_state_file = os.path.join(
            cfg.get('paths', 'demand'),
            cfg.get('demand', 'heat_profile_state').format(year=year))
    if os.path.isfile(heat_demand_state_file):
        logging.info("Demand profiles by state exist. Reading file.")
        demand_state = pd.read_csv(heat_demand_state_file, index_col=[0],
                                   parse_dates=True, header=[0, 1, 2])
        demand_state = demand_state.tz_localize('UTC').tz_convert(
            'Europe/Berlin')
    else:
        demand_state = reegis_tools.heat_demand.get_heat_profiles_by_state(
            year, to_csv=True)

    my_index = demand_state.index
    my_columns1 = pd.MultiIndex(levels=[[], []], labels=[[], []])
    demand_region = pd.DataFrame(index=my_index, columns=my_columns1)

    my_columns2 = pd.MultiIndex(levels=[[], [], [], []],
                                labels=[[], [], [], []])
    district_heat_region = pd.DataFrame(index=my_index, columns=my_columns2)

    logging.info("Fetching inhabitants table.")
    my_ew = de21.inhabitants.get_ew_by_de21_subregions(year)
    my_ew = my_ew.replace({'state': cfg.get_dict('STATES')})

    state_ew = my_ew.groupby('state').sum()
    for region in my_ew.index:
        my_ew.loc[region, 'share_state'] = float(
            my_ew.loc[region, 'ew'] / state_ew.loc[my_ew.loc[region, 'state']])

    logging.info("Convert demand profile...")

    fuels = demand_state.columns.get_level_values(2).unique()
    sectors = demand_state.columns.get_level_values(1).unique()
    demand_state = demand_state.swaplevel(2, 0, axis=1)
    district_heat_state = None
    for fuel in fuels:
        if fuel != 'district heating':
            demand_region['DE_demand', fuel] = demand_state[fuel].sum(axis=1)
        else:
            district_heat_state = demand_state[fuel]

    for subregion in my_ew.index:
        state = my_ew.loc[subregion, 'state']
        region = my_ew.loc[subregion, 'region']
        share = my_ew.loc[subregion, 'share_state']
        for sector in sectors:
            district_heat_region[
                region, 'district_heating', sector, subregion] = (
                    district_heat_state[sector, state] * share)
    district_heat_region.sort_index(1, inplace=True)
    # print(district_heat_region)
    district_heat_region = district_heat_region.groupby(
        level=[0, 1], axis=1).sum()
    de21_demand = pd.concat([district_heat_region, demand_region], axis=1)

    if time_index is not None:
        de21_demand.index = time_index

    if not keep_unit:
        msg = ("The unit of the source is 'TJ'. "
               "Will be divided by {0} to get 'MWh'.")
        converter = 0.0036
        de21_demand = de21_demand.div(converter)
        logging.warning(msg.format(converter))

    return de21_demand


if __name__ == "__main__":
    logger.define_logging()
    # print(openego_demand_share())
    # exit(0)
    # elec_demand_tester(2013)
    # prepare_ego_demand()
    # exit(0)
    for y in [2014]:
        df = reegis_tools.heat_demand.get_heat_profiles_by_state(
            y, state=['BE'])['BE']
        print(df.groupby(level=1, axis=1).sum())
        # df = df.swaplevel(axis=1)
        # df = df['district heating'].sum(axis=1)
        # print(df)
        # df.copy().to_csv('/home/uwe/test.csv')
        # exit(0)
        print(get_heat_profiles_de21(y))
        # print(heat_demand(y).loc['BE'].sum().sum() / 3.6)
    logging.info("Done!")