import pandas as pd
import os


def join_results(sc_dir):
    dirs = [os.path.join(sc_dir, dir) for dir in os.listdir(sc_dir)
            if dir.startswith('postproc')]

    all_f_res = pd.DataFrame()

    for dir in dirs:
        f_res = pd.read_csv(os.path.join(dir, 'formatted_results.csv'), index_col=0)
        all_f_res = pd.concat([all_f_res, f_res], ignore_index=True)



    sc_translation = {'2012_de02_electricity-only_100': 'Baseline battery costs|Full geographic coverage|Electricity only',
                      '2012_de02_electricity-only_25': '25percent battery costs|Full geographic coverage|Electricity only',
                      '2012_de02_electricity-only_50': '50percent battery costs|Full geographic coverage|Electricity only',
                      '2012_de02_full_100': 'Baseline battery costs|Full geographic coverage|Electricity and Heat',
                      '2012_de02_full_50': '50percent battery costs|Full geographic coverage|Electricity and Heat',
                      '2012_de02_full_25': '25percent battery costs|Full geographic coverage|Electricity and Heat'}

    all_f_res = all_f_res.replace(sc_translation)
    all_f_res.to_csv(os.path.join(sc_dir, 'deflex_combined_results.csv'))


if __name__ ==  '__main__':
    sc_dir = '/home/jann/reegis/scenarios/deflex/2012/'
    join_results(sc_dir)
