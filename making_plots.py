# Use tools from plotting file
from plotting import *

sampling = '1M_sampled/'
# Uniform Ensemble A
#uAfilename = '/../../../data1/acme/Elections/Uniform_Flips/Utah_ensemble_Dec_2020_original_US_District_flips99999999.parquet'
#uAfilename = 'Utah_ensemble_Dec_2020_original_US_District_flips.parquet'
uAfilename = sampling + 'UniformA.parquet.gzip'

# Weighted Ensemble A
#wAfilename = '/../../../data1/acme/Elections/jwmurri_files/1608336391d.parquet.gzip'
#wAfilename = '10k_sampled/uniformA.parquet.gzip'
wAfilename = sampling + 'WeightedA.parquet.gzip'

# Uniform Ensemble B
#uBfilename = '/../../../data1/acme/Elections/Uniform_Flips/Utah_ensemble_Dec_2020_original_US_District_flips99999999.parquet'

# Weighted Ensemble B
#wBfilename = '/../../../data1/acme/Elections/jwmurri_files/10chains10000000d.parquet.gzip'

# Recom ensemble
#recom_filename = '/../../../data1/acme/Elections/Uniform_Recom/Utah_ensemble_recom_paper.parquet'
#recom_filename = 'Utah_ensemble_recom_paper.parquet'
recom_filename = sampling + 'Recom.parquet.gzip'

# Filename dictionary
# filenames = {'UniformA': uAfilename, 'WeightedA': wAfilename, 'UniformB': uBfilename, 'WeightedB': wBfilename, 'Recom': recom_filename}
filenames = {'UniformA': uAfilename, 'WeightedA': wAfilename, 'Recom': recom_filename}

formal_names = {'UniformA': 'Uniform Emsemble (100M Plans)',
                'WeightedA': 'Weighted Ensemble (100M Plans)',
                'UniformB': 'Uniform Ensemble B (100M Plans)',
                'WeightedB': 'Weighted Ensemble B (100M Plans)',
                'Recom': 'Recom Ensemble (1M Plans)'}

# Figsizes
line = 6.0
aspect_ratio = 2.0/3.0
size = np.array([line, aspect_ratio*line])
figsizes = {'full_page': tuple(size), 'half_page': tuple(size/2), 'third_page': tuple(size/3)}

current_scores = np.array([[ 5.00000000e+00,  4.50878123e+02,  3.09000000e+02,  9.16399169e-02,
        6.97623474e-02,  7.98859963e-02, 2.06731581e-03, 3.84714492e-02,
       2.18760007e-02, 1.95924691e-01, 1.58250192e-01, 1.76520838e-01,
        0.00000000e+00, 2.50000000e-01, 2.50000000e-01,  4.13463162e-03,
        7.69428984e-02,  4.37520013e-02,  4.00000000e+00,  4.00000000e+00,
        4.00000000e+00,  1.95092889e-01,  2.16981746e-01,  1.92343543e-01,
        2.14267378e-01,  6.90971000e+05,  6.90975000e+05,  6.90971000e+05,
        6.90968000e+05,  5.93691352e-01,  6.15434047e-01,  6.92016443e-01,
        7.05489875e-01,  5.48816137e-01,  7.04678118e-01,  7.10064374e-01,
        7.12040558e-01,  5.70726821e-01,  6.63692777e-01,  7.02281252e-01,
        7.07743206e-01]])

enacted_plan = pd.DataFrame(current_scores, columns=['County Splits', 'Mattingly Splits Score', 'Cut Edges',
       'Avg Abs Partisan Dislocation - SEN',
       'Avg Abs Partisan Dislocation - G',
       'Avg Abs Partisan Dislocation - COMB', 'Mean Median - SEN',
       'Mean Median - G', 'Mean Median - COMB', 'Efficiency Gap - SEN',
       'Efficiency Gap - G', 'Efficiency Gap - COMB', 'Partisan Bias - SEN',
       'Partisan Bias - G', 'Partisan Bias - COMB', 'Partisan Gini - SEN',
       'Partisan Gini - G', 'Partisan Gini - COMB', 'Seats Won - SEN',
       'Seats Won - G', 'Seats Won - COMB', 'PP1', 'PP2', 'PP3', 'PP4', 'POP1',
       'POP2', 'POP3', 'POP4', 'Sorted SenRep Vote Share 1',
       'Sorted SenRep Vote Share 2', 'Sorted SenRep Vote Share 3',
       'Sorted SenRep Vote Share 4', 'Sorted GRep Vote Share 1',
       'Sorted GRep Vote Share 2', 'Sorted GRep Vote Share 3',
       'Sorted GRep Vote Share 4', 'Sorted CombRep Vote Share 1',
       'Sorted CombRep Vote Share 2', 'Sorted CombRep Vote Share 3',
       'Sorted CombRep Vote Share 4'], index=['ep'])

def annika_data_transform(df):
    """
    Function to transform Annika's data into Jacob's format
    """
    column_renames = {'sen_mean_median': 'Mean Median - SEN',
                      'gub_mean_median': 'Mean Median - G',
                      'com_mean_median': 'Mean Median - COMB',
                      'sen_partisan_bias': 'Partisan Bias - SEN',
                      'gub_partisan_bias': 'Partisan Bias - G',
                      'com_partisan_bias': 'Partisan Bias - COMB',
                      'sen_efficiency_gap': 'Efficiency Gap - SEN',
                      'gub_efficiency_gap': 'Efficiency Gap - G',
                      'com_efficiency_gap': 'Efficiency Gap - COMB',
                      'sen_partisan_gini': 'Partisan Gini - SEN',
                      'gub_partisan_gini': 'Partisan Gini - G',
                      'com_partisan_gini': 'Partisan Gini - COMB',
                      'sen_seats_won': 'Seats Won - SEN',
                      'gub_seats_won': 'Seats Won - G',
                      'com_seats_won': 'Seats Won - COMB',
                      'sen_partisan_dislocation': 'Avg Abs Partisan Dislocation - SEN',
                      'gub_partisan_dislocation': 'Avg Abs Partisan Dislocation - G',
                      'com_partisan_dislocation': 'Avg Abs Partisan Dislocation - COMB',
                      'Sorted SenRep Vote Share 1': 'Sorted SenRep Vote Share 1',
                      'Sorted SenRep Vote Share 2': 'Sorted SenRep Vote Share 2',
                      'Sorted SenRep Vote Share 3': 'Sorted SenRep Vote Share 3',
                      'Sorted SenRep Vote Share 4': 'Sorted SenRep Vote Share 4',
                      'Sorted GRep Vote Share 1': 'Sorted GRep Vote Share 1',
                      'Sorted GRep Vote Share 2': 'Sorted GRep Vote Share 2',
                      'Sorted GRep Vote Share 3': 'Sorted GRep Vote Share 3',
                      'Sorted GRep Vote Share 4': 'Sorted GRep Vote Share 4',
                      'Sorted CombRep Vote Share 1': 'Sorted CombRep Vote Share 1',
                      'Sorted CombRep Vote Share 2': 'Sorted CombRep Vote Share 2',
                      'Sorted CombRep Vote Share 3': 'Sorted CombRep Vote Share 3',
                      'Sorted CombRep Vote Share 4': 'Sorted CombRep Vote Share 4'}

    # col_index_map = np.array([6, 12, 9, 15, 18, 3, 7, 13, 10, 16, 19, 4, 8, 14, 11, 17, 20, 5, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])

    df.rename(columns=column_renames, inplace=True)
    df[['Seats Won - SEN', 'Seats Won - G', 'Seats Won - COMB']] = 4-df[['Seats Won - SEN', 'Seats Won - G', 'Seats Won - COMB']]


def make_all_figsizes(ensemble_type, subdirectory='Plots/', dpi=300, just_one=None):
    """
    Add docstring
    """
    assert ensemble_type in list(filenames.keys())

    # Import data
    f = filenames[ensemble_type]
    print(f'Importing data from {f}')
    data  = pd.read_parquet(filenames[ensemble_type])

    # Transform data to Jacob's format if necessary
    if 'm' in ensemble_type:
        annika_data_transform(data)

    if just_one is not None:
        make_all_plots(data, ensemble_type, just_one, subdirectory, dpi)
        return 'Finished'

    for figsize_code in figsizes.keys():
        make_all_plots(data, ensemble_type, figsize_code, subdirectory, dpi)

    return 'Finished'

def declination_utah(vs):

    # Get basic parameters
    N = vs.shape[1]
    k = np.count_nonzero(vs > 0.5, axis=1) + 1
    kp = N - k + 2

    # Get vote share in each precinct (p vector)
    won_p = (vs-0.5)*(vs > 0.5)+0.5
    lost_p = (vs-0.5)*(vs < 0.5)+0.5

    # Calculate y and z
    y = (1/k)*np.sum(0.5 - lost_p, axis=1)
    z = (1/kp)*np.sum(won_p - 0.5, axis=1)

    # Calculate declination
    theta_D = np.arctan((2*z)/(kp/N))
    theta_R = np.arctan((2*y)/(k/N))
    dec = (2/np.pi)*(theta_D-theta_R)

    return dec


def make_all_plots(data, ensemble_type, figsize_code, subdirectory='Plots/', dpi=300):
    """
    Add docstring
    """
    # make sure we have avalid key
    assert figsize_code in list(figsizes.keys())

    # Set parameters
    figsize = figsizes[figsize_code]
    params = {'figsize':figsize, 'dpi':dpi, 'save':True}
    ensemble = ensemble_type

    # Number of categories
    m = 4

    # Switch sign of signed measures to match convention
    signed_measures = ['Partisan Bias - SEN', 'Partisan Bias - G', 'Partisan Bias - COMB', 'Efficiency Gap - SEN', 'Efficiency Gap - G', 'Efficiency Gap - COMB', 'Mean Median - SEN', 'Mean Median - G', 'Mean Median - COMB']
    data[signed_measures] = -data[signed_measures]
    # data.iloc[:, 6:15] = -data.iloc[:, 6:15]

    # Append current plan data
    data = pd.concat((enacted_plan[list(data.columns)], data))

    try:
        pp = data[['PP1', 'PP2', 'PP3', 'PP4']]
        # pp = data.iloc[:, 21:21+m]
        data['Mean Polsby Popper'] = pp.mean(axis=1)
        data['Max Polsby Popper'] = pp.max(axis=1)
    except KeyError:
        print('Polsby-popper data not available')

    try:
        # pop = data.iloc[:, 21+m:21+2*m]
        pop = data[['POP1', 'POP2', 'POP3', 'POP4']]
        data['Population Standard Deviation, % of Ideal'] = pop.std(axis=1, ddof=0)/pop.mean(axis=1)
        data['Population Max-Min, % of Ideal'] = (pop.max(axis=1) - pop.min(axis=1))/pop.mean(axis=1)
    except KeyError:
        print('Population data not available')

    # Vote Shares
    sen10 = ['Sorted SenRep Vote Share 1', 'Sorted SenRep Vote Share 2', 'Sorted SenRep Vote Share 3', 'Sorted SenRep Vote Share 4']
    gov10 = ['Sorted GRep Vote Share 1', 'Sorted GRep Vote Share 2', 'Sorted GRep Vote Share 3', 'Sorted GRep Vote Share 4']
    comb10 = ['Sorted CombRep Vote Share 1', 'Sorted CombRep Vote Share 2', 'Sorted CombRep Vote Share 3', 'Sorted CombRep Vote Share 4']

    vote_share_sen10 = pd.DataFrame(data[sen10].values, columns=np.arange(1, m+1))
    data['Stdev-SEN'] = np.std(data[sen10].values, axis=1)

    vote_share_gov10 = pd.DataFrame(data[gov10].values, columns=np.arange(1, m+1))
    data['Stdev-GOV'] = np.std(data[gov10].values, axis=1)

    vote_share_comb10 = pd.DataFrame(data[comb10].values, columns=np.arange(1, m+1))
    data['Stdev-COMB'] = np.std(data[comb10].values, axis=1)

    # Get buffered declination data
    data['Buffered Declination - SEN'] = declination_utah(data[sen10].values)
    data['Buffered Declination - G'] = declination_utah(data[gov10].values)
    data['Buffered Declination - COMB'] = declination_utah(data[comb10].values)

    # Discretize efficiency_gap scores
    #for i in range(9, 12):
    #    eg = np.array(data.iloc[:, i])
    #    mask = eg < 0
    #    eg[mask] = np.mean(eg[mask], axis=0)
    #    eg[~mask] = np.mean(eg[~mask], axis=0)
    #    data.iloc[:, i] = eg

    #1: Vote share distribution plots
    title = 'Vote Shares in {}'.format(formal_names[ensemble_type])
    ylabel = 'Republican Vote Share'
    xlabel = 'Sorted US Congressional District'


    boxplots = {'Box Plot Sen 2010':   {'title': title,
                                        'ylabel': f'{ylabel} (Senate 2010)',
                                        'xlabel': xlabel,
                                        'savetitle': f'{subdirectory}{ensemble}-SEN-VoteShares-BoxPlot-{figsize_code}-{dpi}dpi.pdf'},

            'Box Plot Gov 2010':       {'title': title,
                                        'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                        'xlabel': xlabel,
                                        'savetitle': f'{subdirectory}{ensemble}-GOV-VoteShares-BoxPlot-{figsize_code}-{dpi}dpi.pdf'},

            'Box Plot Comb 2010':       {'title': title,
                                        'ylabel': f'{ylabel} (Combined 2010)',
                                        'xlabel': xlabel,
                                        'savetitle': f'{subdirectory}{ensemble}-COMB-VoteShares-BoxPlot-{figsize_code}-{dpi}dpi.pdf'},

            'Violin Plot Sen 2010':    {'title': title,
                                        'ylabel': f'{ylabel} (Senate 2010)',
                                        'xlabel': xlabel,
                                        'savetitle': f'{subdirectory}{ensemble}-SEN-VoteShares-ViolinPlot-{figsize_code}-{dpi}dpi.pdf'},

            'Violin Plot Gov 2010':    {'title': title,
                                        'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                        'xlabel': xlabel,
                                        'savetitle': f'{subdirectory}{ensemble}-GOV-VoteShares-ViolinPlot-{figsize_code}-{dpi}dpi.pdf'},

            'Violin Plot Comb 2010':    {'title': title,
                                        'ylabel': f'{ylabel} (Combined 2010)',
                                        'xlabel': xlabel,
                                        'savetitle': f'{subdirectory}{ensemble}-COMB-VoteShares-ViolinPlot-{figsize_code}-{dpi}dpi.pdf'},
               }

    # Box plot: Senate 2010
    key = 'Box Plot Sen 2010'
    make_box_plot(vote_share_sen10, **boxplots[key], **params)
    print('Finished Box Plot 1')

    # Violin plot: Senate 2010
    key = 'Violin Plot Sen 2010'
    make_violin_plot(vote_share_sen10, **boxplots[key], **params)

    print('Finished Violin Plot 1')

    # Box plot: Governor 2010
    key = 'Box Plot Gov 2010'
    make_box_plot(vote_share_gov10, **boxplots[key], **params)

    print('Finished Box Plot 2')

    # Violin plot: Gov 2010
    key = 'Violin Plot Gov 2010'
    make_violin_plot(vote_share_gov10, **boxplots[key], **params)

    print('Finished Violin Plot 2')

    # Box plot: Governor 2010
    key = 'Box Plot Comb 2010'
    make_box_plot(vote_share_comb10, **boxplots[key], **params)

    print('Finished Box Plot 3')

    # Violin plot: Gov 2010
    key = 'Violin Plot Comb 2010'
    make_violin_plot(vote_share_comb10, **boxplots[key], **params)

    print('Finished Violin Plot 3')

    plt.close('all')

    counter = 6


    #2: Summary plots for other metrics


    ylabel = 'Number of Plans'

    metricplots = {'Avg Abs Partisan Dislocation - SEN': {'title': 'Avg Abs Partisan Dislocation in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Average Absolute Partisan Dislocation (Senate 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-AvgAbsPD-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Avg Abs Partisan Dislocation - G': {'title': 'Avg Abs Partisan Dislocation in {}'.format(formal_names[ensemble_type]),
                                                   'xlabel': 'Average Absolute Partisan Dislocation (Gubernatorial 2010)',
                                                   'ylabel': ylabel,
                                                   'savetitle': f'{subdirectory}{ensemble}-GOV-AvgAbsPD-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Avg Abs Partisan Dislocation - COMB': {'title': 'Avg Abs Partisan Dislocation in {}'.format(formal_names[ensemble_type]),
                                                    'xlabel': 'Average Absolute Partisan Dislocation (Senate 2010)',
                                                    'ylabel': ylabel,
                                                    'savetitle': f'{subdirectory}{ensemble}-COMB-AvgAbsPD-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Mean Median - SEN': {'title': 'Mean-Median Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Mean-Median Score (Senate 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-MeanMedian-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Mean Median - G': {'title': 'Mean-Median Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Mean-Median Score (Gubernatorial 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-MeanMedian-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Mean Median - COMB': {'title': 'Mean-Median Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Mean-Median Score (Combined 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-MeanMedian-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Efficiency Gap - SEN': {'title': 'Efficiency Gap in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Efficiency Gap (Senate 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-EfficiencyGap-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Efficiency Gap - G': {'title': 'Efficiency Gap in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Efficiency Gap (Gubernatorial 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-EfficiencyGap-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Efficiency Gap - COMB': {'title': 'Efficiency Gap in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Efficiency Gap (Combined 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-EfficiencyGap-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Partisan Bias - SEN': {'title': 'Partisan Bias Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Bias Score (Senate 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-PartisanBias-BarChart-{figsize_code}-{dpi}dpi.pdf'},
                    'Partisan Bias - G': {'title': 'Partisan Bias Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Bias Score (Gubernatorial 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-PartisanBias-BarChart-{figsize_code}-{dpi}dpi.pdf'},
                    'Partisan Bias - COMB': {'title': 'Partisan Bias Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Bias Score (Combined 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-PartisanBias-BarChart-{figsize_code}-{dpi}dpi.pdf'},
                    'Partisan Gini - SEN': {'title': 'Partisan Gini Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Gini Score (Senate 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-PartisanGini-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Partisan Gini - G': {'title': 'Partisan Gini Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Gini Score (Gubernatorial 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-PartisanGini-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Partisan Gini - COMB': {'title': 'Partisan Gini Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Gini Score (Combined 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-PartisanGini-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Seats Won - SEN': {'title': 'Districts Won in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Districts Won by Republican Party (Senate 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-SeatsWon-BarChart-{figsize_code}-{dpi}dpi.pdf'},
                    'Seats Won - G':  {'title': 'Districts Won in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Districts Won by Republican Party (Gubernatorial 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-SeatsWon-BarChart-{figsize_code}-{dpi}dpi.pdf'},
                    'Seats Won - COMB': {'title': 'Districts Won in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Districts Won by Republican Party (Combined 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-SeatsWon-BarChart-{figsize_code}-{dpi}dpi.pdf'},
                    'Buffered Declination - SEN': {'title': 'Buffered Declination in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Buffered Declination (Senate 2010)',
                                                  'ylabel': f'{ylabel} (Senate 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-BufferedDeclination-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Buffered Declination - G':  {'title': 'Buffered Declination in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Buffered Declination (Gubernatorial 2010)',
                                                  'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-BufferedDeclination-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Buffered Declination - COMB': {'title': 'Buffered Declination in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Buffered Declination (Combined 2010)',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-BufferedDeclination-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'County Splits' : {'title': 'Number of Split Counties in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Number of Split Counties',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-CountySplits-BarChart-{figsize_code}-{dpi}dpi.pdf'},
                    'Mattingly Splits Score' :  {'title': 'Mattingly Split Counties Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Mattingly Split Counties Score',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-MattinglySplitsScore-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Cut Edges' :  {'title': 'Number of Cut Edges in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Number of Cut Edges',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-CutEdges-BarChart-{figsize_code}-{dpi}dpi.pdf'},
                    'Mean Polsby Popper':  {'title': 'Mean Polsby-Popper Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Mean Polsby-Popper Score',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-MeanPP-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Max Polsby Popper': {'title': 'Max Polsby-Popper Score in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Max Polsby-Popper Score',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-MaxPP-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Population Standard Deviation, % of Ideal':  {'title': 'Population Deviation in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Standard Deviation of District Populations, % of Ideal',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-StdevPop-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Population Max-Min, % of Ideal': {'title': 'Max Population Deviation in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Largest Deviation in District Populations (Max-Min, % of Ideal)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-PopMaxMin-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Stdev-SEN': {'title': 'Standard Deviation of Vote Shares in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Standard Deviation of Vote Shares (Senate 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-StdevVS-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Stdev-GOV': {'title': 'Standard Deviation of Vote Shares in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Standard Deviation of Vote Shares (Gubernatorial 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-StdevVS-Histogram-{figsize_code}-{dpi}dpi.pdf'},
                    'Stdev-COMB': {'title': 'Standard Deviation of Vote Shares in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Standard Deviation of Vote Shares (Combined 2010)',
                                                  'ylabel': ylabel,
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-StdevVS-Histogram-{figsize_code}-{dpi}dpi.pdf'},
            }

    for key, p in metricplots.items():
        counter += 1
        try:
            metric = pd.Series(data[key])
            if 'Histogram' in p['savetitle']:
                if len(data) > 999999:
                    bins = 200
                elif len(data) > 9999999:
                    bins = 300
                make_histogram(metric, bins=bins, **params, **p)
            elif 'BarChart' in p['savetitle']:
                if 'Cut Edges' not in key:
                    make_bar_chart(metric, **params, **p)
                else:
                    make_bar_chart(metric, width=1, histogram_like=True, **params, **p)

        except KeyError:
            print(f'There was a KeyError with {key}')
        else:
            print(f'Finished Plot {counter}')

    plt.close('all')


    #3. Correlation Plots

    ylabel = 'LRVS'

    corrplots = {'Avg Abs Partisan Dislocation - SEN': {'title': 'Avg Abs Partisan Dislocation and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Average Absolute Partisan Dislocation (Senate 2010)',
                                                  'ylabel': f'{ylabel} (Senate 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-AvgAbsPD-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted SenRep Vote Share 1'},
                    'Avg Abs Partisan Dislocation - G': {'title': 'Avg Abs Partisan Dislocation and LRVS in {}'.format(formal_names[ensemble_type]),
                                                   'xlabel': 'Average Absolute Partisan Dislocation (Gubernatorial 2010)',
                                                   'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                                   'savetitle': f'{subdirectory}{ensemble}-GOV-AvgAbsPD-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                   'LRVS_col': 'Sorted GRep Vote Share 1'},
                    'Avg Abs Partisan Dislocation - COMB': {'title': 'Avg Abs Partisan Dislocation and LRVS in {}'.format(formal_names[ensemble_type]),
                                                    'xlabel': 'Average Absolute Partisan Dislocation (Senate 2010)',
                                                    'ylabel': f'{ylabel} (Combined 2010)',
                                                    'savetitle': f'{subdirectory}{ensemble}-COMB-AvgAbsPD-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                    'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Mean Median - SEN': {'title': 'Mean-Median Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Mean-Median Score (Senate 2010)',
                                                  'ylabel': f'{ylabel} (Senate 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-MeanMedian-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted SenRep Vote Share 1'},
                    'Mean Median - G': {'title': 'Mean-Median Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Mean-Median Score (Gubernatorial 2010)',
                                                  'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-MeanMedian-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted GRep Vote Share 1'},
                    'Mean Median - COMB': {'title': 'Mean-Median Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Mean-Median Score (Combined 2010)',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-MeanMedian-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Efficiency Gap - SEN': {'title': 'Efficiency Gap and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Efficiency Gap (Senate 2010)',
                                                  'ylabel': f'{ylabel} (Senate 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-EfficiencyGap-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted SenRep Vote Share 1'},
                    'Efficiency Gap - G': {'title': 'Efficiency Gap in and LRVS {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Efficiency Gap (Gubernatorial 2010)',
                                                  'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-EfficiencyGap-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted GRep Vote Share 1'},
                    'Efficiency Gap - COMB': {'title': 'Efficiency Gap and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Efficiency Gap (Combined 2010)',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-EfficiencyGap-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Partisan Bias - SEN': {'title': 'Partisan Bias Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Bias Score (Senate 2010)',
                                                  'ylabel': f'{ylabel} (Senate 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-PartisanBias-LRVSCorr-ViolinPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted SenRep Vote Share 1'},
                    'Partisan Bias - G': {'title': 'Partisan Bias Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Bias Score (Gubernatorial 2010)',
                                                  'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-PartisanBias-LRVSCorr-ViolinPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted GRep Vote Share 1'},
                    'Partisan Bias - COMB': {'title': 'Partisan Bias Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Bias Score (Combined 2010)',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-PartisanBias-LRVSCorr-ViolinPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Partisan Gini - SEN': {'title': 'Partisan Gini Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Gini Score (Senate 2010)',
                                                  'ylabel': f'{ylabel} (Senate 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-PartisanGini-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted SenRep Vote Share 1'},
                    'Partisan Gini - G': {'title': 'Partisan Gini Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Gini Score (Gubernatorial 2010)',
                                                  'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-PartisanGini-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted GRep Vote Share 1'},
                    'Partisan Gini - COMB': {'title': 'Partisan Gini Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Partisan Gini Score (Combined 2010)',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-PartisanGini-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Seats Won - SEN': {'title': 'Districts Won and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Districts Won by Republican Party (Senate 2010)',
                                                  'ylabel': f'{ylabel} (Senate 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-SeatsWon-LRVSCorr-ViolinPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted SenRep Vote Share 1'},
                    'Seats Won - G':  {'title': 'Districts Won and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Districts Won by Republican Party (Gubernatorial 2010)',
                                                  'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-SeatsWon-LRVSCorr-ViolinPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted GRep Vote Share 1'},
                    'Seats Won - COMB': {'title': 'Districts Won and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Districts Won by Republican Party (Combined 2010)',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-SeatsWon-LRVSCorr-ViolinPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Buffered Declination - SEN': {'title': 'Buffered Declination and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Buffered Declination (Senate 2010)',
                                                  'ylabel': f'{ylabel} (Senate 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-BufferedDeclination-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted SenRep Vote Share 1'},
                    'Buffered Declination - G':  {'title': 'Buffered Declination and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Buffered Declination (Gubernatorial 2010)',
                                                  'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-BufferedDeclination-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted GRep Vote Share 1'},
                    'Buffered Declination - COMB': {'title': 'Buffered Declination and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Buffered Declination (Combined 2010)',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-BufferedDeclination-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'County Splits' : {'title': 'Number of Split Counties and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Number of Split Counties',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-CountySplits-LRVSCorr-ViolinPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Mattingly Splits Score' :  {'title': 'Mattingly Split Counties Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Mattingly Split Counties Score',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-MattinglySplitsScore-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Cut Edges' :  {'title': 'Number of Cut Edges and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Number of Cut Edges',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-CutEdges-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Mean Polsby Popper':  {'title': 'Mean Polsby-Popper Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Mean Polsby-Popper Score',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-MeanPP-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Max Polsby Popper': {'title': 'Max Polsby-Popper Score and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Max Polsby-Popper Score',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-MaxPP-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Population Standard Deviation, % of Ideal':  {'title': 'Population Deviation and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Standard Deviation of District Populations, % of Ideal',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-StdevPop-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Population Max-Min, % of Ideal': {'title': 'Max Population Deviation and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Largest Deviation in District Populations (Max-Min, % of Ideal)',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-PopMaxMin-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'},
                    'Stdev-SEN': {'title': 'Standard Deviation of Vote Shares and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Standard Deviation of Vote Shares (Senate 2010)',
                                                  'ylabel': f'{ylabel} (Senate 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-SEN-StdevVS-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted SenRep Vote Share 1'},
                    'Stdev-GOV': {'title': 'Standard Deviation of Vote Shares and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Standard Deviation of Vote Shares (Gubernatorial 2010)',
                                                  'ylabel': f'{ylabel} (Gubernatorial 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-GOV-StdevVS-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted GRep Vote Share 1'},
                    'Stdev-COMB': {'title': 'Standard Deviation of Vote Shares and LRVS in {}'.format(formal_names[ensemble_type]),
                                                  'xlabel': 'Standard Deviation of Vote Shares (Combined 2010)',
                                                  'ylabel': f'{ylabel} (Combined 2010)',
                                                  'savetitle': f'{subdirectory}{ensemble}-COMB-StdevVS-LRVSCorr-ScatterPlot-{figsize_code}-{dpi}dpi.pdf',
                                                  'LRVS_col': 'Sorted CombRep Vote Share 1'}
            }

    for key, p in corrplots.items():
        counter += 1
        try:
            if 'Scatter' in p['savetitle']:
                ten_recom = 'B' in ensemble_type
                best_fit_line = ('Avg Abs' not in key) and ('Partisan Gini' not in key) and ('Buffered' not in key) and ('Efficiency' not in key)

                make_scatter_correlation(data, key, best_fit_line=best_fit_line, ten_recom=ten_recom, **params, **p)

            elif 'Violin' in p['savetitle']:
                make_violin_correlation(data, key, **params, **p)

        except:
            print(f'There was an error with {key}.')
        else:
            print(f'Finished Plot {counter}')

    plt.close('all')

    #4. Make 10-step histogram for LRVS scores

    title = 'LRVS in {} by Starting Point'.format(formal_names[ensemble_type])
    xlabel = 'LRVS'
    ylabel = 'Number of Plans'

    LRVS_histplots = {'Sorted SenRep Vote Share 1':   {'title': title,
                                        'xlabel': f'{xlabel} (Senate 2010)',
                                        'ylabel': ylabel,
                                        'savetitle': f'{subdirectory}{ensemble}-SEN-10StepLRVS-Histogram-{figsize_code}-{dpi}dpi.pdf'},

            'Sorted GRep Vote Share 1':       {'title': title,
                                        'xlabel': f'{xlabel} (Gubernatorial 2010)',
                                        'ylabel': ylabel,
                                        'savetitle': f'{subdirectory}{ensemble}-GOV-10StepLRVS-Histogram-{figsize_code}-{dpi}dpi.pdf'},

            'Sorted CombRep Vote Share 1':       {'title': title,
                                        'xlabel': f'{xlabel} (Combined 2010)',
                                        'ylabel': ylabel,
                                        'savetitle': f'{subdirectory}{ensemble}-COMB-10StepLRVS-Histogram-{figsize_code}-{dpi}dpi.pdf'}}

    for key, p in LRVS_histplots.items():
        counter += 1
        make_10step_histogram(data, key, bins=200, discard=0.1, **params, **p)
        print(f'Finished Plot {counter}')

    #5. Make side by side correlation plots

    ylabel = 'LRVS'
    LRVS_cols = ['Sorted SenRep Vote Share 1', 'Sorted GRep Vote Share 1', 'Sorted CombRep Vote Share 1']
    xlabels = ['Senate 2010', 'Gubernatorial 2010', 'Combined 2010']


    corrplots = [{'keys': ['Avg Abs Partisan Dislocation - SEN', 'Avg Abs Partisan Dislocation - G', 'Avg Abs Partisan Dislocation - COMB'],
                  'title': 'Avg Abs Partisan Dislocation and LRVS in {}'.format(formal_names[ensemble_type]),
                  'xlabel': 'Average Absolute Partisan Dislocation',
                  'savetitle': f'{subdirectory}{ensemble}-AvgAbsPD-LRVSCorr-3ScatterPlot-{figsize_code}-{dpi}dpi.pdf'},
                 {'keys': ['Mean Median - SEN', 'Mean Median - G', 'Mean Median - COMB'],
                  'title': 'Mean-Median Score and LRVS in {}'.format(formal_names[ensemble_type]),
                  'xlabel': 'Mean-Median Score',
                  'savetitle': f'{subdirectory}{ensemble}-MeanMedian-LRVSCorr-3ScatterPlot-{figsize_code}-{dpi}dpi.pdf'},
                 {'keys': ['Efficiency Gap - SEN', 'Efficiency Gap - G', 'Efficiency Gap - COMB'],
                  'title': 'Efficiency Gap and LRVS in {}'.format(formal_names[ensemble_type]),
                  'xlabel': 'Efficiency Gap',
                  'savetitle': f'{subdirectory}{ensemble}-EfficiencyGap-LRVSCorr-3ScatterPlot-{figsize_code}-{dpi}dpi.pdf'},
                 {'keys': ['Partisan Bias - SEN', 'Partisan Bias - G', 'Partisan Bias - COMB'],
                  'title': 'Partisan Bias and LRVS in {}'.format(formal_names[ensemble_type]),
                  'xlabel': 'Partisan Bias',
                  'savetitle': f'{subdirectory}{ensemble}-PartisanBias-LRVSCorr-3ViolinPlot-{figsize_code}-{dpi}dpi.pdf'},
                 {'keys': ['Partisan Gini - SEN', 'Partisan Gini - G', 'Partisan Gini - COMB'],
                  'title': 'Partisan Gini and LRVS in {}'.format(formal_names[ensemble_type]),
                  'xlabel': 'Partisan Gini',
                  'savetitle': f'{subdirectory}{ensemble}-PartisanGini-LRVSCorr-3ScatterPlot-{figsize_code}-{dpi}dpi.pdf'},
                 {'keys': ['Seats Won - SEN', 'Seats Won - G', 'Seats Won - COMB'],
                  'title': 'Districts Won and LRVS in {}'.format(formal_names[ensemble_type]),
                  'xlabel': 'Districts Won',
                  'savetitle': f'{subdirectory}{ensemble}-SeatsWon-LRVSCorr-3ViolinPlot-{figsize_code}-{dpi}dpi.pdf'},
                 {'keys': ['Buffered Declination - SEN', 'Buffered Declination - G', 'Buffered Declination - COMB'],
                   'title': 'Buffered Declination and LRVS in {}'.format(formal_names[ensemble_type]),
                   'xlabel': 'Buffered Declination',
                   'savetitle': f'{subdirectory}{ensemble}-BufferedDeclination-LRVSCorr-3ScatterPlot-{figsize_code}-{dpi}dpi.pdf'},
                 {'keys': ['Stdev-SEN', 'Stdev-GOV', 'Stdev-COMB'],
                   'title': 'Standard Deviation of Vote Shares and LRVS in {}'.format(formal_names[ensemble_type]),
                   'xlabel': 'StandardDeviation of Vote Shares',
                   'savetitle': f'{subdirectory}{ensemble}-StdevVS-LRVSCorr-3ScatterPlot-{figsize_code}-{dpi}dpi.pdf'}
                ]

    for p in corrplots:
        counter += 1
        if 'Scatter' in p['savetitle']:
            ten_recom = 'B' in ensemble_type
            key = p['keys'][0]
            best_fit_line = ('Avg Abs' not in key) and ('Partisan Gini' not in key) and ('Buffered' not in key) and ('Efficiency' not in key)

            make_scatter_correlation_3plots(data, p['keys'], LRVS_col=LRVS_cols, title=p['title'], ylabel=ylabel, common_xlabel=p['xlabel'], xlabels=xlabels, best_fit_line=best_fit_line, ten_recom=ten_recom, savetitle=p['savetitle'], **params)

        elif 'Violin' in p['savetitle']:
            make_violin_correlation_3plots(data, p['keys'], LRVS_col=LRVS_cols, title=p['title'], ylabel=ylabel, common_xlabel=p['xlabel'], xlabels=xlabels, savetitle=p['savetitle'], **params)

        print(f'Finished Plot {counter}')

    plt.close('all')

make_all_figsizes('Recom', subdirectory='PaperPlots-v1/', dpi=300, just_one='full_page')
make_all_figsizes('UniformA', subdirectory='PaperPlots-v1/', dpi=300, just_one='full_page')
make_all_figsizes('WeightedA', subdirectory='PaperPlots-v1/', dpi=300, just_one='full_page')
#make_all_figsizes('WeightedB', subdirectory='PaperPlots/', dpi=300, just_one='full_page')
#make_all_figsizes('UniformB', subdirectory='PaperPlots/', dpi=300, just_one='full_page')
