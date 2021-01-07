# Import necessary packages
from gerrychain import (GeographicPartition, Partition, Graph)
import pandas as pd
import geopandas as gp
import numpy as np
from scipy import optimize as opt
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon

def plot_district_map(assignment, size=(3,2), dpi=300, cmap='viridis', precincts='UtahData/gdf_august.shp', save=False, savetitle=None):
    """
    Given a districting assignment, plots the state of Utah divided into the given districts using 2018 precinct data.

    Parameters:
        assignment (gerrychain.Assignment): a districting assignment. A dictionary will work as long as it has the right keys
        size ((2,) tuple): figure size
        dpi (int): figure resolution
        precincts: Filename of a geodataframe containing the shapefiles. Alternatively, pass it in directly

    """
    # Load in the shapefiles
    if type(precincts) == str:
        precincts = gp.read_file(precincts)
    else:
        precincts = precincts.copy()

    # Load the district data into the geodataframe containing the shapefiles
    if type(assignment) != np.ndarray and type(assignment) != pd.Series:
        precincts['plot_val'] = [assignment[i] for i in range(len(assignment))]
    else:
        precincts['plot_val'] = assignment

    # Plot the data
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    precincts.plot(column='plot_val', ax=ax, cmap=cmap)
    plt.axis("off")

    # Save if desired
    if save: plt.savefig(savetitle, dpi=dpi)
    plt.show()

def plot_graph(precincts, graph, window=None, node_size=0.1, line_size=0.05, dpi=400, size=7, save=False, savetitle=None):
    """
    Plots the precinct adjacency graph, over the state of Utah.

    Selected Parameters:
        precincts (str): filename of precinct shapefiles. Alternatively, pass them in directly
        graph (str): filename of graph. Alternatively, pass in directly
        cmap: The matplotlib cmap used in plotting.
        size (int): figure size (we use a fixed aspect ratio for Utah)
        dpi (int): figure resolution
    """

    # Load in the shapefiles
    if type(precincts) == str:
        precincts = gp.read_file(precincts)
    else:
        precincts = precincts.copy()

    # Load in the graph
    if type(graph) == str:
        graph = Graph.from_json(graph)

    # Obtain a graph coloring
    d = nx.greedy_color(graph, strategy='largest_first')
    coloring = np.array([d[i] for i in range(len(graph))])

    precincts['color'] = coloring
    precincts['center'] = precincts.centroid   # the location of the nodes
    nodes = gp.GeoDataFrame(geometry=precincts.center)    # make a GeoDataFrame of nodes
    E = [LineString([precincts.loc[a,'center'],precincts.loc[b,'center']])
         for a,b in list(graph.edges)]         # Construct a line for each edge
    edges = gp.GeoDataFrame(list(graph.edges), geometry=E) # make a geoDataFrame of edges

    fig = plt.figure(dpi=dpi)               # Set up the figure
    fig.set_size_inches(size,size*2)        # Make it have the same proportions as Utah
    ax = plt.subplot(1,1,1)

    precincts.plot('color', cmap='tab20', ax=ax, alpha=0.5)   # Plot precincts
    nodes.plot(ax=ax,color='k', markersize=node_size)               # Plot nodes
    edges.plot(ax=ax, lw=line_size, color='k')                      # plot edges
    if window is None:
        plt.axis('off')                                     # no coordinates
    else:
        plt.axis(window)

    if save: plt.savefig(savetitle, bbox_inches='tight', dpi=dpi)       # Save it

def calc_percentile(val, data):
    return opt.bisect(lambda x: np.percentile(data, x) - val, 0, 100)

def make_box_plot(data, title='', ylabel='', xlabel='', figsize=(6,8), dpi=400, savetitle=None, save=False, current_plan_name='Enacted plan'):
    """
    Makes a box plot of the given data, with the specified parameters.

    Parameters:
        data (DataFrame) dataframe with columns corresponding to the vote shares.
    """
    # set parameters
    n = len(data)
    m = len(data.iloc[0])
    k = max(1, m//14)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axhline(0.5, color="#cccccc")
    data.boxplot(ax=ax, positions=range(1, m+1), sym='', zorder=1)
    ax.scatter(data.iloc[0].index, data.iloc[0], color="r", marker="o", s=25/k, alpha=0.5, zorder=5, label=current_plan_name)
    ax.legend(loc='lower right')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks([i for i in range(1, m+1)])

    # Hide enough xlabels to that they don't overlap or look crowded
    if k > 1:
        for i,label in enumerate(ax.xaxis.get_ticklabels()):
            if i%k:
                label.set_visible(False)

    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches='tight')
    plt.clf()

def get_areas(shapes, norm='L1', vert=True):

    areas = []
    for i in range(4):
        if vert:
            v = shapes[i].__dict__['_paths'][0].vertices-np.array([0,i+1])
            if norm=='L2':
                v[:, 1] = v[:, 1]*np.abs(v[:, 1])
                area = np.sqrt(Polygon(v).area)
            else:
                area = Polygon(v).area
        else:
            v = shapes[i].__dict__['_paths'][0].vertices-np.array([i+1, 0])
            if norm=='L2':
                v[:, 0] = v[:, 0]*np.abs(v[:, 0])
                area = np.sqrt(Polygon(v).area)
            else:
                area = Polygon(v).area

        areas.append(area)

    return np.array(areas)

def expand_segments(segments, expansion, vert=True):

    paths = segments.get_paths()
    new_paths = []
    for i, path in enumerate(paths):
        if vert:
            offset = np.array([i+1, 0])
            new_path = (path.vertices-offset)*np.array([expansion[i],1])+offset
            new_paths.append(new_path)
        else:
            offset = np.array([0, i+1])
            new_path = (path.vertices-offset)*np.array([1, expansion[i]])+offset
            new_paths.append(new_path)

    segments.set_paths(new_paths)

def expand_polygons(shapes, expansion, offsets=None, vert=True):

    new_paths = []
    for i, p in enumerate(shapes):
        path = p.get_paths()[0]
        if vert:
            offset = np.array([i+1, 0])
            if offsets is not None: offset = np.array([offsets[i], 0])
            new_path = (path.vertices-offset)*np.array([expansion[i],1])+offset
        else:
            offset = np.array([0, i+1])
            if offsets is not None: offset = np.array([0, offsets[i]])
            new_path = (path.vertices-offset)*np.array([1, expansion[i]])+offset

        p.set_paths([new_path])

def make_violin_plot(data, title='', ylabel='', xlabel='', figsize=(6,8), dpi=400, savetitle=None, positions=None, save=False, area_normalizer='Linf', xticks=None, current_plan_name='Enacted plan', vert=True, bw_method=0.1, alpha=0.8, dist_height=1, widths=0.2, points=200, **kwargs):
    """
    Make a violin plot of the given data, with the specified parameters.
    Only pass in the columns of the dataframe which contain the vote shares.
    """
    d = data.T
    m = len(d)

    # Construct initial plots
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    result = ax.violinplot(d, bw_method=bw_method, widths=widths, points=points, vert=vert, **kwargs)

    if area_normalizer != 'Linf':
        areas = get_areas(result['bodies'], norm=area_normalizer, vert=vert)
        expand_polygons(result['bodies'], expansion=dist_height*np.min(areas)/(areas*widths), vert=vert)
    else:
        expand_polygons(result['bodies'], expansion=dist_height*np.ones(m)/widths, vert=vert)

    for pc in result['bodies']:
        pc.set_alpha(alpha)

    if vert:
        ax.axhline(0.5, color="#cccccc")
        ax.hlines(y=d.iloc[:, 0], xmin = np.arange(m)+1-0.2, xmax=np.arange(m)+1+0.2, color='r', lw=2, label=current_plan_name)
        for i in range(m):
            plt.text(i+1+0.2, d.iloc[i, 0], str(int(np.round(calc_percentile(d.iloc[i, 0], d.iloc[i]),0)))+'%', horizontalalignment='left', verticalalignment='center')

        ax.set_ylim(0.4, 0.9)
        # ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        if xticks is None:
            ax.set_xticks([i for i in range(1, m+1)])
            ax.set_xlim(0.5, m+0.5)
        else:
            ax.set_xticks(xticks)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    else:
        ax.axvline(0.5, color="#cccccc")
        ax.vlines(x=d.iloc[:, 0], ymin = np.arange(m)+1-0.2, ymax=np.arange(m)+1+0.2, color='r', lw=2, label=current_plan_name)
        for i in range(m):
            plt.text(d.iloc[i, 0]+0.007, i+1+0.2, str(int(np.round(calc_percentile(d.iloc[i, 0], d.iloc[i]),0)))+'%', horizontalalignment='center', verticalalignment='bottom')

        ax.set_xlim(0.4, 0.9)
        #ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        if xticks is None:
            ax.set_yticks([i for i in range(1, m+1)])
            ax.set_ylim(0.5, m+0.5)
        else:
            ax.set_yticks(xticks)

        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)


    ax.set_title(title)
    ax.legend(loc='lower right')

    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.clf()

def make_histogram(data, title='', ylabel='', xlabel='', figsize=(6,8), dpi=400, bins=50, savetitle=None, save=False):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    data.hist(bins=bins)
    ax.axvline(x=data.iloc[0], color='r', lw=2, label='Enacted plan, '+str(np.round(calc_percentile(data[0], data),1))+'%')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')

    plt.show()
    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches='tight')
    plt.clf()

def make_bar_chart(data, title='', ylabel='', xlabel='', figsize=(6,8), dpi=400, savetitle=None, save=False):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    unique_vals = np.unique(data)
    ax.bar(unique_vals, [np.count_nonzero(unique_vals[i] == data) for i in range(len(unique_vals))], width=np.min(unique_vals[1:]-unique_vals[:-1])/2)
    ax.set_xticks(unique_vals)

    ax.axvline(x=data.iloc[0], color='r', lw=2, label='Enacted plan, ' +str(np.round(calc_percentile(data[0], data),1))+'%')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    plt.show()

    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches='tight')
    plt.clf()

def make_violin_correlation(data, key, LRVS_col, title='', ylabel='', xlabel='', figsize=(6,8), dpi=400, savetitle=None, save=False, area_normalizer='Linf', current_plan_name='Enacted plan', vert=True, bw_method=0.1, alpha=0.8, dist_height=1, widths=0.2, points=200, **kwargs):
    """
    Parameters:
        data (DataFrame)
        key (str)
        LRVS_col (str)
    """

    # Extract unique values
    unique_vals = np.unique(data[key])

    # Create a list of rows in the data corresponding to each unique value
    LRVS_separated = [data[data[key]==unique_vals[i]][LRVS_col].values for i in range(len(unique_vals))]

    # Get the minimum separation distance between the unique values
    scale = np.min(unique_vals[1:]-unique_vals[:-1])

    # Get the amount of data in each list
    bar_heights = [len(samples) for samples in LRVS_separated]
    width_scaling = np.min(bar_heights)/bar_heights

    # Construct plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    result = ax.violinplot(LRVS_separated, positions=unique_vals, bw_method=bw_method, widths=widths, points=points, vert=vert, **kwargs)

    # Normalize areas
    if area_normalizer != 'Linf':
        areas = get_areas(result['bodies'], norm=area_normalizer, vert=vert)
        expand_polygons(result['bodies'], offsets=unique_vals, expansion=scale*width_scaling*dist_height*np.min(areas)/(areas*widths), vert=vert)
    else:
        expand_polygons(result['bodies'], offsets=unique_vals, expansion=scale*width_scaling*dist_height*np.ones_like(unique_vals)/widths, vert=vert)

    # Set alpha
    for pc in result['bodies']:
        pc.set_alpha(alpha)

    if vert:
        ax.axhline(0.5, color="#cccccc")
        ax.set_ylim(0.4, 0.6)
        ax.set_yticks([0.4, 0.5, 0.6])
        ax.set_xlim(np.min(unique_vals)-scale/2, np.max(unique_vals)+scale/2)
        ax.set_xticks(unique_vals)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    else:
        ax.axvline(0.5, color="#cccccc")
        ax.set_xlim(0.4, 0.6)
        ax.set_xticks([0.4, 0.5, 0.6])
        ax.set_ylim(np.min(unique_vals)-scale/2, np.max(unique_vals)+scale/2)
        ax.set_yticks(unique_vals)
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)

    ax.set_title(title)
    ax.legend(loc='lower right')

    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.clf()

def make_violin_correlation_3plots(data, key, LRVS_col, title='', ylabel='', common_xlabel='', xlabels='', figsize=(6,8), dpi=400, savetitle=None, save=False, area_normalizer='Linf', current_plan_name='Enacted plan', vert=True, bw_method=0.1, alpha=0.8, dist_height=1, widths=0.2, points=200, **kwargs):
    """
    Parameters:
        data (DataFrame)
        key (list) list of 3 strings
        LRVS_col (list) list of 3 strings
    """
    # Construct plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharey=True)

    for i, ax in enumerate([ax1, ax2, ax3]):

        # Extract unique values
        unique_vals = np.unique(data[key[i]])

        # Create a list of rows in the data corresponding to each unique value
        LRVS_separated = [data[data[key[i]]==unique_vals[j]][LRVS_col[i]].values for j in range(len(unique_vals))]

        # Get the minimum separation distance between the unique values
        scale = np.min(unique_vals[1:]-unique_vals[:-1])

        # Get the amount of data in each list
        bar_heights = [len(samples) for samples in LRVS_separated]
        width_scaling = np.min(bar_heights)/bar_heights

        result = ax.violinplot(LRVS_separated, positions=unique_vals, bw_method=bw_method, widths=widths, points=points, vert=vert, **kwargs)

        # Normalize areas
        if area_normalizer != 'Linf':
            areas = get_areas(result['bodies'], norm=area_normalizer, vert=vert)
            expand_polygons(result['bodies'], offsets=unique_vals, expansion=scale*width_scaling*dist_height*np.min(areas)/(areas*widths), vert=vert)
        else:
            expand_polygons(result['bodies'], offsets=unique_vals, expansion=scale*width_scaling*dist_height*np.ones_like(unique_vals)/widths, vert=vert)

        # Set alpha
        for pc in result['bodies']:
            pc.set_alpha(alpha)

        ax.axhline(0.5, color="#cccccc")
        ax.set_xlim(np.min(unique_vals)-scale/2, np.max(unique_vals)+scale/2)
        ax.set_xticks(unique_vals)
        ax.set_xlabel(xlabels[i])

        ax.set_ylim(0.4, 0.6)
        ax.set_yticks([0.4, 0.45, 0.5, 0.55, 0.6])
        ax.legend(loc='lower right')

    fig.suptitle(title, x=0.5, y=1)
    fig.text(0.5, 0, common_xlabel, ha='center', va='bottom')
    fig.text(0, 0.5, ylabel, ha='left', va='center', rotation='vertical')
    fig.tight_layout()

    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.clf()

def make_scatter_correlation(data, key, LRVS_col, best_fit_line=True, ten_recom=False, step=None, title='', ylabel='', alpha=0.4, xlabel='', figsize=(8,6), dpi=400, savetitle=None, save=False):

    n = len(data)
    m = int(n/10)

    if step is None:
        step = max(1, int(n/10000))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y = data[LRVS_col].values[::step]
    x = data[key].values[::step]

    if best_fit_line: ax.axvline(0.0, c="#cccccc", lw=2)
    ax.axhline(0.5, color="#cccccc", lw=2)

    if ten_recom:
        m = int(len(x)/10)
        for j in range(10):
            x1 = x[j*m:(j+1)*m].copy()
            y1 = y[j*m:(j+1)*m].copy()
            plt.scatter(x1, y1, s=1, alpha=alpha)
    else:
        plt.scatter(x, y, s=1, alpha=alpha)

    plt.scatter(x[0], y[0], s=10, c='red', marker='*', label='Enacted Plan')

    if best_fit_line:

        # Draw a line of best fit
        SStot = np.sum(np.square(y-np.mean(y)))
        p, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
        m, c = p[0], p[1]
        SSres = np.sum(residuals)
        R2 = 1-SSres/SStot
        domain = np.linspace(np.min(x), np.max(x), 200)
        plt.plot(domain, m*domain+c, label=r'Linear Best Fit, $R^2={}$'.format(np.round(R2, 2)), c='black', lw=1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    plt.show()

    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches='tight')
    plt.clf()

def make_scatter_correlation_3plots(data, key, LRVS_col, best_fit_line=True, ten_recom=False, step=None, title='', ylabel='', common_xlabel='', xlabels='', figsize=(8,6), dpi=400, alpha=0.4, savetitle=None, save=False):


    # Construct plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharey=True)

    n = len(data)
    m = int(n/10)

    for i, ax in enumerate([ax1, ax2, ax3]):

        if step is None: step = max(1, int(n/10000))

        y = data[LRVS_col[i]].values[::step]
        x = data[key[i]].values[::step]

        if best_fit_line: ax.axvline(0.0, c="#cccccc", lw=1)
        ax.axhline(0.5, color="#cccccc", lw=1)

        if ten_recom:
            m = int(len(x)/10)
            for j in range(10):
                x1 = x[j*m:(j+1)*m].copy()
                y1 = y[j*m:(j+1)*m].copy()
                ax.scatter(x1, y1, s=1, alpha=alpha)
        else:
            ax.scatter(x, y, s=1, alpha=alpha)

        ax.scatter(x[0], y[0], s=10, c='red', marker='*', label='Enacted Plan')

        if best_fit_line:

            # Draw a line of best fit
            SStot = np.sum(np.square(y-np.mean(y)))
            p, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
            m, c = p[0], p[1]
            SSres = np.sum(residuals)
            R2 = 1-SSres/SStot
            domain = np.linspace(np.min(x), np.max(x), 200)
            ax.plot(domain, m*domain+c, label= r'$R^2={}$'.format(np.round(R2, 2)), c='black', lw=1)

        ax.set_xlabel(xlabels[i])

        ax.set_ylim(0.4, 0.6)
        ax.set_yticks([0.4, 0.45, 0.5, 0.55, 0.6])
        ax.legend(loc='best')

    fig.suptitle(title, x=0.5, y=1)
    fig.text(0.5, 0, common_xlabel, ha='center', va='bottom')
    fig.text(0, 0.5, ylabel, ha='left', va='center', rotation='vertical')
    fig.tight_layout()

    plt.show()


def make_10step_histogram(data, LRVS_col, bins=200, discard=0.1, title='', ylabel='', xlabel='', figsize=(6,8), dpi=400, savetitle=None, save=False):

    n = len(data)
    m = int(n/10)


    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = data[LRVS_col].values

    for i in range(10):
        if i > 0:
            label = 'Recom {}'.format(i)
        else:
            label = 'Original'
        x1 = pd.Series(x[i*m:(i+1)*m].copy())
        x1[int(discard*m):].hist(ax = ax, bins = bins, alpha = .1, color='C'+str(i))
        x1[int(discard*m):].hist(ax = ax, histtype='step', bins = bins, lw=1, facecolor='None', color='C'+str(i), label=label)
        ax.axvline(x1[0], lw=2, color='C'+str(i))

    ax.axvline(0.5, color="#cccccc")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    plt.show()

    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches='tight')
    plt.clf()

# Analyzing the rejection rate of a chain
def acceptance_series(d):
    """
    Creates an np.array with length d.shape[0], where the ith entry is 1 if
    rows i and i-1 of d are different, and 0 otherwise.

    Parameters:
        d (iterable)

    Returns:
        s (np.array)
    """
    series = np.zeros(d.shape[0], dtype=np.uint8)
    for i in range(1, d.shape[0]):
        if not np.allclose(d.iloc[i, :], d.iloc[i-1, :]):
            series[i] = 1
    return series

def running_mean(x, N):
    """
    Returns a moving average array of the data in x over an N-period interval.

    Parameters:
        x (iterable)

    Returns:
        m (np.array)
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_acceptance_rate(data, period):
    """
    Plot the (period)-moving-average of the acceptance rate of a chain.

    Parameters:
        data (pd.DataFrame)
        period (int): the number of iterations to average over
    """
    s = acceptance_series(data)

    # Position the moving average at the center of the period it is the average over
    plt.plot(np.linspace(period/2, len(s)-period/2, len(s)-period+1), running_mean(s, period))

    plt.title('{}-Iteration Moving Average Acceptance Rate'.format(period))
    plt.ylabel('Acceptance Rate')
    plt.xlabel('Iteration')
    plt.axis([0, len(s), 0, 1])
    plt.show()

def make_plots(idnum, kind, subdirectory='Plots/', figsize=(8,6), dpi=400, file_type='.pdf'):
    """
    Given the id number of a chain run, creates relevant plots (Utah data format only).

    Parameters:
        idnum (int): the Unix timestamp of the second when the chain was created
                    (part of the filename)
                    note: if idnum is string, will use the given name of the file
        kind (str): the type of chain run
        subdirectory (str): the subdirectory to save the resulting plots to
        figsize (tup): the desired figure size for the plots. Default: (8, 6)
        dpi (int) the desired dots per inch for the plots. Default: 400 dpi
        file_type (str): the desired filetype of the saved plots

    Creates the following plots:
        - Box Plot + Violin Plot of Sen, Gov, and Combined Vote Shares
        - histogram of the following:
            - Average Absolute Partisan Dislocation (Sen, Gov, Combined)
            - Mean Median (Sen, Gov, Combined)
            - Partisan Bias (Sen, Gov, Combined)
            - Partisan Gini (Sen, Gov, Combined)
            - Efficiency Gap (Sen, Gov, Combined)
            - Seats Won  (Sen, Gov, Combined)
            - County splits
            - Mattingly county split score
            - Mean polsby popper
            - Max polsby popper
            - Population standard deviation
            - Population max-min

    Total: 33 plots.
    """
    assert kind in ['flip-uniform', 'flip-mh', 'recom-uniform', 'recom-mh']

    # Extract the data
    if type(idnum) == int:
        if idnum < 1593561600:
            data = pd.read_hdf(str(idnum)+'.h5', 'data')
        else:
            data = pd.read_parquet(str(idnum)+'d.parquet.gzip')
    else:
        data = pd.read_hdf(idnum)

    # Set parameters
    params = {'figsize':figsize, 'dpi':dpi, 'save':True}
    n = len(data)
    if n == 10**8:
        n = '100M'
    elif n == 10**7:
        n = '10M'

    m = int((len(data.columns)-21)/5)

    pp = data.iloc[:, 21:21+m]
    data['Mean Polsby Popper'] = pp.mean(axis=1)
    data['Max Polsby Popper'] = pp.max(axis=1)

    pop = data.iloc[:, 21+m:21+2*m]
    data['Population Standard Deviation, % of Ideal'] = pop.std(axis=1, ddof=0)/pop.mean(axis=1)
    data['Population Max-Min, % of Ideal'] = (pop.max(axis=1) - pop.min(axis=1))/pop.mean(axis=1)

    # Switch sign of signed measures to match convention
    data.iloc[:, 6:15] = -data.iloc[:, 6:15]



    # Set parameters
    common_file_ending = '-'+str(len(data))+'-'+kind+'-'+str(idnum)+file_type

    boxplots = {'Box Plot Sen 2010':   {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Senate 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'BoxPlotSen2010'+common_file_ending},

            'Box Plot Gov 2010':       {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Gubernatorial 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'BoxPlotGov2010'+common_file_ending},

            'Box Plot Comb 2010':       {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Combined 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'BoxPlotComb2010'+common_file_ending},

            'Violin Plot Sen 2010':    {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Senate 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'ViolinPlotSen2010'+common_file_ending},

            'Violin Plot Gov 2010':    {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Gubernatorial 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'ViolinPlotGov2010'+common_file_ending},

            'Violin Plot Comb 2010':    {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Combined 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'ViolinPlotComb2010'+common_file_ending}
               }


    metricplots = {'Avg Abs Partisan Dislocation - SEN': {'title': 'Avg Abs Partisan Dislocation in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Avg Abs Partisan Dislocation (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'AvgAbsPDSen2010'+common_file_ending},
                    'Avg Abs Partisan Dislocation - G': {'title': 'Avg Abs Partisan Dislocation in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Avg Abs Partisan Dislocation (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'AvgAbsPDGov2010'+common_file_ending},
                    'Avg Abs Partisan Dislocation - COMB': {'title': 'Avg Abs Partisan Dislocation in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Avg Abs Partisan Dislocation (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'AvgAbsPDComb2010'+common_file_ending},
                    'Mean Median - SEN': {'title': 'Mean-Median Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Mean-Median Score (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MeanMedianSen2010'+common_file_ending},
                    'Mean Median - G': {'title': 'Mean-Median Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Mean-Median Score (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MeanMedianGov2010'+common_file_ending},
                    'Mean Median - COMB': {'title': 'Mean-Median Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Mean-Median Score (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MeanMedianComb2010'+common_file_ending},
                    'Efficiency Gap - SEN': {'title': 'Efficiency Gap in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Efficiency Gap (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'EfficiencyGapSen2010'+common_file_ending},
                    'Efficiency Gap - G': {'title': 'Efficiency Gap in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Efficiency Gap (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'EfficiencyGapGov2010'+common_file_ending},
                    'Efficiency Gap - COMB': {'title': 'Efficiency Gap in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Efficiency Gap (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'EfficiencyGapComb2010'+common_file_ending},
                    'Partisan Bias - SEN': {'title': 'Partisan Bias Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Bias Score (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanBiasSen2010'+common_file_ending},
                    'Partisan Bias - G': {'title': 'Partisan Bias Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Bias Score (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanBiasGov2010'+common_file_ending},
                    'Partisan Bias - COMB': {'title': 'Partisan Bias Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Bias Score (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanBiasComb2010'+common_file_ending},
                    'Partisan Gini - SEN': {'title': 'Partisan Gini Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Gini Score (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanGiniSen2010'+common_file_ending},
                    'Partisan Gini - G': {'title': 'Partisan Gini Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Gini Score (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanGiniGov2010'+common_file_ending},
                    'Partisan Gini - COMB': {'title': 'Partisan Gini Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Gini Score (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanGiniComb2010'+common_file_ending},
                    'Seats Won - SEN': {'title': 'Seats Won in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Seats Won (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'SeatsWonSen2010'+common_file_ending},
                    'Seats Won - G': {'title': 'Seats Won in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Seats Won (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'SeatsWonGov2010'+common_file_ending},
                    'Seats Won - COMB': {'title': 'Seats Won in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Seats Won (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'SeatsWonComb2010'+common_file_ending},
                    'County Splits' : {'title': 'Split Counties in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Number of Split Counties',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'SplitCounties'+common_file_ending},
                    'Mattingly Splits Score' :  {'title': 'Mattingly Split Counties Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Mattingly Split Counties Score',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MattinglySplitCounties'+common_file_ending},
                    'Cut Edges' :  {'title': 'Cut Edges in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Number of Cut Edges',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'CutEdges'+common_file_ending},
                    'Mean Polsby Popper':  {'title': 'Mean Polsby-Popper Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Mean Polsby-Popper Score',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MeanPolsbyPopper'+common_file_ending},
                    'Max Polsby Popper': {'title': 'Max Polsby-Popper Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Max Polsby-Popper Score',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MaxPolsbyPopper'+common_file_ending},
                    'Population Standard Deviation, % of Ideal':  {'title': 'Population Deviation in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Standard Deviation of District Populations, % of Ideal',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'StdevPop'+common_file_ending},
                    'Population Max-Min, % of Ideal': {'title': 'Population Deviation in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Largest Deviation in District Populations (Max-Min, % of Ideal)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MaxMinPop'+common_file_ending},
            }



    # Box plot: Senate 2010
    key = 'Box Plot Sen 2010'
    vote_share_sen10 = pd.DataFrame(list(data.iloc[:, 21+2*m:21+3*m].values), columns=np.arange(1, m+1))
    make_box_plot(vote_share_sen10, **boxplots[key], **params)

    print('Finished Box Plot 1')

    # Violin plot: Senate 2010
    key = 'Violin Plot Sen 2010'
    make_violin_plot(vote_share_sen10, **boxplots[key], **params)

    print('Finished Violin Plot 1')

    # Box plot: Governor 2010
    key = 'Box Plot Gov 2010'
    vote_share_gov10 = pd.DataFrame(list(data.iloc[:, 21+3*m:21+4*m].values), columns=np.arange(1, m+1))
    make_box_plot(vote_share_gov10, **boxplots[key], **params)

    print('Finished Box Plot 2')

    # Violin plot: Gov 2010
    key = 'Violin Plot Gov 2010'
    make_violin_plot(vote_share_gov10, **boxplots[key], **params)

    print('Finished Violin Plot 2')

    # Box plot: Governor 2010
    key = 'Box Plot Comb 2010'
    vote_share_comb10 = pd.DataFrame(list(data.iloc[:, 21+4*m:21+5*m].values), columns=np.arange(1, m+1))
    make_box_plot(vote_share_comb10, **boxplots[key], **params)

    print('Finished Box Plot 3')

    # Violin plot: Gov 2010
    key = 'Violin Plot Comb 2010'
    make_violin_plot(vote_share_comb10, **boxplots[key], **params)

    print('Finished Violin Plot 3')

    plt.close('all')

    # Construct plots for the various metrics
    for key in metricplots.keys():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        metric = pd.Series(data[key])
        metric.hist(bins=50)
        ax.axvline(x=metric[0], color='r', lw=2, label='Enacted plan, '+str(np.round(calc_percentile(metric[0], metric),1))+'%')
        ax.set_title(metricplots[key]['title'])
        ax.set_xlabel(metricplots[key]['xlabel'])
        ax.set_ylabel(metricplots[key]['ylabel'])
        ax.legend(loc='upper right')
        plt.savefig(metricplots[key]['savetitle'], dpi=dpi, bbox_inches='tight')
        plt.clf()

        print('Finished Plot: {}'.format(key))

        plt.close('all')


def make_correlation_plots(idnum, kind, comment='', step=None, subdirectory='Plots/', figsize=(8,6), dpi=400, file_type='.pdf'):
    """
    Produces a set of correlation plots used to analyze how well the partisan gerrymandering metrics
    perform in the case of Utah.

    Parameters:
        idnum (int): the unix timestamp for when the chain was started.
            if passed in as str, the filename of the chain
        subdirectory (str): the subdirectory to save the resulting plots

    Total: 15 plots.
    """
    assert kind in ['flip-uniform', 'flip-mh', 'recom-uniform', 'recom-mh']

    # Extract the data
    if type(idnum) == int:
        if idnum < 1593561600:
            data = pd.read_hdf(str(idnum)+'.h5', 'data')
        else:
            data = pd.read_hdf(idnum)
    else:
        data = pd.read_parquet(idnum)
        idnum = ''

    # Set parameters
    common_file_ending = '-'+str(len(data))+'-'+kind+'-'+str(idnum)+comment+file_type

    # Set parameters
    params = {'figsize':figsize, 'dpi':dpi, 'save':True}
    n = len(data)
    if step is None:
        step = max(1, int(n/10000))


    correlationplot_xaxis = {'Avg Abs Partisan Dislocation': {'name': 'Average Absolute Partisan Dislocation', 'savetitle':'AvgAbsPD'},
                             'Efficiency Gap': {'name':'Efficiency Gap', 'savetitle':'EG'},
                             'Mean Median': {'name':'Mean Median Score', 'savetitle':'MM'},
                             'Partisan Bias': {'name':'Partisan Bias Score', 'savetitle':'PB'},
                             'Partisan Gini': {'name':'Partisan Gini Score', 'savetitle':'PG'}}

    correlationplot_yaxis = {'G': {'ending': ' - G', 'colname': 'Sorted GRep Vote Share 1', 'title': ' (Gubernatorial 2010)'},
                             'SEN': {'ending': ' - SEN', 'colname': 'Sorted SenRep Vote Share 1', 'title': ' (Senate 2010)'},
                             'COMB': {'ending': ' - COMB', 'colname':'Sorted CombRep Vote Share 1', 'title':' (Combined 2010)'}}

    # Construct plots for the various metrics
    for key in correlationplot_yaxis.keys():
        for key1 in correlationplot_xaxis.keys():
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            if key1 in ['Mean Median', 'Partisan Bias', 'Efficiency Gap']:
                x = -np.array(data[key1+correlationplot_yaxis[key]['ending']])[::step]
            else:
                x = np.array(data[key1+correlationplot_yaxis[key]['ending']])[::step]
            y = np.array(data[correlationplot_yaxis[key]['colname']])[::step]
            m = int(len(x)/10)

            for i in range(10):
                x1 = x[i*m:(i+1)*m].copy()
                y1 = y[i*m:(i+1)*m].copy()
                plt.scatter(x1, y1, s=1, alpha=0.3)

            if key1 in ['Mean Median', 'Partisan Bias', 'Efficiency Gap']:
                ax.axvline(0.0, color="#cccccc")


            SStot = np.sum(np.square(y-np.mean(y)))
            p, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
            m, c = p[0], p[1]
            SSres = np.sum(residuals)
            R2 = 1-SSres/SStot
            domain = np.linspace(np.min(x), np.max(x), 200)

            plt.plot(domain, m*domain+c, label=r'Best Fit, $R^2={}, m={}$'.format(np.round(R2, 2), np.round(m, 2)), c='orange')
            ax.axhline(0.5, color="#cccccc")
            ax.set_title(correlationplot_xaxis[key1]['name']+' and R Vote Share in Least R District in a {}-Plan Ensemble'.format(n)+correlationplot_yaxis[key]['title'])
            ax.set_xlabel(correlationplot_xaxis[key1]['name'])
            ax.set_ylabel('R Vote Share in Least R District')
            plt.legend(loc='upper right')
            plt.savefig(subdirectory+correlationplot_xaxis[key1]['savetitle']+key+'Correlation'+common_file_ending, dpi=dpi, bbox_inches='tight')
            plt.clf()

            print('Finished Plot')

        plt.close('all')

def precincts_moving_frequency(idnum, subdirectory='Plots/', save=False):
    """
    Given the id number of a chain run, generates an array mapping precinct ids to
    the number of recorded times that that precinct changed assignments.

    Parameters:
        idnum (int): the unix timestamp for when the chain was started
        subdirectory (str): the subdirectory to save the result in
        save (bool): whether or not to save the result to a .npy file

    Returns:
        move_frequencies (ndarray): array mapping precinct ids to the number of
        recorded times that the precinct changed assignments
    """
    # Extract the data
    if idnum < 1593561600:
        assignments = pd.read_hdf(str(idnum)+'.h5', 'stored_assignments')
    else:
        assignments= pd.read_parquet(str(idnum)+'a.parquet.gzip')
    n = len(assignments)

    # Compute the changes
    changes = np.array(assignments.iloc[1:, :]) - np.array(assignments.iloc[:-1, :])
    move_frequencies = np.count_nonzero(changes, axis=1)

    # Save if desired
    if save: np.save(subdirectory+str(i)+'moves.npy', move_frequencies)

    # Result the result if desired
    return np.count_nonzero(changes, axis=1)

def create_overlapping_histogram(idnum, kind, discard=0.1, comment='', subdirectory='Plots/', figsize=(8,6), dpi=400, file_type='.pdf'):
    """
    Produces histograms to visualize the distribution of least R district vote shares in Utah

    Parameters:
        idnum (int): the unix timestamp for when the chain was started.
            if passed in as str, the filename of the chain
        subdirectory (str): the subdirectory to save the resulting plots

    Total: 3 plots.
    """
    assert kind in ['flip-uniform', 'flip-mh', 'recom-uniform', 'recom-mh']

    # Extract the data
    if type(idnum) == int:
        if idnum < 1593561600:
            data = pd.read_hdf(str(idnum)+'.h5', 'data')
        else:
            data = pd.read_hdf(idnum)
    else:
        data = pd.read_parquet(idnum)
        idnum = ''

    # Set parameters
    common_file_ending = '-'+str(len(data))+'-'+kind+'-'+str(idnum)+comment+file_type

    # Set parameters
    params = {'figsize':figsize, 'dpi':dpi, 'save':True}
    n = len(data)
    m = int(n/10)


    hist_vals = {'G': {'ending': ' - G', 'colname': 'Sorted GRep Vote Share 1', 'title': ' (Gubernatorial 2010)'},
                 'SEN': {'ending': ' - SEN', 'colname': 'Sorted SenRep Vote Share 1', 'title': ' (Senate 2010)'},
                 'COMB': {'ending': ' - COMB', 'colname':'Sorted CombRep Vote Share 1', 'title':' (Combined 2010)'}}

    # Construct plots for the various metrics
    for key in hist_vals.keys():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        x = np.array(data[hist_vals[key]['colname']])

        for i in range(10):
            if i > 0:
                label = 'Recom {}'.format(i)
            else:
                label = 'Original'
            x1 = pd.Series(x[i*m:(i+1)*m].copy())
            x1[int(discard*m):].hist(ax = ax, bins = 99, alpha = .1, color='C'+str(i))
            x1[int(discard*m):].hist(ax = ax, histtype='step', bins = 99, lw=3, facecolor='None', color='C'+str(i), label=label)
            ax.axvline(x1[0], lw=2, color='C'+str(i))

        ax.axvline(0.5, color="#cccccc")
        ax.set_title('R Vote Share in Least R District in a {}-Plan Ensemble'.format(n)+hist_vals[key]['title'])
        ax.set_ylabel('Number of Plans in Ensemble')
        ax.set_xlabel('R Vote Share in Least R District'+hist_vals[key]['title'])
        plt.legend(loc='upper right')
        plt.savefig(subdirectory+key+'Hist'+common_file_ending, dpi=dpi, bbox_inches='tight')
        plt.clf()

        print('Finished Plot')

        plt.close('all')
