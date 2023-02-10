import math
from dash import Dash, html, dcc, Input, Output  # pip install dash
import dash_bootstrap_components as dbc
import numpy as np
import flask

import matplotlib
import matplotlib.pyplot as plt             # pip install matplotlib
import mpld3  
matplotlib.use("agg")                              


import pandas as pd

# bring data into app

df_avg=pd.read_csv("/Users/abhinaymannepalli/Desktop/Deplatdforming_ALL/deplatforming_2/df_avg.csv")
COLOR_DICT = {'gtab': ['tab:purple', 'tab:olive', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan'],
              'wikimedia': ['tab:green', 'tab:red', 'tab:cyan', 'tab:purple', 'tab:olive', 'tab:orange'],
              'mediacloud': ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive']}
#print(df_avg.columns)
PLATFORM_COLORS = ['#1DA1F2', '#4267B2', '#C13584', '#FF0000']
PLATFORM_COLOR_DICT = {'gtab': PLATFORM_COLORS,
                       'wikimedia': PLATFORM_COLORS,
                       'mediacloud': PLATFORM_COLORS}
# Set up Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Set up the page layout
app.layout = dbc.Container([
    html.H1("Deplatforming"),
    
   #html.Img(id='image'),

    html.Iframe(
        id='avg_plot',
        srcDoc=None, # here is where we will put the graph we make
        style={'border-width': '10', 'width': '100%', 'height': '500px'}),

    html.H5("Source", className='mt-2'),
    dcc.Checklist(
        id='mydropdown',
        value=['gtab'],
        options=[{'label': x, 'value': x} for x in df_avg.source.unique()]),
    html.H5("Filter", className='mt-2'),
    dcc.Dropdown(
        placeholder = 'Select filter',
        id='mydropdown2', 
        options=[{'label':"no_filter", 'value':"no_filter"},
        {'label':"ban_group_single", 'value':"ban_group_single"},
        {'label':"ban_group_multiple", 'value':"ban_group_multiple"},
        {'label':"temporary", 'value':"temporary"},
        {'label':"kind", 'value':"kind"},
        {'label':"reason", 'value':"reason"},
       {'label':"platform", 'value':"platform"}], 
        multi=False, 
        value='no_filter'
    )
])

# Create interactivity between dropdown and graph
@app.callback(
   # Output(component_id='image', component_property='src'),
   Output(component_id='avg_plot', component_property='srcDoc'),
   Input(component_id='mydropdown', component_property='value'),
   Input(component_id='mydropdown2', component_property='value'))
def plot_data(options,filter):
    #print(f"User Seleceted this dropdown value: {selected_ycol}")
    if len(options)==1 and filter=='no_filter':
        avg=avg_plot(df_avg,sources=options[0], title='Average normalized value around the ban - averaged per ban group')
    elif filter=='no_filter':
        avg=avg_plot1(df_avg,sources=options, title='Average normalized value around the ban - averaged per ban group')
    
    elif len(options)==1 and filter=='ban_group_single':
        avg=avg_plot(df_avg[df_avg.bans_in_ban_group == 'single'],sources=options[0], title="Only ban groups with a single ban", which_color=1)
    elif filter=='ban_group_single':
        print("execute")
        avg=avg_plot1(df_avg[df_avg.bans_in_ban_group == 'single'],sources=options, title="Only ban groups with a single ban", which_color=1)

        
    elif len(options)==1 and filter=='ban_group_multiple':
        avg=avg_plot(df_avg[df_avg.bans_in_ban_group == 'multiple'],sources=options[0], title="Only ban groups with a multiple ban", which_color=2)
    elif filter=='ban_group_multiple':
        avg=avg_plot1(df_avg[df_avg.bans_in_ban_group == 'multiple'],sources=options, title="Only ban groups with a multiple ban", which_color=2)

 #  html.Img(src="/Users/abhinaymannepalli/Desktop/depklatforming_2/plot.png")
    elif len(options)==1 and filter=='temporary':
       avg= avg_cat_plot1(df_avg, sources=options[0],cat_feat='temporary', title="Temporary vs permanent bans", labels=['permanent','temporary'])#, horizontal=False)
    elif filter=='temporary':
      avg= avg_cat_plot(df_avg, sources=options,cat_feat='temporary', title="Temporary vs permanent bans", labels=['permanent','temporary'])#, horizontal=False)
    elif len(options)==1 and filter=='kind':
        avg=avg_cat_plot1(df_avg,sources=options[0], cat_feat='bans_in_ban_group', title="Ban groups with single ban vs ban groups with multiple bans", labels=['multiple bans','single ban'])
    elif filter=='kind':
        avg=avg_cat_plot(df_avg,sources=options,cat_feat='kind', title="Kind of ban", labels=['Internet Pers.','Mainstream Pers.','Politician'])   
    elif len(options)==1 and filter=='reason':
        avg=avg_cat_plot1(df_avg, cat_feat='reason',sources=options[0], plot_ci=False, title="Reason of the ban", labels=['Hate','Lewd','Misinfo','Other','Manipulation','Sharing priv info'])
    elif filter=='reason':
        avg=avg_cat_plot(df_avg, cat_feat='reason',sources=options, plot_ci=False, title="Reason of the ban", labels=['Hate','Lewd','Misinfo','Other','Manipulation','Sharing priv info'])
    elif len(options)==1 and filter=='platform':
        avg=avg_cat_plot1(df_avg, cat_feat='platform',sources=options[0], plot_ci=False, title="Platform of the ban", labels=['facebook','instagram','twitter','youtube'], color_dict=PLATFORM_COLOR_DICT)#, horizontal=False)
    else:
        avg=avg_cat_plot(df_avg, cat_feat='platform',sources=options, plot_ci=False, title="Platform of the ban", labels=['facebook','instagram','twitter','youtube'], color_dict=PLATFORM_COLOR_DICT)#, horizontal=False)
    html_matplotlib = mpld3.fig_to_html(avg)
   # html_matplotlib=html.Img(src=app.get_asset_url('rap.png'))
   # image_directory='/Users/abhinaymannepalli/Desktop/Deplatdforming_ALL/deplatforming_2/assets'
    #image_name='rap.png'
    #return   flask.send_from_directory(image_directory, image_name)
    return html_matplotlib

def avg_plot(df,sources, steps=True, plot_ci=True, same_y=False, savefig=False, savename=None, title=None, horizontal=True, color_dict=COLOR_DICT, which_color=0):
    ''' Plot 'simple' average plot, i.e. one line per source'''
    #print("avg fns called")  
    if horizontal: fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    else: fig, axes = plt.subplots(1, 1, figsize=(3, 10))
    #sources = ['gtab']
   # sources=source
    #print(sources)
    #print(axes)
    # Average per ban group (ban_group_id) before averaging per seq, so that entities with many bans don't count more than once
    # dfm = df.groupby(['source','ban_group_id','seq'])['value_norm'].mean().reset_index()
    
    # Alternative: take only first ban per ban group (ban_group_id) before averaging per seq, so that entities with many bans don't count more than once
    dfm = df.groupby(['source','ban_group_id','seq'])['value_norm'].first().reset_index()
    # Plot once per source
    #for source, axis in zip(sources, axes):
    source=sources
    axis=axes
    color = color_dict[source][which_color]

        # Compute average for each seq, together with std and count for plotting the CI
    stats = dfm[dfm.source == source].groupby(['seq'])['value_norm'].agg(['mean', 'count', 'std'])            

    if plot_ci:
        stats = compute_confidence_interval(stats)

    if source in ['mediacloud', 'wikimedia']:
        stats_roll = stats.rolling(7, center=True).mean()
        axis.plot(stats_roll['mean'], color=color)
        if plot_ci: axis.fill_between(stats_roll.index, stats_roll.ci95_hi, stats_roll.ci95_lo, alpha=.1, color=color)
    else:
        xs = stats.index
        ys = stats['mean']
        if steps:
            axis.plot(xs, ys, color=color, ds='steps-post')
            if plot_ci: axis.fill_between(stats.index, stats.ci95_hi, stats.ci95_lo, alpha=.1, color=color, step='post')
        else:
            axis.plot(xs, ys, color=color, ds='default')
            if plot_ci: axis.fill_between(stats.index, stats.ci95_hi, stats.ci95_lo, alpha=.1, color=color)
    
    # Plot decorations
    #for source, axis in zip(sources, axes):
        
        # Correct x axis ticks
    if source == 'gtab':
        axis.set_xticks(np.arange(-6,6,1))
        xticklabels = ['+'+str(xtl+1) if xtl >= 0 else str(xtl) for xtl in axis.get_xticks()]
        axis.set_xticklabels(xticklabels)
    else:            
        axis.set_xticks(axis.get_xticks()[1:-1])
        xticklabels = ['+'+str(int(xtl)) if xtl > 0 else str(int(xtl)) for xtl in axis.get_xticks()]
        axis.set_xticklabels(xticklabels)

        # Vertical line for ban date
    pos = -0.5 if source == 'gtab' else 0
    axis.axvline(pos, color='tab:grey', ls='--')

        # Set title and axis labels
    axis.set_title(source)
    if source == 'gtab': axis.set_xlabel('months before or after the ban')
    else: axis.set_xlabel('days before or after the ban')
    axis.set_ylabel('average normalized value')

        # Trim edges
        # if source == 'gtab': axis.set_xlim(-6,6)
        # else: axis.set_xlim(-180,180)

        # Add annotation
    if source in ['mediacloud', 'wikimedia']:
        plt.text(.02, .97, '* 7-day-rolling averaged', ha='left', va='top', transform=axis.transAxes, style='italic')

        # Set hardcoded ylims
    if same_y:
        if source == 'gtab': axis.set_ylim(0.15, 0.5)
        if source == 'wikimedia': axis.set_ylim(0.0, 0.35)
        if source == 'mediacloud': axis.set_ylim(0.1, 0.45)
        # axis.set_ylim(0,0.5)

    # Add main title
    if title is not None:
        if horizontal: plt.suptitle(title, fontsize=15)
        else: plt.suptitle(title, fontsize=15, y=0.915)

    plt.savefig("plot")
    return fig

def avg_plot1(df, sources,steps=True, plot_ci=True, same_y=False, savefig=False, savename=None, title=None, horizontal=True, color_dict=COLOR_DICT, which_color=0):
    ''' Plot 'simple' average plot, i.e. one line per source'''
        
    if horizontal: 
        if len(sources)==2:
            fig,axes=plt.subplots(1,2,figsize=(20,6))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(30, 6))
    else: 
        if len(sources)==2:
            fig, axes = plt.subplots(2, 1, figsize=(6, 20))

        else:
            fig, axes = plt.subplots(3, 1, figsize=(9, 20))
   # sources = ['gtab', 'wikimedia', 'mediacloud']
    
    # Average per ban group (ban_group_id) before averaging per seq, so that entities with many bans don't count more than once
    # dfm = df.groupby(['source','ban_group_id','seq'])['value_norm'].mean().reset_index()
    
    # Alternative: take only first ban per ban group (ban_group_id) before averaging per seq, so that entities with many bans don't count more than once
    dfm = df.groupby(['source','ban_group_id','seq'])['value_norm'].first().reset_index()
    
    # Plot once per source
    for source, axis in zip(sources, axes):

        color = color_dict[source][which_color]

        # Compute average for each seq, together with std and count for plotting the CI
        stats = dfm[dfm.source == source].groupby(['seq'])['value_norm'].agg(['mean', 'count', 'std'])            

        if plot_ci:
            stats = compute_confidence_interval(stats)

        if source in ['mediacloud', 'wikimedia']:
            stats_roll = stats.rolling(7, center=True).mean()
            axis.plot(stats_roll['mean'], color=color)
            if plot_ci: axis.fill_between(stats_roll.index, stats_roll.ci95_hi, stats_roll.ci95_lo, alpha=.1, color=color)
        else:
            xs = stats.index
            ys = stats['mean']
            if steps:
                axis.plot(xs, ys, color=color, ds='steps-post')
                if plot_ci: axis.fill_between(stats.index, stats.ci95_hi, stats.ci95_lo, alpha=.1, color=color, step='post')
            else:
                axis.plot(xs, ys, color=color, ds='default')
                if plot_ci: axis.fill_between(stats.index, stats.ci95_hi, stats.ci95_lo, alpha=.1, color=color)
    
    # Plot decorations
    for source, axis in zip(sources, axes):
        
        # Correct x axis ticks
        if source == 'gtab':
            axis.set_xticks(np.arange(-6,6,1))
            xticklabels = ['+'+str(xtl+1) if xtl >= 0 else str(xtl) for xtl in axis.get_xticks()]
            axis.set_xticklabels(xticklabels)
        else:            
            axis.set_xticks(axis.get_xticks()[1:-1])
            xticklabels = ['+'+str(int(xtl)) if xtl > 0 else str(int(xtl)) for xtl in axis.get_xticks()]
            axis.set_xticklabels(xticklabels)

        # Vertical line for ban date
        pos = -0.5 if source == 'gtab' else 0
        axis.axvline(pos, color='tab:grey', ls='--')

        # Set title and axis labels
        axis.set_title(source)
        if source == 'gtab': axis.set_xlabel('months before or after the ban')
        else: axis.set_xlabel('days before or after the ban')
        axis.set_ylabel('average normalized value')

        # Trim edges
        # if source == 'gtab': axis.set_xlim(-6,6)
        # else: axis.set_xlim(-180,180)

        # Add annotation
        if source in ['mediacloud', 'wikimedia']:
            plt.text(.02, .97, '* 7-day-rolling averaged', ha='left', va='top', transform=axis.transAxes, style='italic')

        # Set hardcoded ylims
        if same_y:
            if source == 'gtab': axis.set_ylim(0.15, 0.5)
            if source == 'wikimedia': axis.set_ylim(0.0, 0.35)
            if source == 'mediacloud': axis.set_ylim(0.1, 0.45)
            # axis.set_ylim(0,0.5)

    # Add main title
    if title is not None:
        if horizontal: plt.suptitle(title, fontsize=15)
        else: plt.suptitle(title, fontsize=15, y=0.915)
    plt.savefig("plot1")
    return fig 

def compute_confidence_interval(stats):
    ci95_hi = []
    ci95_lo = []
    for i in stats.index:
        m, c, s = stats.loc[i]
        ci95_hi.append(m + 1.96*s/math.sqrt(c))
        ci95_lo.append(m - 1.96*s/math.sqrt(c))
        
    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo
    
    return stats


#Avg_cat_plot code

def avg_cat_plot(df, sources,cat_feat, steps=True, plot_ci=True, same_y=False, savefig=False, savename=None, title=None, horizontal=True, labels=[], color_dict=COLOR_DICT):
    ''' Plot 'categorical' average plot, i.e. one line per source and category of the cat_feat feature'''
    
    if horizontal: 
        if len(sources)==2:
            fig,axes=plt.subplots(1,2,figsize=(20,6))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(30, 6))
    else: 
        if len(sources)==2:
            fig, axes = plt.subplots(2, 1, figsize=(6, 20))

        else:
            fig, axes = plt.subplots(3, 1, figsize=(9, 20))
    #sources = ['gtab', 'wikimedia', 'mediacloud']
    
    df = df.groupby(['source','ban_group_id','seq',cat_feat])['value_norm'].mean().reset_index()
    
    # Plot once per category and per source
    for i, (cat, cat_group) in enumerate(df.groupby(cat_feat)):

        for source, axis in zip(sources, axes):
            
            color = color_dict[source][i]

            # Compute average for each seq, together with std and count for plotting the CI
            # stats = df[df.source == source].groupby(['seq'])['value_norm'].agg(['mean', 'count', 'std'])
            stats = cat_group[cat_group.source == source].groupby(['seq'])['value_norm'].agg(['mean', 'count', 'std'])            
            
            if plot_ci:
                stats = compute_confidence_interval(stats)
            
            label = labels[i] if labels else f"{cat_feat}:{cat}"
            if source in ['mediacloud', 'wikimedia']:
                stats_roll = stats.rolling(7, center=True).mean()
                axis.plot(stats_roll['mean'], label=label, color=color)
                if plot_ci: axis.fill_between(stats_roll.index, stats_roll.ci95_hi, stats_roll.ci95_lo, alpha=.1, color=color)
            else:
                xs = stats.index
                ys = stats['mean']
                if steps:
                    axis.plot(xs, ys, label=label, color=color, ds='steps-post')
                    if plot_ci: axis.fill_between(stats.index, stats.ci95_hi, stats.ci95_lo, alpha=.1, color=color, step='post')
                else:
                    axis.plot(xs, ys, label=label, color=color, ds='default')
                    if plot_ci: axis.fill_between(stats.index, stats.ci95_hi, stats.ci95_lo, alpha=.1, color=color)
    
    # Plot decorations
    for source, axis in zip(sources, axes):
        
        # Correct x axis ticks
        if source == 'gtab':
            axis.set_xticks(np.arange(-6,6,1))
            xticklabels = ['+'+str(xtl+1) if xtl >= 0 else str(xtl) for xtl in axis.get_xticks()]
            axis.set_xticklabels(xticklabels)
        else:            
            axis.set_xticks(axis.get_xticks()[1:-1])
            xticklabels = ['+'+str(int(xtl)) if xtl > 0 else str(int(xtl)) for xtl in axis.get_xticks()]
            axis.set_xticklabels(xticklabels)

        # Vertical line for ban date
        pos = -0.5 if source == 'gtab' else 0
        axis.axvline(pos, color='tab:grey', ls='--')

        # Set title and axis labels
        axis.set_title(source)
        if source == 'gtab': axis.set_xlabel('months before or after the ban')
        else: axis.set_xlabel('days before or after the ban')
        axis.set_ylabel('average normalized value')

        # Add legend
        axis.legend()

        # Trim edges
        # if source == 'gtab': axis.set_xlim(-6,6)
        # else: axis.set_xlim(-180,180)

        # Add annotation
        if source in ['mediacloud', 'wikimedia']:
            plt.text(.02, .97, '* 7-day-rolling averaged', ha='left', va='top', transform=axis.transAxes, style='italic')

        # Set hardcoded ylims
        if same_y:
            if source == 'gtab': axis.set_ylim(0.15, 0.5)
            if source == 'wikimedia': axis.set_ylim(0.0, 0.35)
            if source == 'mediacloud': axis.set_ylim(0.1, 0.45)
            # axis.set_ylim(0,0.5)

    # Add main title
    if title is not None:
        if horizontal: plt.suptitle(title, fontsize=15)
        else: plt.suptitle(title, fontsize=15, y=0.915)

    plt.savefig("plot3")
    return fig 

#avg_cat plot for 1 source
def avg_cat_plot1(df, sources,cat_feat, steps=True, plot_ci=True, same_y=False, savefig=False, savename=None, title=None, horizontal=True, labels=[], color_dict=COLOR_DICT):
    ''' Plot 'categorical' average plot, i.e. one line per source and category of the cat_feat feature'''
    if horizontal: fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    else: fig, axes = plt.subplots(1, 1, figsize=(3, 10))

    #sources = ['gtab', 'wikimedia', 'mediacloud']
    
    df = df.groupby(['source','ban_group_id','seq',cat_feat])['value_norm'].mean().reset_index()
    source=sources
    axis=axes

    # Plot once per category and per source
    for i, (cat, cat_group) in enumerate(df.groupby(cat_feat)):
        

        color = color_dict[source][i]

            # Compute average for each seq, together with std and count for plotting the CI
            # stats = df[df.source == source].groupby(['seq'])['value_norm'].agg(['mean', 'count', 'std'])
        stats = cat_group[cat_group.source == source].groupby(['seq'])['value_norm'].agg(['mean', 'count', 'std'])            
            
        if plot_ci:
            stats = compute_confidence_interval(stats)
            
        label = labels[i] if labels else f"{cat_feat}:{cat}"
        if source in ['mediacloud', 'wikimedia']:
            stats_roll = stats.rolling(7, center=True).mean()
            axis.plot(stats_roll['mean'], label=label, color=color)
            if plot_ci: axis.fill_between(stats_roll.index, stats_roll.ci95_hi, stats_roll.ci95_lo, alpha=.1, color=color)
        else:
            xs = stats.index
            ys = stats['mean']
            if steps:
                axis.plot(xs, ys, label=label, color=color, ds='steps-post')
                if plot_ci: axis.fill_between(stats.index, stats.ci95_hi, stats.ci95_lo, alpha=.1, color=color, step='post')
            else:
                axis.plot(xs, ys, label=label, color=color, ds='default')
                if plot_ci: axis.fill_between(stats.index, stats.ci95_hi, stats.ci95_lo, alpha=.1, color=color)
    
    # Plot decorations

        # Correct x axis ticks
    if source == 'gtab':
        axis.set_xticks(np.arange(-6,6,1))
        xticklabels = ['+'+str(xtl+1) if xtl >= 0 else str(xtl) for xtl in axis.get_xticks()]
        axis.set_xticklabels(xticklabels)
    else:            
        axis.set_xticks(axis.get_xticks()[1:-1])
        xticklabels = ['+'+str(int(xtl)) if xtl > 0 else str(int(xtl)) for xtl in axis.get_xticks()]
        axis.set_xticklabels(xticklabels)

        # Vertical line for ban date
    pos = -0.5 if source == 'gtab' else 0
    axis.axvline(pos, color='tab:grey', ls='--')

        # Set title and axis labels
    axis.set_title(source)
    if source == 'gtab': axis.set_xlabel('months before or after the ban')
    else: axis.set_xlabel('days before or after the ban')
    axis.set_ylabel('average normalized value')

        # Add legend
    axis.legend()

        # Trim edges
        # if source == 'gtab': axis.set_xlim(-6,6)
        # else: axis.set_xlim(-180,180)

        # Add annotation
    if source in ['mediacloud', 'wikimedia']:
        plt.text(.02, .97, '* 7-day-rolling averaged', ha='left', va='top', transform=axis.transAxes, style='italic')

        # Set hardcoded ylims
    if same_y:
        if source == 'gtab': axis.set_ylim(0.15, 0.5)
        if source == 'wikimedia': axis.set_ylim(0.0, 0.35)
        if source == 'mediacloud': axis.set_ylim(0.1, 0.45)
            # axis.set_ylim(0,0.5)

    # Add main title
    if title is not None:
        if horizontal: plt.suptitle(title, fontsize=15)
        else: plt.suptitle(title, fontsize=15, y=0.915)

    plt.savefig("plot4")
    return fig 



if __name__ == '__main__':
    app.run_server(debug=True, port=9000)