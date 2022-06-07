import pandas as pd
import datarobot as dr
import seaborn as sns
import matplotlib.pyplot as plt
from datarobot_bp_workshop import Visualize


def _matplotlib_pair_histogram(labels, counts, target_avgs, bin_count, ax1, feature, target_feature_name):
    """
    Helper function to configure feature histogram.
    """
    if feature.feature_type not in ['Categorical', 'Text']:
        xtick_labels = ["{:,}".format(round(float(l), 1)) for l in labels]
    else:
        xtick_labels = labels
    ax1.set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax1.set_ylabel(feature.name, color='cornflowerblue')
    ax1.bar(labels, counts, color='cornflowerblue')
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(target_feature_name, color='darkorange')
    ax1.set_facecolor('midnightblue')
    ax2.plot(labels, target_avgs, marker='o', lw=1, color='darkorange')
    title = 'Histogram for {} ({} bins)'.format(feature.name, bin_count)
    ax1.set_title(title)
    ax1.grid(False)
    ax2.grid(False)
    
    
def draw_feature_histogram(project, feature_name, target_feature_name, bin_count):
    """
    Retrieve downsampled histogram data from server based on desired bin count.
    """
    feature = dr.Feature.get(project.id, feature_name)
    data = feature.get_histogram(bin_count).plot
    labels = [row['label'] for row in data]
    counts = [row['count'] for row in data]
    target_averages = [row['target'] for row in data]
    f, axarr = plt.subplots()
    f.set_size_inches((10, 4))
    _matplotlib_pair_histogram(labels, counts, target_averages, bin_count, axarr, feature, target_feature_name)
    

def get_top_of_leaderboard(project, metric='AUC', verbose=True):
    """ 
    A helper method to assemble a dataframe with leaderboard results and print a summary.
    """
    # list of metrics that get better as their value increases
    desc_metric_list = ['AUC', 'Area Under PR Curve', 'Gini Norm', 'Kolmogorov-Smirnov', 'Max MCC', 'Rate@Top5%',
                        'Rate@Top10%', 'Rate@TopTenth%', 'R Squared', 'FVE Gamma', 'FVE Poisson', 'FVE Tweedie',
                        'Accuracy', 'Balanced Accuracy', 'FVE Multinomial', 'FVE Binomial'
                        ]
    asc_flag = False if metric in desc_metric_list else True
    
    leaderboard = []
    for m in project.get_models():
        leaderboard.append(
            [m.blueprint_id, m.featurelist.id, m.id, m.model_type, m.sample_pct, m.metrics[metric]['validation'],
             m.metrics[metric]['crossValidation']])
    leaderboard_df = pd.DataFrame(
        columns=['bp_id', 'featurelist', 'model_id', 'model', 'pct', f'validation_{metric}', f'cross_validation_{metric}'],
        data=leaderboard)
    leaderboard_top = leaderboard_df[leaderboard_df['pct'] == 64].sort_values(by=f'cross_validation_{metric}',
                                                                              ascending=asc_flag).head().reset_index(
        drop=True)

    if verbose:
        # Print a leaderboard summary:
        print("Unique blueprints tested: " + str(len(leaderboard_df['bp_id'].unique())))
        print("Feature lists tested: " + str(len(leaderboard_df['featurelist'].unique())))
        print("Models trained: " + str(len(leaderboard_df)))
        print("Blueprints in the project repository: " + str(len(project.get_blueprints())))

        # Print key info for top models, sorted by accuracy on validation data:
        print("\n\nTop models in the leaderboard:")
        display(leaderboard_top.drop(columns=['bp_id', 'featurelist'], inplace=False))

        # Show blueprints of top models:
        for index, m in leaderboard_top.iterrows():
            Visualize.show_dr_blueprint(dr.Blueprint.get(project.id, m['bp_id']))

    return leaderboard_top


def plot_feature_impact(datarobot_model, title=None):
    """This function plots feature impact
    Input:
        datarobot_model: <Datarobot Model object>
        title : <string> --> title of graph
    """
    # Get feature impact
    feature_impacts = datarobot_model.get_or_request_feature_impact()

    # Sort feature impact based on normalised impact
    feature_impacts.sort(key=lambda x: x["impactNormalized"], reverse=True)

    fi_df = pd.DataFrame(feature_impacts)  # Save feature impact in pandas dataframe
    fig, ax = plt.subplots(figsize=(14, 5))
    b = sns.barplot(
        x="featureName", y="impactNormalized", data=fi_df[0:5], color="navy"
    )
    b.axes.set_xlabel("")
    b.axes.set_ylabel("Impact Normalized", fontsize=14)
    b.axes.set_title("Feature Impact" if not title else title, fontsize=16)