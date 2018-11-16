def missing_values_overview(df):
    """Provides an overview about the missing values and zeros of each column of passed DataFrame df"""

    import matplotlib.pyplot as plt
    import numpy as np

    try:
        counts = df.columns.value_counts()
        duplicates = counts[counts > 1]
        if len(duplicates) > 0:
            raise IndexError(
                'DataFrame must not contain duplicate column names:')
    except IndexError as e:
        print('Error: ', e, '\n', duplicates)
        return
        
    (rows, cols) = df.shape
    non_nans = []
    nans = []
    zeros = []

    for col in df.columns:
        col_zeros = np.sum(df[col] == 0)
        col_non_nans = df[col].count() - col_zeros
        col_nans = rows - col_non_nans - col_zeros
        non_nans = non_nans + [col_non_nans, ]
        nans = nans + [col_nans, ]
        zeros = zeros + [col_zeros, ]
        
    nans_zeros = [sum(x) for x in zip(nans, zeros)]   # element wise addition

    plt.figure(figsize=(10, 5))
    ind = np.arange(cols)   # location of the bars
    width = 0.9
    p1 = plt.bar(ind, nans, width, color='#d62728')
    p2 = plt.bar(ind, zeros, width, color='#FF7F50', bottom=nans)
    p3 = plt.bar(ind, non_nans, width, bottom=nans_zeros)
    
    plt.title('non-Null, Zero and NaN values per column')
    plt.ylabel('Rows')
    plt.xlabel('Columns')
    plt.xticks(ind, df.columns, rotation='vertical')
    plt.legend((p1[0], p2[0], p3[0]), ('NaN', 'Zero', 'non-NaN'))
    
    
def find_max_value_counts(df):
    """Returns the number of ocurrences of that value wich ocurrs most across all columns.
    Counts only ocurrences within a column"""

    max_count = 0
    for idx, feature in enumerate(df.columns):
        col_max = df[feature].value_counts().iloc[0]
        if col_max > max_count:
            max_count = col_max
    return max_count


def plot_value_counts(df, grid_cols=3):
    """Plots value_count of each column of the given Dataframe on a grid.
    grid_cols indicates the number of columns to be displayed in each row.

    The display works best for grid_cols=3 or 4
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    cols = len(df.columns)
    grid_rows = int(np.ceil(cols / grid_cols))
    max_ticks = 15   # max number of xtick labels per subplot

    # adjust height of fig depending on overall number of rows to be plotted
    figsize_h = 15 * (np.ceil(cols / grid_cols) / grid_cols)
    ymax = find_max_value_counts(df)

    fig, axes = plt.subplots(
        nrows=grid_rows, ncols=grid_cols, figsize=(15, figsize_h))
    for idx, feature in enumerate(df.columns):
        val_counts = df[feature].value_counts()

        ax = val_counts.plot(kind='bar', title=feature, sharey='col', ylim=(0, ymax,),
                             ax=axes[idx // grid_cols, idx % grid_cols])
        # set all labels to invisible
        for label in ax.get_xticklabels():
            label.set_visible(False)

        # add information how many unique values per column
        unique_values = len(val_counts)
        uv_label = str(unique_values) + ' u.v.'
        ax.text(0, ymax - ymax * 0.1, uv_label, fontsize=22)

        len_labels = len(ax.get_xticklabels())
        stepper = int(np.ceil(len_labels / max_ticks))

        # set selected labels to visible and truncate long labels
        # for better understanding, read
        # https://stackoverflow.com/questions/11244514/modify-tick-label-text
        labels = ax.get_xticks().tolist()
        c = 0
        for label in ax.get_xticklabels()[::stepper]:
            text = label.get_text()[:12]    # max characters per tick label
            label.set_visible(True)
            labels[c] = text
            c += stepper
        ax.set_xticklabels(labels)

    plt.xlabel('u.v. = unique values')

    # fig.tight_layout()
    #save_fig("col_val_counts_plot")
    

def bar_chart_autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height.
    
    Apparently, user ImportanceOfBeingErnest provided a better method for autolabel
    https://stackoverflow.com/questions/47805767/why-does-pyplot-plot-create-an-additional-rectangle-with-width-1-height-1/47812553#47812553
    """
    
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()  
            
        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        if p_height > 0.95: # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width()/2., label_position,
                '%d' % int(height),
                ha='center', va='bottom')