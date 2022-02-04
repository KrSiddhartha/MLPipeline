import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


class modeleval():
    
    def confusion_mat_plt(y_test, y_pred, cat_cnt, figs=(10, 10), cat_names=None):
        cf_matrix = metrics.confusion_matrix(y_test, y_pred)
        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cf_matrix.flatten()/np.sum(cf_matrix)]

        labels = [f"{v1}\n{v2}\n" for v1, v2 in
                  zip(group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(cat_cnt,cat_cnt)

        fig, ax = plt.subplots(figsize=figs)
        ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
        ax.set_title('Confusion Matrix\n\n');
        ax.set_xlabel('\nPredicted')
        ax.set_ylabel('Actual');
        if cat_names:
            cat_names = [' '.join(j) for j in [i.split(' ')[:2] for i in cat_names]]
            ax.xaxis.set_ticklabels(cat_names)
            ax.yaxis.set_ticklabels(cat_names)

        plt.show()
        
        return fig
    
    def classification_report_cust(ytrain, ypred, cat_names=None):
        report = metrics.classification_report(ytrain, ypred,
                                               target_names=cat_names)
        report_data = []
        lines = report.split('\n')
        for line in lines[2:]:
            if line!='':
                row = {}
                row_data = line.strip().split('      ')
                row_data = [np.nan if i=='' else i for i in row_data ]
                row['class'] = row_data[0]
                row['precision'] = float(row_data[1])
                row['recall'] = float(row_data[2])
                row['f1_score'] = float(row_data[3])
                row['support'] = float(row_data[4])
                report_data.append(row)

        return pd.DataFrame.from_dict(report_data)
