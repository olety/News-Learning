import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

FILENAME = 'results/res.csv'
FILENAME_IDF = 'results/res_temp.csv'

res_df = pd.DataFrame.from_csv(FILENAME)
res_idf_df = pd.DataFrame.from_csv(FILENAME_IDF)

res_df = pd.concat([res_df, res_idf_df])
res_df = res_df.sort_values(by='score', ascending=True)

colors_dict = {
    'All features': '#9DBBD3',
    'Percentile selector (30%)': '#F8BBC0',
    'Model selector (random forest)': '#0BC4A7',
}
colors = [colors_dict[i] for i in res_df['feature_selector']]

ax = res_df.plot(kind='barh', y='score', x='name', color=colors, legend=False)
ax.set_ylabel('Model')
ax.set_xlabel('Test set accuracy')

legend_squares = []
legend_labels = []

for k, v in colors_dict.items():
    legend_squares.append(Rectangle((0, 0), 1, 1, fc=v))
    legend_labels.append(k)


plt.legend(legend_squares, legend_labels,
           bbox_to_anchor=(0, 1), loc=3, ncol=3, borderaxespad=0.75)
plt.title('Accuracy per option', y=1.075)
plt.tight_layout()
plt.subplots_adjust(left=0.3)
fig = plt.gcf()
fig.set_size_inches(10, 12)
for p in ax.patches:
    b = p.get_bbox()
    val = f'{b.x1 + b.x0:.3f}'
    ax.annotate(val, ((b.x0 + b.x1) - 0.06, (b.y0 + b.y1) / 2 - 0.15))
plt.show()
# plt.savefig('results.png', orientation='landscape', dpi=300)
plt.close()
