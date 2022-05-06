import joblib 
import numpy as np

focal = joblib.load('../not_on_git/focal_tune_dict_v2.pkl')

from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm 
#alpha , gamma  , precision , recall , f1 , acc , bal_acc = [] , [] ,[] , [],[],[],[]
recall_grid = []
precision_grid = [] 
mcc_grid = []
f1_grid = []
pulsar_recall = []
pulsar_f1 = []
pulsar_precision = []
for f in tqdm(focal):
    recall_grid.append([f['param']['alpha'] ,f['param']['gamma'] , f['res']["recall"] ])
    precision_grid.append([f['param']['alpha'] ,f['param']['gamma'] , f['res']["precision"] ])
    f1_grid.append([f['param']['alpha'] ,f['param']['gamma'] , f['res']["f1"] ])
    mcc = matthews_corrcoef(f['res']['res_table']['true_class'] ,f['res']['res_table']['true_class'])
    pulsar_recall.append([f['param']['alpha'] ,f['param']['gamma'] , f['res']['class_scores'].loc['PULSAR']['recall_score'] ])
    pulsar_precision.append([f['param']['alpha'] ,f['param']['gamma'] , f['res']['class_scores'].loc['PULSAR']['precision_score'] ])
    pulsar_f1.append([f['param']['alpha'] ,f['param']['gamma'] , f['res']['class_scores'].loc['PULSAR']['f1_score'] ])
    mcc_grid.append([f['param']['alpha'] , f['param']['gamma'] , mcc])

    #alpha.append(f['param']['alpha'])
    #gamma.append(f['param']['alpha'])



from matplotlib import pyplot as plt 
fig = plt.figure(figsize=(24,8))

to_plot = pulsar_precision
x = np.asarray([el[0] for el in to_plot])
y = np.asarray([el[1] for el in to_plot])
z = np.asarray([el[2] for el in to_plot])
ax = fig.add_subplot(131, projection='3d')
ax.plot_trisurf(x,y,z)

to_plot = pulsar_recall
x = np.asarray([el[0] for el in to_plot])
y = np.asarray([el[1] for el in to_plot])
z = np.asarray([el[2] for el in to_plot])
ax = fig.add_subplot(132, projection='3d')
ax.plot_trisurf(x,y,z)

to_plot = pulsar_f1
x = np.asarray([el[0] for el in to_plot])
y = np.asarray([el[1] for el in to_plot])
z = np.asarray([el[2] for el in to_plot])
ax = fig.add_subplot(133, projection='3d')
ax.plot_trisurf(x,y,z)

plt.show()