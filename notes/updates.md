# LightGBM-OVA
## Confusion matrix


<table><tr>
<td>
<img src='../plots/higher_models/lgb_ova.jpg'>
</td>
<td><img src='../plots/higher_models/lbg_ova_prob.jpg'></td>
</tr></table>



## Scores 


<table><tr>
<td>

| class   |   recall_score |   precision_score |   f1_score |
|:--------|---------------:|------------------:|-----------:|
| AGN     |       0.939875 |          0.969841 |   0.954623 |
| CV      |       0.596386 |          0.428571 |   0.498741 |
| HMXB    |       0.838235 |          0.891892 |   0.864232 |
| LMXB    |       0.832168 |          0.915385 |   0.871795 |
| PULSAR  |       0.534653 |          0.247706 |   0.338558 |
| STAR    |       0.932616 |          0.965492 |   0.948769 |
| ULX     |       0.744076 |          0.596958 |   0.662447 |
| YSO     |       0.91819  |          0.923818 |   0.920995 |
</td>
<td>
    <ul>
        <li> Accuracy - 
        <li> Precision - 
        <li> Recall - 
    </ul>
</td>
</tr></table>


## On unidentified dataset 

> Number of sources : 39893

<table><tr>
<td>

|Class   |Number of sources|
|:-------|--------:|
| STAR   |    6298 |
| AGN    |    4985 |
| YSO    |    2347 |
| CV     |    1177 |
| HMXB   |    1167 |
| PULSAR |    1133 |
| ULX    |     978 |
| LMXB   |     127 |
</td>
<td><img src = '../plots/higher_models/lbg_ova_unid_prob.jpg'></td>
</tr></table>




# LightGBM-Multiclass
## Confusion matrix


<table><tr>
<td>
<img src='../plots/higher_models/lgb.jpg'>
</td>
<td><img src='../plots/higher_models/lbg_prob.jpg'></td>
</tr></table>



## Scores 


<table><tr>
<td>

| class   |   recall_score |   precision_score |   f1_score |
|:--------|---------------:|------------------:|-----------:|
| AGN     |       0.971608 |          0.969583 |   0.970594 |
| CV      |       0.566265 |          0.630872 |   0.596825 |
| HMXB    |       0.909091 |          0.915209 |   0.91214  |
| LMXB    |       0.79021  |          0.933884 |   0.856061 |
| PULSAR  |       0.475248 |          0.432432 |   0.45283  |
| STAR    |       0.956631 |          0.959382 |   0.958004 |
| ULX     |       0.720379 |          0.703704 |   0.711944 |
| YSO     |       0.950392 |          0.92464  |   0.937339 |
</td>
<td>
    <ul>
        <li> Accuracy - 
        <li> Precision - 
        <li> Recall - 
    </ul>
</td>
</tr></table>


## On unidentified dataset 

> Number of sources : 39893

<table><tr>
<td>

|   Class     |   Number of sources |
|:-------|--------:|
| AGN    |    7567 |
| STAR   |    7196 |
| YSO    |    6074 |
| HMXB   |    1551 |
| ULX    |     754 |
| CV     |     569 |
| PULSAR |     394 |
| LMXB   |     156 |
</td>
<td><img src = '../plots/higher_models/lbg_unid_prob.jpg'></td>
</tr></table>