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
| AGN     |       0.974948 |          0.969282 |   0.972107 |
| CV      |       0.560241 |          0.607843 |   0.583072 |
| HMXB    |       0.898396 |          0.915531 |   0.906883 |
| LMXB    |       0.811189 |          0.943089 |   0.87218  |
| PULSAR  |       0.544554 |          0.466102 |   0.502283 |
| STAR    |       0.95448  |          0.95964  |   0.957053 |
| ULX     |       0.725118 |          0.689189 |   0.706697 |
| YSO     |       0.944299 |          0.928144 |   0.936152 |
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

|        |   class |
|:-------|--------:|
| STAR   |    7745 |
| AGN    |    7371 |
| YSO    |    6069 |
| HMXB   |    1675 |
| ULX    |     760 |
| CV     |     555 |
| PULSAR |     419 |
| LMXB   |     151 |
</td>
<td><img src = '../plots/higher_models/lbg_unid_prob.jpg'></td>
</tr></table>







# GB
## Confusion matrix


<table><tr>
<td>
<img src='../plots/higher_models/GB.jpg'>
</td>
<td><img src='../plots/higher_models/GB_prob.jpg'></td>
</tr></table>



## Scores 


<table><tr>
<td>

| class   |   recall_score |   precision_score |   f1_score |
|:--------|---------------:|------------------:|-----------:|
| AGN     |       0.896451 |          0.978578 |   0.935716 |
| CV      |       0.584337 |          0.40249  |   0.476658 |
| HMXB    |       0.874332 |          0.841699 |   0.857705 |
| LMXB    |       0.804196 |          0.793103 |   0.798611 |
| PULSAR  |       0.594059 |          0.28436  |   0.384615 |
| STAR    |       0.912545 |          0.972127 |   0.941394 |
| ULX     |       0.744076 |          0.468657 |   0.575092 |
| YSO     |       0.935596 |          0.910246 |   0.922747 |
</td>
<td>
    <ul>
        <li> Accuracy - 0.889
        <li> Precision - 0.91
        <li> Recall - 0.89
        <li> f1 score - 0.90
    </ul>
</td>
</tr></table>


## On unidentified dataset 

> Number of sources : 39893

<table><tr>
<td>

|   Class     |   Number of sources |
|:-------|--------:|
| STAR   |    4921 |
| AGN    |    3624 |
| YSO    |    2611 |
| HMXB   |     574 |
| LMXB   |      88 |
| PULSAR |      48 |
| CV     |      48 |
| ULX    |      15 |
</td>
<td><img src = '../plots/higher_models/GB_unid_prob.jpg'></td>
</tr></table>









# RF Moode Tuned
## Confusion matrix

<table><tr>
<td>
<img src='../plots/higher_models/RF_mod_tuned.jpg'>
</td>
<td><img src='../plots/higher_models/RF_mod_tuned_prob.jpg'></td>
</tr></table>

## Scores 

<table><tr>
<td>

| class   |   recall_score |   precision_score |   f1_score |
|:--------|---------------:|------------------:|-----------:|
| AGN     |       0.931942 |          0.966234 |   0.948778 |
| CV      |       0.506024 |          0.48     |   0.492669 |
| HMXB    |       0.824866 |          0.801299 |   0.812912 |
| LMXB    |       0.811189 |          0.852941 |   0.831541 |
| PULSAR  |       0.50495  |          0.31677  |   0.389313 |
| STAR    |       0.937993 |          0.96072  |   0.94922  |
| ULX     |       0.630332 |          0.507634 |   0.562368 |
| YSO     |       0.923412 |          0.91073  |   0.917027 |
</td>
<td>
    <ul>
        <li> Accuracy - 0.89
        <li> Precision - 0.90
        <li> Recall - 0.89
        <li> f1 score - 0.90
    </ul>
</td>
</tr></table>


## On unidentified dataset 

> Number of sources : 39893

<table><tr>
<td>

|   Class     |   Number of sources |
|:-------|--------:|
| STAR |    1095 |
| HMXB |     272 |
| CV   |       7 |
| LMXB |       5 |
| YSO  |       1 |
| AGN  |       1 |
</td>
<td><img src = '../plots/higher_models/RF_mod_tuned_unid_prob.jpg'></td>
</tr></table>




# RF rfimp Tuned
## Confusion matrix

<table><tr>
<td>
<img src='../plots/higher_models/RF_rfimp_tuned.jpg'>
</td>
<td><img src='../plots/higher_models/RF_rfimp_tuned_prob.jpg'></td>
</tr></table>

## Scores 

<table><tr>
<td>

| class   |   recall_score |   precision_score |   f1_score |
|:--------|---------------:|------------------:|-----------:|
| AGN     |       0.924008 |          0.953879 |   0.938706 |
| CV      |       0.518072 |          0.457447 |   0.485876 |
| HMXB    |       0.791444 |          0.766839 |   0.778947 |
| LMXB    |       0.776224 |          0.925    |   0.844106 |
| PULSAR  |       0.415842 |          0.330709 |   0.368421 |
| STAR    |       0.937276 |          0.952988 |   0.945067 |
| ULX     |       0.540284 |          0.445312 |   0.488223 |
| YSO     |       0.926023 |          0.904762 |   0.915269 |
</td>
<td>
    <ul>
        <li> Accuracy - 0.89
        <li> Precision - 0.89
        <li> Recall - 0.89
        <li> f1 score - 0.89
    </ul>
</td>
</tr></table>


## On unidentified dataset 

> Number of sources : 39893

<table><tr>
<td>

|   Class     |   Number of sources |
|:-------|--------:|
| AGN  |    3310 |
| STAR |     118 |
</td>
<td><img src = '../plots/higher_models/RF_rfimp_tuned_unid_prob.jpg'></td>
</tr></table>


# LightGBM classification Results 
