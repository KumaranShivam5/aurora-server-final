# Issues 

*   Number of sources dispropotinate for the case of Random Forest
*   Class balance $\gamma$ factor tuning 
*   Why GB is has very high preoformance gain compared to random forest ?
*   How LightGBM is handling Missing values.

## Number of sources Issue

> Probability threshold should not be chosen same for different models\
> Different models have different confidance levels.

<table>
<tr><th colspan=3>Trainind data probability distribution</th></tr>
<tr>
    <td><img src='../temp_res_comp/unid_prob_dist/RF_mode.jpg'></td>
    <td><img src='../temp_res_comp/unid_prob_dist/GB_mode.jpg'></td>
    <td><img src='../temp_res_comp/unid_prob_dist/lightGBM_none.jpg'></td>
</tr>
</table>