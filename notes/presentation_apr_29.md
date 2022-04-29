# Recap

\>

# Issues 

*   Number of sources dispropotinate for the case of Random Forest
*   Class balance $\gamma$ factor tuning 
*   Why GB is has very high preoformance gain compared to random forest ?
*   How LightGBM is handling Missing values.

# Number of sources Issue

> Probability threshold should not be chosen same for different models\
> Different models have different confidance levels.

<table>
<tr><th>RF</th><th>GB</th><th>LightGBM</th></tr>
<tr><td colspan=3>Unidentified sources probability distribution - classwise</td></tr>
<tr>
    <td><img src='../temp_res_comp/unid_prob_dist/RF_mode.jpg'></td>
    <td><img src='../temp_res_comp/unid_prob_dist/GB_mode.jpg'></td>
    <td><img src='../temp_res_comp/unid_prob_dist/lightGBM_none.jpg'></td>
</tr>
<tr><td colspan=3>Unidentified sources probability distribution</td></tr>
<tr>
    <td><img src='../temp_res_comp/unid_prob_dist/RF_mode.jpg'></td>
    <td><img src='../temp_res_comp/unid_prob_dist/GB_mode.jpg'></td>
    <td><img src='../temp_res_comp/unid_prob_dist/lightGBM_none.jpg'></td>
</tr>
<tr><td colspan=3>Unidentified sources classification numbers</td></tr>
<tr>
<td>   

| class   |   Argmax |   0.5 |   0.9 |   MP > 0.98 |
|:--------|---------:|------:|------:|------------:|
| AGN     |    66390 | 54354 | 37909 |       25279 |
| PULSAR  |    39429 | 29821 | 19247 |       11727 |
| STAR    |    34400 | 26788 | 19500 |       13332 |
| ULX     |    33199 | 22497 | 10975 |        4428 |
| HMXB    |    27362 | 19093 | 11858 |        7095 |
| CV      |    17929 |  8162 |  3099 |        1283 |
| YSO     |    17889 | 14208 |  8421 |        5111 |
| LMXB    |      578 |   345 |   169 |         151 |

</td>
<td>

| class   |   Argmax |   0.5 |   0.9 |   MP > 0.98 |
|:--------|---------:|------:|------:|------------:|
| STAR    |    11844 |  8965 |  4911 |        2300 |
| AGN     |    11674 |  8391 |  3645 |        1307 |
| YSO     |     8934 |  8105 |  2591 |         281 |
| HMXB    |     4364 |  2268 |   569 |          99 |
| ULX     |     1321 |   491 |    15 |         nan |
| CV      |      948 |   488 |    48 |           6 |
| PULSAR  |      594 |   162 |    17 |           4 |
| LMXB    |      214 |   161 |    88 |          27 |

</td>
</tr>
</table>

# How Gradient Boost does the magic to improve RF


[Simple Video Explanation](https://www.youtube.com/watch?v=TyvYZ26alZs)

It is because of the way the trees are build in both the cases.

In RF Decision trees are built which are independednt of each other and the outputs are combined in parallel

For Gradient Boosted Trees, the trees are built sequentially. With a given loss function the loss at each newly constructed tree is built. The gradient of this loss function at $m-1^{th}$ tree is used to construct a new tree. This new tree is combined to the previous trees after multiplying ith a weight factor called Learning rate $\eta$, which generally varies from 0 to 1 

> Intitively new tree is built to minimize the error from the previous tree.

For clasification, generally categorical cross entropy is chosen as the loss function.

$Loss (p,q) = \sum p(x)\times Log(q(x))$

$F(2) = F(1)\times Second\ Tree$

$Second\ Tree = -\frac{\partial L}{\partial F(1)} = - \frac{\partial Loss}{\partial Previous\ Model \ Output}$

In general 

Output at the end of tree m is 

$F(m) = F(m-1)+\eta \times -\frac{\partial L}{\partial F(m-1)}$

Compared to Random Forest, Gradient Boosted trees can learn more complex decision boundaries.


# How LightGBM improves GB

*   Creates Histogram and uses the binned values
*   A decision on which side to go for a given exmple, in the case of missing feature is decided by the side of maximum gain.
*   More flexible hyper parameter tuning
*   More details in the paper, still trying to understand.

# $\gamma$ tuning for LightGBM class imbalance