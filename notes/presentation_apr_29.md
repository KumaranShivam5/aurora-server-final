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

# $\gamma$ tuning for LightGBM class imbalance