# MUWCLASS details

## Source variability

All sources were considered whether theu are variable or not
*  Variable sources - 1075
* Non-variable sources - 1887 
<table>
    <tr>
        <th> CLASS </th>
        <th> Non-var sources</th>
        <th> var sources </th>
        <th> All sources </th>
    </tr>
    <tr>
        <td>AGN</td><td>1133</td><td>257</td><td>1390</td>
    </tr>
    <tr>
        <td>CV</td><td>24</td><td>20</td><td>44</td>
    </tr>
    <tr>
        <td>HM-STAR</td><td>82</td><td>36</td><td>118</td>
    </tr>
    <tr>
        <td>YSO</td><td>366</td><td>658</td><td>1024</td>
    </tr>
    <tr>
        <td>HMXB</td><td>14</td><td>12</td><td>26</td>
    </tr>
    <tr>
        <td>LMXB</td><td>37</td><td>28</td><td>65</td>
    </tr>
    <tr>
        <td>LM-STAR</td><td>154</td><td>54</td><td>208</td>
    </tr>
    <tr>
        <td>NS</td><td>77</td><td>10</td><td>87</td>
    </tr>
    <tr>
        <td><b>Total</b></td>
        <td></td>
        <td></td>
        <td>2962</td>
    </tr>
    <hr>
</table>

---
## MW data download

In the pipeline they are downloading data using 'astroquery'

--- 

# Effect of Features 
## Original

# Result 
Check if their result is on training data or test data. Find out their train-test split (email sent..)

They are not doing train-test split, rather Leave-One-Out cross validation is used. (computationally very expensive). That's how they have geerated Confusion matrix on training dataset.