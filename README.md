# offensive-language
Aggressive language detection (ALD), detecting the abusive and offensive
language in texts, is one the crucial applications in NLP community. Most
existing works treat

# Dataset
The [TRAC](https://sites.google.com/view/trac1/home) dataset is published in a shared task for aggressive language detection. The
sources are from English social media, e.g., Facebook and Twitter, and there are two
corresponding testing sets, i.e., FB and TW. There are three labels for indicating the aggression
degree: covertly aggressive(CAG), non-aggressive(NAG) and overtly aggressive(
OAG)
<div class="datagrid" style="width:500px;">
<table>
<thead><tr><th>Dataset</th><th>CAG</th><th>OAG</th><th>NAG</th><th>Total</th></tr></thead>
<tbody>
<tr><th>Train</th><td>4240</td><td>2708</td><td>5051</td><td>12000</td></tr>
<tr><th>Develop</th><td>1057</td><td>711</td><td>1233</td><td>3000</td></tr>
<tr><th>Facebook</th><td>142</td><td>144</td><td>629</td><td>916</td></tr>
<tr><th>Twitter</th><td>413</td><td>361</td><td>482</td><td>1256</td></tr>
</tbody>
</table>
</div>

# Requirements
python 3.6 with pip, pytorch==1.1

# Quick start
## data pre-process
If we want to use hierarchical neural network to train our model, we should pre-process the dataset. 
To pre-process the data run:
```angular2html
python data_processor/util.py
```
