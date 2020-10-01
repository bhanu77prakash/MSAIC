# MSAIC
## Main files
<ol>
  <li> 
    train.py data_preprocess.py
  </li>
</ol>

## Running
<ol>
  <li>
    python train.py <b><i>train_file_name(including .tsv)</i></b>
  </li>
</ol>

## File ordering

Place the training data in <b><i>data</i></b> folder in the current directory<br>
Place the <b><i>test_file</i></b> in <b><i>data</i></b> directory<br>
In case preprocessed files obtained after training for the first time are available, put them in the data directory so that model loads them and don't preprocess again.

## Output
Trained model, preprocessed files and test file output named as <b><i>answer.tsv</i></b>


Updated some bugs
