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
    python train.py $train_file_name<including .tsv>$
  </li>
  </ol>
## File ordering
Place the training data in $data$ folder in the current directory\\
Place the $test_file$ in $data$ directory\\ 
In case preprocessed files obtained after training for the first time are available, put them in the data directory so that model loads them and don't preprocess again.
## Output
Trained model, preprocessed files and test file output named as $answer.tsv$
