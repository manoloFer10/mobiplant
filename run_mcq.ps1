$env:PYTHONPATH = Get-Location


$models = 'claude'
$whos = 'everyone'
$resultsDatasetPathJson = 'results_shuffled2'
$setting = 'mcq-answering'
$style = 'CoT'
$num_samples = 'all'
$data_path = 'data\questionsMCQ_named_shuffled2.json'
# $restart_from = '352'
# $restart_path = "results\mcq-answering_['claude']_results.csv"

# Run the Python script with the arguments
python predict_answers.py `
    --data_path $data_path `
    --models $models `
    --setting $setting `
    --whos $whos `
    --results_dataset_path $resultsDatasetPathJson `
    --evaluation_style $style `
    --num_samples $num_samples `
#    --restart_from $restart_from `
#    --restart_path $restart_path
