# PowerShell equivalent of run.sh
$env:PYTHONPATH = Get-Location


$models = 'v3'
$whos = "everyone"
$resultsDatasetPathJson = 'long_answer_results_everyone'
$setting = "long-form-answering"
$num_samples = 'all'
$restart_from = '222'
$restart_path = "temp/['v3']_temp_results.csv"

# Run the Python script with the arguments
python predict_answers.py `
    --models $models `
    --setting $setting `
    --whos $whos `
    --results_dataset_path $resultsDatasetPathJson `
    --num_samples $num_samples `
    --restart_from $restart_from `
    --restart_path $restart_path


