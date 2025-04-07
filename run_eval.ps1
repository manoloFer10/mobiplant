$env:PYTHONPATH = Get-Location

python eval.py `
    --num_samples "all" `
    --results_dataset "inference.json" `
    --evaluation_style "automatic-metrics" `
    --output_folder "test_eval"