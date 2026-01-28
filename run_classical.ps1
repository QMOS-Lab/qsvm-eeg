$sample_sizes = @(200, 500, 1000, 2000, 3000, 6000, 8000)

$patient_configs = @(
    "48"
    # "411",
    # "48 411"
)

foreach ($patients in $patient_configs) {
    foreach ($n in $sample_sizes) {
        Write-Host "------------------------------------------------"
        Write-Host "Running classical SVR: Patients [$patients] | Samples: $n"
        Write-Host "------------------------------------------------"

        uv run classical.py -p $patients.Split(' ') -n $n
    }
}

Write-Host "All experiments completed!"
