# Benchmarks

Contains input templates (and their prerequisites) for benchmarks. As well as the results of each and every template. The changes are divided in:
* DDL changes with their online equivalents
* Migration strategies

## Starting a Benchmark
Run:
`./start.sh inputs/change_directory`

After the benchmark you can process the results to nice charts using:
`./process_benchmarks.py <change_directory>`.


## Processing all benchmarks
In case no specific `change_directory` is provided to `process_benchmarks.py`, all benchmark results will be processed. The axes of all charts will have the same dimensions and will be based on the global min/max values of all results to keep them comparable.