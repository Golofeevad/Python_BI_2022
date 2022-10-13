# Homework 2

## _fastq files filter_

This programm contains `main` function that filteres fastq file entries based on their multiple properties

### Agruments:

- input_fastq - path to "fastq" files to be processed
- output_file_prefix - prefix for output file. Output will be written to `<output_file_prefix>_passed.fastq`. If save_filtered flag is true, 
rejected entries will be written to `<output_file_prefix>_failed.fastq`
- gc_bounds - allowed interval of guanine + cytosine over total amount of nucliotides
- length_bounds - allowed length interval of read
- quality_threshold - minimum mean quality of read
- save_filtered - flag, if true - writes rejected entries to `<output_file_prefix>_passed.fastq`
