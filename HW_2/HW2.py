# .fastq filter

def check_conditions(read_str, quality_str, gc_bounds, length_bounds, quality_threshold):
    return gc_check(read_str, gc_bounds) and \
           len_check(read_str, length_bounds) and \
           quality_check(quality_str, quality_threshold)


def gc_check(read_str, interval):
    read_str = list(read_str)
    if type(interval) == int:
        interval = (0, interval)
    gc_percent = 100 * (read_str.count('G') + read_str.count('C')) / len(read_str)
    return interval[0] <= gc_percent <= interval[1]


def len_check(read_str, length_bounds):
    if type(length_bounds) == int:
        length_bounds = (0, length_bounds)
    return length_bounds[0] <= len(read_str) <= length_bounds[1]


def quality_check(quality_str, quality_threshold):
    quality_sum = 0
    for q in quality_str:
        quality_sum += ord(q) - 33
    mean = quality_sum / len(quality_str)
    return mean >= quality_threshold


def print_to_file(file, lines):
    for line in lines:
        file.write(line)
        file.write('\n')


def main(input_fastq, output_file_prefix, gc_bounds=(0, 100), length_bounds=(0, 2 ** 32),
                  quality_threshold=0, save_filtered=False):
    with open(input_fastq) as inputFile:
        checked_output_file = open(output_file_prefix + "_passed.fastq", 'w')
        if save_filtered:
            non_checked_output_file = open(output_file_prefix + "_failed.fastq", 'w')

        while True:
            line1 = inputFile.readline().strip()
            if line1 == '':
                break
            read = inputFile.readline().strip()
            line3 = inputFile.readline().strip()
            input_quality = inputFile.readline().strip()

            checked = check_conditions(read, input_quality, gc_bounds, length_bounds, quality_threshold)
            if checked:
                print_to_file(checked_output_file, (line1, read, line3, input_quality))
            elif save_filtered:
                print_to_file(non_checked_output_file, (line1, read, line3, input_quality))

        checked_output_file.close()
        if save_filtered:
            non_checked_output_file.close()
