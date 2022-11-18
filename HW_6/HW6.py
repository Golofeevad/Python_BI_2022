import re
import matplotlib.pyplot as pl


# Task 1
with open('references.txt') as input_file:
    with open('ftps', 'w') as output_file:
        for line in input_file:
            matches = re.findall(r'(ftp[\w./#:@]+)', line)
            if matches is not None:
                for match in matches:
                    output_file.write(match)
                    output_file.write("\n")

# Task 2,3,4
task1_res = []
task2_res = []
task3_res = []


def findall_overlapping(regex, seq):
    result_list = []
    search_start_pos = 0

    while True:
        result = regex.search(seq, search_start_pos)
        if result is None:
            break
        result_list.append(result.groups()[0])
        search_start_pos = result.start() + 1
    return result_list


numbers_pattern = re.compile(r'\D(\d+(?:\.\d+)?)\D')
words_with_a_pattern = re.compile(r'\W(\w+[aA]\w+)\W')
exclamation_sentences_pattern = re.compile(r'(?:(?:[.!?] )|")((?:[\w\d;, -]|(?:\d\.\d))+!)')

with open('2430AD.txt') as input_file:
    for line in input_file:
        matches = numbers_pattern.findall(line)
        task1_res += matches

        matches = words_with_a_pattern.findall(line)
        task2_res += matches

        matches = findall_overlapping(exclamation_sentences_pattern, line)
        task3_res += matches


def findall_overlapping_words(regex, seq):
    result_list = []
    search_start_pos = 0

    while True:
        result = regex.search(seq, search_start_pos)
        if result is None:
            break
        groups = result.groups()
        if groups[0] is not None:
            result_list.append(result.groups()[0].lower())
        elif groups[1] is not None:
            result_list.append(result.groups()[1].lower())
        search_start_pos = result.end() - 1
    return result_list


# Task 5
def read_words():
    word_pattern = re.compile(r'\W((?:\w\.\w\.)|(?:\d+(?:\.\d+)+)|[\w\']+)\W|^((?:\w\.\w\.)|(?:\d+(?:\.\d+)+)|[\w\']+)\W')
    result = set()
    with open('2430AD.txt') as input_file:
        for line in input_file:
            for word in findall_overlapping_words(word_pattern, line):
                result.add(word)
    return result


def count(list):
    res = {}
    for i in list:
        if i in res:
            res[i] = res[i] + 1
        else:
            res[i] = 1
    return res


words = read_words()

lengths = [len(word) for word in words]
counts = count(lengths)

pl.bar(counts.keys(), counts.values())
pl.show()


# Task 6
def replacement(match):
    part = match[0]
    return part + 'к' + part[1]


def translate(line):
    return re.sub(r'([йцкнгшщзхфвпрлджчсмтб][уеаоэяию])', replacement, line)


# Task 7
def split_words(sentence):
    return tuple(re.split(r"[^\w']", sentence))


def findall_overlapping_arbitrary_groups(regex, seq):
    result_list = []
    search_start_pos = 0

    while True:
        result = regex.search(seq, search_start_pos)
        if result is None:
            break
        groups = result.groups()
        for group in groups:
            if group is not None:
                result_list.append(split_words(group))
                break
        search_start_pos = result.end() - 1
    return result_list


def find_n_words_sentences(text, n):
    exclamation_sentences_pattern = re.compile(rf'(?:[.?!] ((?:[\w\']+ [\w\']+){{{n - 1}}})[.?!])|(?:^((?:[\w\']+ [\w\']+){{{n - 1}}})[.?!])')
    return findall_overlapping_arbitrary_groups(exclamation_sentences_pattern, text)


