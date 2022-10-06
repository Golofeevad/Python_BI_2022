# Complementator
transcribe_dict = dict(A='A', T='U', G='G', C='C', a='a', t='u', g='g', c='c')
complementDNA_dict = dict(A='T', T='A', G='C', C='G', a='t', t='a', g='c', c='g')
complementRNA_dict = dict(A='U', U='A', G='C', C='G', a='u', u='a', g='c', c='g')
while True:
    user_command = input('Enter your command:')
    if user_command == 'exit':
        print('Bye!')
        break
    elif user_command == 'transcribe':
        sequence = list(input('Enter your sequence:'))
        flag_correct = 1
        for i in range(len(sequence)):
            if sequence[i] in transcribe_dict.keys():
                sequence[i] = transcribe_dict[sequence[i]]
            else:
                print('What did you give me?!')
                flag_correct = 0
                break
        if flag_correct == 1:
            print("".join(sequence))
    elif user_command == 'complement':
        sequence = list(input('Enter your sequence:'))
        flag_RNA = 0
        flag_DNA = 0
        for i in range(len(sequence)):
            if (sequence[i] == 'U') | (sequence[i] == 'u'):
                flag_RNA = 1
            elif (sequence[i] == 'T') | (sequence[i] == 't'):
                flag_DNA = 1
        if (flag_RNA == 1) & (flag_DNA == 1):
            print('What did you give me?!')
        elif flag_RNA == 1:
            flag_correct = 1
            for i in range(len(sequence)):
                if sequence[i] in complementRNA_dict.keys():
                    sequence[i] = complementRNA_dict[sequence[i]]
                else:
                    print('What did you give me?!')
                    flag_correct = 0
                    break
            if flag_correct == 1:
                print("".join(sequence))
        elif (flag_DNA == 1) | (flag_RNA == 0) & (flag_DNA == 0):
            flag_correct = 1
            for i in range(len(sequence)):
                if sequence[i] in complementDNA_dict.keys():
                    sequence[i] = complementDNA_dict[sequence[i]]
                else:
                    print('What did you give me?!')
                    flag_correct = 0
                    break
            if flag_correct == 1:
                print("".join(sequence))
    elif user_command == 'reverse':
        sequence = list(input('Enter your sequence:'))
        flag_correct = 0
        flag_RNA = 0
        flag_DNA = 0
        for i in range(len(sequence)):
            if (sequence[i] == 'U') | (sequence[i] == 'u'):
                flag_RNA = 1
            elif (sequence[i] == 'T') | (sequence[i] == 't'):
                flag_DNA = 1
        if (flag_RNA == 1) & (flag_DNA == 1):
            print('What did you give me?!')
        else:
            for i in range(len(sequence)):
                if (sequence[i] in complementDNA_dict.keys()) | (sequence[i] in complementRNA_dict.keys()):
                    flag_correct = 1
                else:
                    print('What did you give me?!')
                    break
            if flag_correct == 1:
                print("".join(reversed(sequence)))
    elif user_command == 'reverse complement':
        sequence = list(input('Enter your sequence:'))
        flag_RNA = 0
        flag_DNA = 0
        for i in range(len(sequence)):
            if (sequence[i] == 'U') | (sequence[i] == 'u'):
                flag_RNA = 1
            elif (sequence[i] == 'T') | (sequence[i] == 't'):
                flag_DNA = 1
        if (flag_RNA == 1) & (flag_DNA == 1):
            print('What did you give me?!')
        elif flag_RNA == 1:
            flag_correct = 1
            for i in range(len(sequence)):
                if sequence[i] in complementRNA_dict.keys():
                    sequence[i] = complementRNA_dict[sequence[i]]
                else:
                    print('What did you give me?!')
                    flag_correct = 0
                    break
            if flag_correct == 1:
                print("".join(reversed(sequence)))
        elif (flag_DNA == 1) | (flag_RNA == 0) & (flag_DNA == 0):
            flag_correct = 1
            for i in range(len(sequence)):
                if sequence[i] in complementDNA_dict.keys():
                    sequence[i] = complementDNA_dict[sequence[i]]
                else:
                    print('What did you give me?!')
                    flag_correct = 0
                    break
            if flag_correct == 1:
                print("".join(reversed(sequence)))
    else:
        print('I am not so smart to do it...')































