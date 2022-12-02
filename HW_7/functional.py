# Functional programming


def sequential_map(*args):
    *funcs, elements = args
    mapped_elements = list()
    for element in elements:
        res = element
        for func in funcs:
            res = func(res)
        mapped_elements.append(res)
    return mapped_elements



def consensus_filter(*args):
    *predicates, elements = args
    filtered_elements = list()
    for element in elements:
        satisfies = True
        for predicate in predicates:
            if not predicate(element):
                satisfies = False
                break
        if satisfies:
            filtered_elements.append(element)
    return filtered_elements


def conditional_reduce(predicate, reducer, elements):
    accumulator = None
    for element in elements:
        if predicate(element):
            if accumulator is None:
                accumulator = element
            else:
                accumulator = reducer(accumulator, element)
    return accumulator


def func_chain(*args):
    def chained(x):
        res = x
        for func in args:
            res = func(res)
        return res

    return lambda x: chained(x)


# Bonus 2 points
def sequential_map_2(*args, elements):
    chain = func_chain(*args)
    return [chain(element) for element in elements]


def multiple_partial(*args, **keywords):
    partial_funcs = list()
    for func in args:
        def partial_func_generator(func, **keywords):
            def partial_func(*fargs, **fkeywords):
                new_keywords = {**keywords, **fkeywords}
                return func(*fargs, **new_keywords)
            return partial_func
        partial_funcs.append(partial_func_generator(func, **keywords))
    return partial_funcs
    
