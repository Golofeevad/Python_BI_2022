from typing import Union, Callable
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import psutil
import time
import warnings
import threading
import math
import multiprocessing


# 1
class RandomForestClassifierCustom(BaseEstimator):

    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'gini',
                 max_depth: int = None,
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 min_weight_fraction_leaf: Union[int, float] = 0.0,
                 max_features: Union[int, float, str] = 'sqrt',
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.0,
                 random_state: int = None,
                 class_weight=None,
                 ccp_alpha: float = 0.0):
        """
        Creates RandomForestClassifierCustom that supports parallel fit, predict_probe, and predict functions execution
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
            Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain, see Mathematical formulation.
            Note: This parameter is tree-specific.
        :param max_depth: The maximum depth of the tree.
            If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        :param min_samples_split: The minimum number of samples required to split an internal node:
            - If int, then consider min_samples_split as the minimum number.
            - If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
            This may have the effect of smoothing the model, especially in regression.
            - If int, then consider min_samples_leaf as the minimum number.
            - If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
        :param min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
        Samples have equal weight when sample_weight is not provided.
        :param max_features: The number of features to consider when looking for the best split:
            - If int, then consider max_features features at each split.
            - If float, then max_features is a fraction and max(1, int(max_features * n_features_in_)) features are considered at each split.
            - If “auto”, then max_features=sqrt(n_features).
            - If “sqrt”, then max_features=sqrt(n_features).
            - If “log2”, then max_features=log2(n_features).
            - If None, then max_features=n_features.
        :param max_leaf_nodes: Grow trees with max_leaf_nodes in best-first fashion.
            Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
        :param min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        :param random_state: Controls the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)
        :param class_weight: Weights associated with classes in the form {class_label: weight}.
            If not given, all classes are supposed to have weight one.
            For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
        :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
        The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.probas = []

    def __fit_impl(self, X: np.ndarray, y: np.ndarray):
        feats_rsm = np.random.choice(X.shape[1],
                                     self.max_features,
                                     replace=False)

        self.feat_ids_by_tree.append(feats_rsm)

        classifier = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha)

        random_x = np.random.choice(X.shape[0],
                                    size=X.shape[0],
                                    replace=True)
        initial_x = X[random_x][:, feats_rsm]
        initial_y = y[random_x]

        classifier.fit(initial_x, initial_y)
        self.trees.append(classifier)

    def fit(self, X: np.ndarray, y: np.ndarray, n_jobs: int = 1):
        """
        Build a forest of trees from the training set (X, y).
        :param X: The training input samples. Internally, its dtype will be converted to dtype=np.float32.
            If a sparse matrix is provided, it will be converted into a sparse csc_matrix.
        :param y: The target values (class labels in classification, real numbers in regression).
        :param n_jobs: amount of threads to run fit function in parallel
        :return: Fitted estimator.
        """
        self.classes_ = sorted(np.unique(y))

        with ThreadPoolExecutor(n_jobs) as pool:
            pool.map(self.__fit_impl,
                     [self] * (self.n_estimators + 1),
                     [X] * (self.n_estimators + 1),
                     [y] * (self.n_estimators + 1))

        return self

    def __predict_proba_impl(self, X: np.ndarray, i: int):
        self.probas.append(self.trees[i].predict_proba(X[:, self.feat_ids_by_tree[i]]))

    def predict_proba(self, X: np.ndarray, n_jobs: int = 1):
        """
        Predict class probabilities for X.
        The predicted class probabilities of an input sample are computed as the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of the same class in a leaf.
        :param X: The input samples. Internally, its dtype will be converted to dtype=np.float32.
            If a sparse matrix is provided, it will be converted into a sparse csr_matrix.
        :param n_jobs: amount of threads to run predict_proba function in parallel
        :return: The class probabilities of the input samples.
            The order of the classes corresponds to that in the attribute classes_.
        """
        self.probas = []

        with ThreadPoolExecutor(n_jobs) as pool:
            pool.map(self.__predict_proba_impl,
                     [self] * (self.n_estimators + 1),
                     [X] * (self.n_estimators + 1),
                     [i for i in range(self.n_estimators + 1)])

        return np.mean(self.probas, axis=0)

    def predict(self, X, n_jobs):
        """
        Predict class for X.
        The predicted class of an input sample is a vote by the trees in the forest, weighted by their probability estimates.
        That is, the predicted class is the one with highest mean probability estimate across the trees.
        :param X: The input samples. Internally, its dtype will be converted to dtype=np.float32.
        If a sparse matrix is provided, it will be converted into a sparse csr_matrix.
        :param n_jobs:
        :return: The predicted classes.
        """
        probas = self.predict_proba(X, n_jobs=n_jobs)
        predictions = np.argmax(probas, axis=0)
        return predictions


X, y = make_classification(n_samples=100000)

random_forest = RandomForestClassifierCustom(max_depth=30, n_estimators=10, max_features=2, random_state=42)

_ = random_forest.fit(X, y, n_jobs=1)

preds_1 = random_forest.predict(X, n_jobs=1)

random_forest = RandomForestClassifierCustom(max_depth=30, n_estimators=10, max_features=2, random_state=42)

_ = random_forest.fit(X, y, n_jobs=2)

preds_2 = random_forest.predict(X, n_jobs=2)

(preds_1 == preds_2).all()  # Количество worker'ов не должно влиять на предсказания


# 2
def get_memory_usage():  # Показывает текущее потребление памяти процессом
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def bytes_to_human_readable(n_bytes: str):
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for idx, s in enumerate(symbols):
        prefix[s] = 1 << (idx + 1) * 10
    for s in reversed(symbols):
        if n_bytes >= prefix[s]:
            value = float(n_bytes) / prefix[s]
            return f"{value:.2f}{s}"
    return f"{n_bytes}B"


SIZE_MAP = {
    'B': 1,
    'K': 1 << 10,
    'M': 1 << 20,
    'G': 1 << 30,
    'T': 1 << 40,
    'P': 1 << 50,
    'E': 1 << 60,
    'Z': 1 << 70,
    'Y': 1 << 80,
}


def to_bytes(inp: str):
    size = float(inp[:-1])
    unit = inp[-1]
    return size * SIZE_MAP[unit]


def safe_to_bytes(inp: str, def_value: float = None):
    if inp is None:
        return def_value
    else:
        return to_bytes(inp)



def memory_limit(soft_limit: str = None, hard_limit: str = None, poll_interval: int = 1):
    """
    Decorator that monitors functions memory usage
    :param soft_limit: if function consumes more than soft_limit amount of memory, a warning will be issued
    :param hard_limit: if function consumes more than hard_limit amount of memory, an exception will be thrown
    :param poll_interval: amount in seconds to check on memory state
    :return: decorator
    """

    def memory_decorator(fn):
        def memory_limit_wrapper(*args, **kwargs):
            soft_limit_bytes = safe_to_bytes(soft_limit, def_value=math.inf)
            hard_limit_bytes = safe_to_bytes(hard_limit, def_value=math.inf)

            stop_event = threading.Event()

            def monitor_memory():
                soft_reached = False
                while not stop_event.is_set():
                    current_memory = get_memory_usage()
                    memory_delta = current_memory - starting_memory
                    if not soft_reached and memory_delta > soft_limit_bytes:
                        soft_reached = True
                        warnings.warn(
                            f'Soft limit exceeded: soft_limit = {soft_limit}, current memory consumption = {bytes_to_human_readable(memory_delta)}')
                    if memory_delta > hard_limit_bytes:
                        stop_event.set()
                        raise MemoryError(
                            f'Memory limit exceeded: hard_limit = {hard_limit}, current memory consumption = {bytes_to_human_readable(memory_delta)}')
                    time.sleep(poll_interval)

            monitor_thread = threading.Thread(target=monitor_memory)
            starting_memory = get_memory_usage()
            monitor_thread.start()
            try:
                return fn(*args, **kwargs)
            finally:
                stop_event.set()
                monitor_thread.join()

        return memory_limit_wrapper

    return memory_decorator

@memory_limit(soft_limit="512M", hard_limit="1.5G", poll_interval=0.1)
def memory_increment():
    """
    Функция для тестирования

    В течение нескольких секунд достигает использования памяти 1.89G
    Потребление памяти и скорость накопления можно варьировать, изменяя код
    """
    lst = []
    for i in range(50000000):
        if i % 500000 == 0:
            time.sleep(0.1)
        lst.append(i)
    return lst


# 3
def parallel_map(target_func: Callable,
                 args_container: list = None,
                 kwargs_container: list = None,
                 n_jobs: int = None):
    """
    Parallelizes execution of given function in efficient way
    :param target_func: function to invoke
    :param args_container: list of positional arguments for invocations. Each element represents an arguments set for one invocation.
        If you want to specify multiple positional arguments, use tuple
    :param kwargs_container: list of named arguments for invokationa. Each element represents an arguments set for one invocation.
    :param n_jobs: amount on threads to run target_func in. Note that the actual number of threads may be lower than n_jobs depending on how many invokations is required by parameters.
    :return: list of invokations results
    """
    if args_container is None:
        args_count = 0
    else:
        args_count = len(args_container)
    if kwargs_container is None:
        kwargs_count = 0
    else:
        kwargs_count = len(kwargs_container)

    if args_count != kwargs_count and args_count != 0 and kwargs_count != 0:
        raise ValueError("Number of arguments in args_container must match number of arguments in kwargs_container")

    if n_jobs is None:
        n_threads = multiprocessing.cpu_count()
    else:
        n_threads = n_jobs

    if args_container is None and kwargs_container is None:
        n_iters = n_threads
    else:
        n_iters = max(args_count, kwargs_count)

    n_threads = min(n_threads, n_iters)

    with ThreadPoolExecutor(n_threads) as pool:
        futures = []
        for i in range(n_iters):
            if args_container is None:
                args = []
            else:
                args = args_container[i]
            if kwargs_container is None:
                kwargs = {}
            else:
                kwargs = kwargs_container[i]
            if not isinstance(args, list):
                args = [args]
            future = pool.submit(target_func, *args, **kwargs)
            futures.append(future)

    return list(map(lambda future: future.result(), futures))


# Это только один пример тестовой функции, ваша parallel_map должна уметь эффективно работать с ЛЮБЫМИ функциями
# Поэтому обязательно протестируйте код на чём-нибудбь ещё
def test_func(x=1, s=2, a=1, b=1, c=1):
    time.sleep(s)
    return a * x ** 2 + b * x + c

# Пример 2.1
# Отдельные значения в args_container передаются в качестве позиционных аргументов
parallel_map(test_func, args_container=[1, 2.0, 3j - 1,
                                        4])  # Здесь происходят параллельные вызовы: test_func(1) test_func(2.0) test_func(3j-1) test_func(4)

# Пример 2.2
# Элементы типа tuple в args_container распаковываются в качестве позиционных аргументов
parallel_map(test_func, [(1, 1), (2.0, 2), (3j - 1, 3),
                         4])  # Здесь происходят параллельные вызовы: test_func(1, 1) test_func(2.0, 2) test_func(3j-1, 3) test_func(4)

# Пример 3.1
# Возможна одновременная передача args_container и kwargs_container, но количества элементов в них должны быть равны
parallel_map(test_func,
             args_container=[1, 2, 3, 4],
             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}, {"s": 3}])

# Здесь происходят параллельные вызовы: test_func(1, s=3) test_func(2, s=3) test_func(3, s=3) test_func(4, s=3)

# Пример 3.2
# args_container может быть None, а kwargs_container задан явно
parallel_map(test_func,
             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}, {"s": 3}])

# Пример 3.3
# kwargs_container может быть None, а args_container задан явно
parallel_map(test_func,
             args_container=[1, 2, 3, 4])

# Пример 3.4
# И kwargs_container, и args_container могут быть не заданы
parallel_map(test_func)

# Пример 3.4
# И kwargs_container, и args_container могут быть не заданы
parallel_map(test_func)

# Пример 3.5
# При несовпадении количеств позиционных и именованных аргументов кидается ошибка
parallel_map(test_func,
             args_container=[1, 2, 3, 4],
             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}])

# Пример 4.1
# Если функция не имеет обязательных аргументов и аргумент n_jobs не был передан, то она выполняется параллельно столько раз, сколько ваш CPU имеет логических ядер
# В моём случае это 24, у вас может быть больше или меньше
parallel_map(test_func)

# Пример 4.2
# Если функция не имеет обязательных аргументов и передан только аргумент n_jobs, то она выполняется параллельно n_jobs раз
parallel_map(test_func, n_jobs=2)

# Пример 4.3
# Если аргументов для target_func указано МЕНЬШЕ, чем n_jobs, то используется такое же количество worker'ов, сколько было передано аргументов
parallel_map(test_func,
             args_container=[1, 2, 3],
             n_jobs=5)  # Здесь используется 3 worker'a

# Пример 4.4
# Аналогичный предыдущему случай, но с именованными аргументами
parallel_map(test_func,
             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}],
             n_jobs=5)  # Здесь используется 3 worker'a

# Пример 4.5
# Комбинация примеров 4.3 и 4.4 (переданы и позиционные и именованные аргументы)
parallel_map(test_func,
             args_container=[1, 2, 3],
             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}],
             n_jobs=5)  # Здесь используется 3 worker'a

# Пример 4.6
# Если аргументов для target_func указано БОЛЬШЕ, чем n_jobs, то используется n_jobs worker'ов
parallel_map(test_func,
             args_container=[1, 2, 3, 4],
             kwargs_container=None,
             n_jobs=2)  # Здесь используется 2 worker'a

# Пример 4.7
# Время выполнения оптимизируется, данный код должен отрабатывать за 5 секунд
parallel_map(test_func,
             kwargs_container=[{"s": 5}, {"s": 1}, {"s": 2}, {"s": 1}],
             n_jobs=2)

def test_func2(string, sleep_time=1):
    time.sleep(sleep_time)
    return string


# Пример 5
# Результаты возвращаются в том же порядке, в котором были переданы соответствующие аргументы вне зависимости от того, когда завершился worker
arguments = ["first", "second", "third", "fourth", "fifth"]
parallel_map(test_func2,
             args_container=arguments,
             kwargs_container=[{"sleep_time": 5}, {"sleep_time": 4}, {"sleep_time": 3}, {"sleep_time": 2},
                               {"sleep_time": 1}])

def test_func3():
    def inner_test_func(sleep_time):
        time.sleep(sleep_time)

    return parallel_map(inner_test_func, args_container=[1, 2, 3])


# Пример 6
# Работает с функциями, созданными внутри других функций
test_func3()
