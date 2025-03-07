from math import log2

from torch import Tensor, sort
import torch


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    """
    Функция для подсчета количества неправильно упорядоченных пар.
    Корректное упорядочивание пары (y_true, y_pred) означает, что y_true >= y_pred.
    :param ys_true: тензор с истинными значениями
    :param ys_pred: тензор с предсказанными значениями
    :return: количество неправильно упорядоченных пар
    """ 
    # return torch.sum(ys_true.unsqueeze(1) < ys_true.unsqueeze(0) & ys_pred.unsqueeze(1) > ys_pred.unsqueeze(0)).item()
    # Сравниваем элементы ys_true и ys_pred
    true_comparison = ys_true.unsqueeze(1) < ys_true.unsqueeze(0)  # Маска для ys_true
    pred_comparison = ys_pred.unsqueeze(1) > ys_pred.unsqueeze(0)  # Маска для ys_pred

    # Используем логическое И (logical_and) вместо побитового И (&)
    swapped_pairs = torch.logical_and(true_comparison, pred_comparison)

    # Считаем количество неправильно упорядоченных пар
    return torch.sum(swapped_pairs).item()


def compute_gain(y_value: float, gain_scheme: str) -> float:
    """
    Функция для вычисления значения gain для одного элемента.
    :param y_value: значение y
    :param gain_scheme: 'const' - константное начисление, 'exp2' - начисление по формуле 2^relevance - 1
    """
    if gain_scheme == 'const':
        return y_value
    elif gain_scheme == 'exp2':
        return 2 ** y_value - 1
    else:
        raise ValueError(f'Unknown gain_scheme: {gain_scheme}')


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    """
    Функция для вычисления значения Discounted Cumulative Gain.
    :param ys_true: тензор с истинными значениями
    :param ys_pred: тензор с предсказанными значениями
    :param gain_scheme: 'const' - константное начисление, 'exp2' - начисление по формуле 2^relevance - 1
    """
    _, indices = sort(ys_pred, descending=True)
    ys_true = ys_true[indices]
    return torch.sum(compute_gain(ys_true, gain_scheme) / torch.log2(torch.arange(ys_true.shape[0], dtype=torch.float32) + 2)).item()


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    """
    Функция для вычисления значения Normalized Discounted Cumulative Gain.
    :param ys_true: тензор с истинными значениями
    :param ys_pred: тензор с предсказанными значениями
    :param gain_scheme: 'const' - константное начисление, 'exp2' - начисление по формуле 2^relevance - 1
    """
    return dcg(ys_true, ys_pred, gain_scheme) / dcg(ys_true, ys_true, gain_scheme)


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    """
    Функция для вычисления значения Precision@k.
    :param ys_true: тензор с истинными значениями, содержит нули (не релевантный документ) и единицы (релевантный документ).
    :param ys_pred: тензор с предсказанными значениями
    :param k: количество элементов для ранжирования.
    Возвращает -1 если нет ни одного релевантного документа.
    """
    _, indices = sort(ys_pred, descending=True)
    ys_true = ys_true[indices]
    if torch.sum(ys_true[:k]) == 0:
        return -1
    return torch.sum(ys_true[:k]).item() / k

def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    """
    Функция для вычисления значения Mean Reciprocal Rank (среднее гармоническое между рангами).
    Метрика MRR - есть только один релевантный документ на запрос, оценивает на сколько далеко этот документ от топа.
    :param ys_true: тензор с истинными значениями, содержит нули (не релевантный документ) и максимум одну единицу (релевантный документ).
    :param ys_pred: тензор с предсказанными значениями.
    """
    _, indices = sort(ys_pred, descending=True)
    ys_true = ys_true[indices]
    return 1 / (torch.argmax(ys_true).item() + 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    """
    Функция для вычисления значения PFound.
    Метрика PFound - вероятность того, что пользователь увидит релевантный документ на запрос.
    pfound = sum(pLook * pRel) 
    :param ys_true: тензор с истинными значениями, нормированными на [0, 1].
    :param ys_pred: тензор с предсказанными значениями.
    :param p_break: вероятность того, что пользователь прекратит просмотр документов.
    pLook[0] = 1
    pLook[i] = pLook[i-1] * (1 - p_break) * (1 - pRel[i-1])
    """
    _, indices = sort(ys_pred, descending=True)
    ys_true = ys_true[indices]
    pLook = torch.zeros(ys_true.shape[0])
    pLook[0] = 1
    pLook[1:] = pLook[:-1] * (1 - p_break) * (1 - ys_true[:-1])
    return torch.sum(pLook * ys_true).item()


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    """
    Функция для вычисления значения Average Precision для бинарной разметки.
    :param ys_true: тензор с истинными значениями, содержит нули (не релевантный документ) и единицы (релевантный документ).
    :param ys_pred: тензор с предсказанными значениями.
    Если нет ни одного релевантного документа, возвращаем -1.
    """
    _, indices = sort(ys_pred, descending=True)
    ys_true = ys_true[indices]
    if torch.sum(ys_true) == 0:
        return -1
    return torch.sum(torch.cumsum(ys_true, 0) * ys_true / torch.arange(1, ys_true.shape[0] + 1, dtype=torch.float32)).item()
