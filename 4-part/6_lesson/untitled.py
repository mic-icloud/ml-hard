import json
import pandas as pd
import numpy as np
import base64
from io import StringIO

from context import context

# metric threshold
# TODO: change value
threshold = 0.30

# here we put indices of incorrectly solved tasks
wrong_answers = []

# a dict of required files and their points
user_input_names = {'prediction': 10}

# a dict with answers in base64 https://www.browserling.com/tools/file-to-base64
answers = {'prediction': 'U0tVLGRhdGVzLGxvd2VyX2JvdW5kLHVwcGVyX2JvdW5kCjEyNTM4LDIwMTktMTItMDIsMTQwLjYsMTU1LjQKMTI1MzgsMjAxOS0xMi0wOSwxMjMuMjUyODAwMDAwMDAwMDEsMTM4Ljk4NzIKMTI1MzgsMjAxOS0xMi0xNiwxMjEuMTIzMjAwMDAwMDAwMDEsMTM5LjM1NjgwMDAwMDAwMDAyCjEyNTM4LDIwMTktMTItMjYsMTIxLjQ0LDE0Mi41NgoxODk2OSwyMDE5LTAzLTExLDE1OC42NSwxNzUuMzUKMTg5NjksMjAxOS0wMy0xOCwxNTEuMzQsMTcwLjY2CjE4OTY5LDIwMTktMDMtMjUsMTQ2Ljk0LDE2OS4wNgoxODk2OSwyMDE5LTA0LTAxLDE1My42NCwxODAuMzYKMjk4NjIsMjAxOC0wNi0xOCwxOTIuMjgsMjEyLjUyCjI5ODYyLDIwMTgtMDYtMjUsMTkwLjI1NiwyMTQuNTQ0CjI5ODYyLDIwMTgtMDctMDIsMTg4Ljk3NjAwMDAwMDAwMDAzLDIxNy40MjQKMjk4NjIsMjAxOC0wNy0wOSwxODYuOTQ0MDAwMDAwMDAwMDIsMjE5LjQ1NjAwMDAwMDAwMDAyCjMxMjA5LDIwMTktMTItMDksMTM3Ljk0LDE1Mi40NTk5OTk5OTk5OTk5OAozMTIwOSwyMDE5LTEyLTE3LDE0My4xMDU2LDE2MS4zNzQ0CjMxMjA5LDIwMTktMTItMjMsMTM1Ljg1NDQsMTU2LjMwNTYwMDAwMDAwMDAzCjMxMjA5LDIwMTktMTItMzEsMTQwLjg3MDQwMDAwMDAwMDAyLDE2NS4zNjk2CjM1MjAwLDIwMTktMTItMDIsMjA0LjgyMDAwMDAwMDAwMDAyLDIyNi4zODAwMDAwMDAwMDAwMgozNTIwMCwyMDE5LTEyLTA5LDIwNC43MzIsMjMwLjg2ODAwMDAwMDAwMDAyCjM1MjAwLDIwMTktMTItMTYsMjA1LjYyMzAwMDAwMDAwMDAyLDIzNi41NzcwMDAwMDAwMDAwMwozNTIwMCwyMDE5LTEyLTIzLDIwMi40MDAwMDAwMDAwMDAwMywyMzcuNjAwMDAwMDAwMDAwMDIKNDgzMzYsMjAxOC0wNy0wMiwzMzcuMjUsMzcyLjc1CjQ4MzM2LDIwMTgtMDctMDksMzM0LjY0LDM3Ny4zNgo0ODMzNiwyMDE4LTA3LTE2LDMzMi45NCwzODMuMDYKNDgzMzYsMjAxOC0wNy0yMywzMzAuMjgsMzg3LjcyCjUwNjI4LDIwMTktMDEtMjMsMzQyLjAsMzc4LjAKNTA2MjgsMjAxOS0wMS0yOCwzMzcuNDYsMzgwLjU0CjUwNjI4LDIwMTktMDItMDQsMzM0LjgsMzg1LjIKNTA2MjgsMjAxOS0wMi0xMSwzMzIuMTIsMzg5Ljg4CjYxODQzLDIwMTktMTItMTAsMTIxLjYsMTM0LjQKNjE4NDMsMjAxOS0xMi0xOSwxMjIuMiwxMzcuOAo2MTg0MywyMDE5LTEyLTIzLDEyMS44MywxNDAuMTcKNjE4NDMsMjAxOS0xMi0zMCwxMTkuNiwxNDAuNAo3MDAzNiwyMDE5LTEyLTAyLDI5NS40NSwzMjYuNTUKNzAwMzYsMjAxOS0xMi0wOSwyOTMuMjgsMzMwLjcyCjcwMDM2LDIwMTktMTItMTYsMjk0LjgxLDMzOS4xOQo3MDAzNiwyMDE5LTEyLTIzLDI4My4zNiwzMzIuNjQKNzE0MjMsMjAxOS0xMC0xNSwxMzMuOTUsMTQ4LjA1CjcxNDIzLDIwMTktMTAtMjEsMTM1LjM2LDE1Mi42NAo3MTQyMywyMDE5LTEwLTI4LDEzMi4wNiwxNTEuOTQKNzE0MjMsMjAxOS0xMS0wNCwxMzAuNjQsMTUzLjM2Cg=='}

def calculate_intersection(lower_bound, upper_bound, predicted_lower_bound, predicted_upper_bound):
    if upper_bound < predicted_lower_bound or lower_bound > predicted_upper_bound :
        intersection = 0
    elif lower_bound <= predicted_lower_bound : 
        if upper_bound <= predicted_upper_bound:
            intersection = upper_bound - predicted_lower_bound
        elif upper_bound > predicted_upper_bound:
            intersection = predicted_upper_bound - predicted_lower_bound
    elif lower_bound > predicted_lower_bound:
        if upper_bound >= predicted_upper_bound:
            intersection = predicted_upper_bound - lower_bound
        elif upper_bound < predicted_upper_bound:
            intersection = upper_bound - lower_bound
    return intersection

def calculate_union(lower_bound, upper_bound, predicted_lower_bound, predicted_upper_bound):
    min_ = np.min([lower_bound, upper_bound, predicted_lower_bound, predicted_upper_bound])
    max_ = np.max([lower_bound, upper_bound, predicted_lower_bound, predicted_upper_bound])
    return max_ - min_


def get_mean_IOU(df:pd.DataFrame, answer:pd.DataFrame):
    """
    Function returns mean IOU for user's answer
    Args:
        df (pd.DataFrame): user input
        answer (pd.DataFrame): correct answer

    Returns:
        float: mean IOU

    """
    calc_df = answer.merge(df[['SKU','dates', 'predicted_lower_bound', 'predicted_upper_bound']]
                           , how='left', on=['SKU','dates'])
    calc_df = calc_df.round(2)
    calc_df['intersection'] = calc_df.apply(lambda x: calculate_intersection(x.lower_bound, x.upper_bound
                                               , x.predicted_lower_bound, x.predicted_upper_bound), axis=1)
    calc_df['union'] = calc_df.apply(lambda x: calculate_union(x.lower_bound, x.upper_bound
                                               , x.predicted_lower_bound, x.predicted_upper_bound), axis=1)
    calc_df['iou'] = calc_df.intersection / calc_df.union
    return np.mean(calc_df['iou'])




def calculate_metric(user, answer):
    """
    Function returns comparison result whether metric threshold is larger then calculated value
    Args:
        user (pd.DataFrame): user input
        answer (pd.DataFrame): correct answer

    Returns:
        bool: whether metric threshold is larger then calculated value

    """

    # TODO: change metric
    #  and comparison logic if needed
    metric = get_mean_IOU(user, answer)
    result = threshold < metric

    # adding score value for students
    wrong_answers.append(f'Score = {metric:.3f}.')

    # if threshold was not passed, the task is unsuccessful, so, adding message
    if not result:
        wrong_answers.append('Не пройден трешхолд.')

    return result


def check_file(i, user_input, answer, file_type, points):
    """
    Function performes check by file type: csv, txt TODO Alex

    Args:
        i (int): task number
        user_input (str): expected user file name
        answer (str): correct answer in base64
        file_type (str): file type like csv or txt (pending)
        points (int): number of points that can be acquired for the task

    Returns:
        int: total number of points for current task
    """

    # reading base64 strings with correct answer and user input
    answer = StringIO(base64.b64decode(answer).decode('utf-8'))
    user = StringIO(base64.b64decode(context[user_input]).decode('utf-8'))

    # here we work only with csv
    if file_type == 'csv':

        # csv can be read with pandas
        answer = pd.read_csv(answer, dtype={'SKU': int, 'price_per_sku': float})
        user = pd.read_csv(user, dtype={'SKU': int, 'price_per_sku': float})

        # if there is only one column, probably the separator is incorrect
        if len(user.columns) == 1:
            wrong_answers.append(f'{i + 1} (неверный формат ответа)')

            # no check for such file
            return 0

        # sorting to compare
        #user, answer = sort_dfs(user, answer)

        # getting number of points for this task:
        # if dfs are equal, then maximum number is set as a result
        # 0 otherwise
        result = calculate_metric(user, answer) * points

    # TODO: ADD OTHER TYPES
    else:
        result = -1

    return result


# Grisha's stuff
result = {'format': 'kchecker', 'v': 1, 'data': {}}

# as we will add points, setting them to 0 at the beginning
result['data']['points'] = 0

# this try-except works for output generation in LMS
try:

    # for each input file
    for i, input_name in enumerate(user_input_names):

        # adding acquired points
        result['data']['points'] += int(check_file(i, input_name, answers[input_name], 'csv',
                                                   user_input_names[input_name]))

    if wrong_answers:
        raise Exception(f"{' '.join(map(str, wrong_answers))}")

# here we get the text for user from above
except Exception as e:
    result['data']['exception'] = f'{e}'

# adding points to output
print(json.dumps(result))

# if there are no points, the task is unsuccessful (Ошибка)
if not result['data'].get('points'):
    exit(1)
