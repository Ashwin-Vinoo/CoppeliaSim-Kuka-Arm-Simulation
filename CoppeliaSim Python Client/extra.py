from typing import Dict, Tuple, Callable, Iterable
import numpy

# This is the penalty function
def model_quadratic(model_parameters: dict):
    # We extract the parameters of the point from the dictionary
    a = model_parameters['a']
    b = model_parameters['b']
    c = model_parameters['c']
    # We return the value of the penalty function
    return 1.75 + (a-0.5)**2 + (b-0.75)**2 + (c-0.25)**2

# The class problem where we define our grid search method
class Problem:
    # We call this as static so it can be run without creating a class variable object
    @staticmethod
    def grid_search(search_space: Dict[str, Iterable], scoring_func: Callable[[Dict[str, float]], float]) -> Tuple[float, Dict[str,float]]:
        # We first extractthe individual lists
        a_list = search_space['a']
        b_list = search_space['b']
        c_list = search_space['c']
        # We next obtain the lenght of a, b and c search space elements
        a_len = len(a_list)
        b_len = len(b_list)
        c_len = len(c_list)
        # We define the tuple we will return
        point_with_lowest_score_tuple = [None, float("inf")]
        # We iterate through all the elements of each search space term
        for i in range(a_len):
            for j in range(b_len):
                for k in range(c_len):
                    # We define a dictionary as follows that matches the current point
                    current_dict = {'a':a_list[i], 'b':b_list[j], 'c':c_list[k]}
                    # We check the score of these coordinates
                    score = model_quadratic(current_dict)
                    # if score less than minimum score obtained so far
                    if score < point_with_lowest_score_tuple[1]:
                        # We update the point
                        point_with_lowest_score_tuple[0] = current_dict
                        # We update the score
                        point_with_lowest_score_tuple[1] = score
        # We return the tuple
        return point_with_lowest_score_tuple

# we run it with input given
print(Problem.grid_search({
'a': numpy.arange(0.0, 1.0, 0.05),
'b': numpy.arange(0.0, 1.0, 0.05),
'c' : numpy.arange(0.0, 1.0, 0.05),
}, model_quadratic))