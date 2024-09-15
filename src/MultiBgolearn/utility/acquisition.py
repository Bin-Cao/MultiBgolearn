import numpy as np
from .funs import get_pareto_front, calculate_lebesgue_measure,Monte_Carlo


def multi_BGO(y, vs_mean, vs_vars, method, max_search=True, times=10):
    """
    Perform Multi-BGO optimization.

    :param y: training targets
    :param vs_mean: means of virtual data points
    :param vs_vars: variances of virtual data points
    :param method: Multi-BGO method (e.g., 'EHVI')
    :param max_search: whether to perform maximization (True) or minimization (False)
    :param times: number of Monte Carlo samples
    :return: index of the virtual data point with the highest expected improvement
    """
    #if method == 'EHVI':
    # Calculate the current Pareto front based on the training targets
    pareto_front = get_pareto_front(y,max_search)
    current_lebesgue_measure = calculate_lebesgue_measure(pareto_front,max_search)

    improvements = []
    
    for k in range(len(vs_mean)):
        # Monte Carlo sampling to generate virtual samples
        y_samples = Monte_Carlo(y, vs_mean[k], vs_vars[k], times=times)
        
        # Compute the new Pareto front by adding y_sample to the original data
        improvement_sum = 0
        
        for y_sample in y_samples:
            extended_y = np.vstack([y, y_sample])
            new_pareto_front = get_pareto_front(extended_y,max_search)
            new_lebesgue_measure = calculate_lebesgue_measure(new_pareto_front,max_search)
            
            # Calculate the difference in Lebesgue measures
            improvement = new_lebesgue_measure - current_lebesgue_measure if max_search else current_lebesgue_measure - new_lebesgue_measure
            improvement_sum += max(improvement,0)
        
        # Average improvement over Monte Carlo samples
        avg_improvement = improvement_sum / times
        improvements.append(avg_improvement)

    # Return the index of the virtual sample with the highest expected improvement
    best_idx = np.argmax(improvements) if max_search else np.argmin(improvements)
    #elif ...
    return best_idx,improvements
