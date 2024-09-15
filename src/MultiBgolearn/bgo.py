from .utility.acquisition import multi_BGO
from .utility.funs import read_in, build_models, pre_model, normalize_data,read_in_vs
import numpy as np

def fit(dataset, VSdataset, object_num,  max_search=True, method='EHVI', assign_model=False, bootstrap=5):
    """
    =======================================================================
    PACKAGE: Multi-objective Bayesian Global Optimization-learn (MultiBgolearn)
    Author: Bin CAO <binjacobcao@gmail.com> 
    Guangzhou Municipal Key Laboratory of Materials Informatics, Advanced Materials Thrust,
    Hong Kong University of Science and Technology (Guangzhou), Guangzhou 511400, Guangdong, China
    =======================================================================
    Please feel free to open issues on GitHub:
    https://github.com/Bin-Cao/MultiBgolearn
    or 
    contact Bin Cao (bcao686@connect.hkust-gz.edu.cn) with any issues, comments, or suggestions 
    regarding the usage of this code.
    =======================================================================
    Thank you for choosing MultiBgolearn for material design! 
    MultiBgolearn is designed to enhance machine learning applications in research and acts as an efficient 
    supplement to the Bgolearn project.
    
    While Bgolearn is intended for optimizing single-objective material properties, MultiBgolearn is 
    specifically developed to optimize multi-objective material properties. 
    =======================================================================

    This function applies Multi-objective Bayesian Global Optimization (MOBO) to a given dataset, optimizing
    for either maximum or minimum material properties based on the selected method.

    :param dataset: str
        The path to the dataset containing both features and response variables, e.g., './data/dataset.csv'.
    :param VSdataset: str
        The path to the virtual space (VS), where candidate data for optimization is stored.
    :param object_num: int
        The number of objectives (target properties) to optimize, e.g., 3 for a three-objective optimization.
    :param max_search: bool, optional, default=True
        Determines the optimization direction. 
        - True for maximizing the objectives.
        - False for minimizing the objectives.
    :param method: str, optional, default='EHVI'
        The method used for multi-objective Bayesian global optimization. 
        Some common methods include:
        - 'EHVI': Expected Hypervolume Improvement
        - 'PI': Probability of Improvement
    :param assign_model: bool or str, optional, default=False
        If `assign_model` is False, the surrogate model is chosen automatically by MultiBgolearn.
        If `assign_model` is one of the following strings, the corresponding model is assigned:
        - 'RandomForest'
        - 'GradientBoosting'
        - 'LinearRegression'
        - 'Lasso'
        - 'Ridge'
        - 'SVR'
        - 'GaussianProcess'
    :param bootstrap: int, optional, default=5
        The number of bootstrap iterations to be used for uncertainty quantification in the model predictions.
    
    :return: tuple
        Returns the recommended data point from the virtual space, along with the corresponding improvement values 
        and the index of the recommended data in the virtual space.
        - VS[res_index]: The recommended data point.
        - improvements: The calculated improvements based on the optimization method.
        - res_index: The index of the recommended data point in the virtual space.

    Example usage:
        VS_recommended, improvements, index = fit('./data/dataset.csv', './virtual_space/',3,  max_search=True, method='EHVI', bootstrap=5)

    Notes:
    - The method selected will guide how the optimization balances the different objectives.
    =======================================================================
    """
    
    # Read the dataset, extracting features (X) and objectives/response variables (y)
    X, ori_y = read_in(dataset, object_num)
    ori_VS = read_in_vs(VSdataset)

    # Normalize data
    X, y, VS = normalize_data(X, ori_y, ori_VS)

    if assign_model == False:
        # Build the surrogate models based on the provided data
        best_model = build_models(X, y,ori_y)
    else: best_model = assign_model

    # Use the trained models to predict the mean and variance of the virtual space (VS)
    vs_mean, vs_vars = pre_model(X, y, VS, best_model, bootstrap)

    # Perform multi-objective Bayesian global optimization (MOBO) to get the next recommended data point
    res_index, improvements = multi_BGO(y, vs_mean, vs_vars, max_search, method)

    # Print the recommended data point and return it along with improvements and its index in VS
    print(f"The next recommended data by method '{method}' is: {np.array(ori_VS)[res_index]}")
    
    return np.array(ori_VS)[res_index], improvements, res_index







