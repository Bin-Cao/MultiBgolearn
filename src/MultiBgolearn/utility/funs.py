import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import lightgbm as lgb
from sklearn.utils import resample
from scipy.stats import multivariate_normal
import os
import pygmo as pg
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut

def normalize_data(X, y, VS):
    """
    Normalize X, y, and VS.
    X and VS will be merged, normalized using the same scaler, and then split back.
    y is normalized independently.

    Parameters:
    X (array-like): Feature matrix
    y (array-like): Target values
    VS (array-like): Validation set

    Returns:
    X_normalized, y_normalized, VS_normalized
    """
    # Concatenate X and VS along the first axis (rows)
    combined = np.vstack((X, VS))
    
    # Normalize the combined data
    scaler = MinMaxScaler()
    combined_normalized = scaler.fit_transform(combined)
    
    # Split the normalized combined data back into X_normalized and VS_normalized
    X_normalized = combined_normalized[:X.shape[0], :]
    VS_normalized = combined_normalized[X.shape[0]:, :]
    
    # Independently normalize y
    y_normalized = (y - y.min()) / (y.max() - y.min())
    
    return X_normalized, y_normalized, VS_normalized



def read_in(dataset, object_num):
    """
    Reads a dataset from a csv or excel file and separates it into features (X) and targets (y).
    
    :param dataset: str, the path to the dataset file (can be .csv or .xlsx)
                    i.e., './data/dataset.csv'
    :param object_num: int, the number of target columns, i.e., 3
    :return: X (features), y (targets)
    """
    # Check file extension and read accordingly
    if dataset.endswith('.csv'):
        df = pd.read_csv(dataset)

    elif dataset.endswith('.xlsx') or dataset.endswith('.xls'):
        df = pd.read_excel(dataset)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    
    # X contains all columns except the last 'object_num' columns
    X = df.iloc[:, :-object_num]
    
    # y contains the last 'object_num' columns
    y = df.iloc[:, -object_num:]
    
    return X, y

def read_in_vs(VSdataset):
    if VSdataset.endswith('.csv'):
        df = pd.read_csv(VSdataset)
    elif VSdataset.endswith('.xlsx') or VSdataset.endswith('.xls'):
        df = pd.read_excel(VSdataset)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    
    # Check if the dataframe has more than 20,000 rows
    if len(df) > 20000:
        #print('Too many candidates, only 20,000 candidates will be considered in the system.')
        df = df.sample(n=20000, random_state=42)  # Random sampling with a fixed seed
    
    return df


def build_models(X, y, ori_y):
    """
    Builds and evaluates multiple multi-target regression models using Leave-One-Out Cross-Validation (LOOCV).
    Recommends the best model based on R2 score.

    :param X: Features (multi-columns) as a DataFrame or ndarray
    :param y: Normalized targets (multi-columns) as a DataFrame or ndarray
    :param ori_y: Original targets before normalization as a DataFrame
    :return: The best model based on R2 score
    """
    
    # Define models to test
    models = {
        'RandomForest': MultiOutputRegressor(RandomForestRegressor()),
        'GradientBoosting': MultiOutputRegressor(GradientBoostingRegressor()),
        'LinearRegression': MultiOutputRegressor(LinearRegression()),
        'Lasso': MultiOutputRegressor(Lasso()),
        'Ridge': MultiOutputRegressor(Ridge()),
        'SVR': MultiOutputRegressor(SVR()),
        'GaussianProcess': MultiOutputRegressor(GaussianProcessRegressor(alpha=0.1))
    }
    
    print('LOOCV is used to evaluate models, and the best one will be recommended after evaluation.')

    # Dictionary to store R2 scores for each model
    r2_scores = {}

    # Iterate through each model, train it, and evaluate performance
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Initialize Leave-One-Out Cross-Validation (LOOCV)
        loo = LeaveOneOut()
        y_pred_list = []
        
        # Perform LOOCV
      
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

            # Fit the model
            model.fit(X_train, y_train)

            # Predict for the test instance
            y_pred_list.append(model.predict(X_test))

        # Convert predictions to array
        y_pred = np.vstack(y_pred_list)  # Stack predictions to get full array

        # Calculate R2 score
        r2 = r2_score(y, y_pred, multioutput='uniform_average')
        r2_scores[model_name] = r2
        print(f"R2 score for {model_name}: {r2:.4f}")
        
        # Create a directory for each model's results
        os.makedirs(f'./MultiBgolearn/{model_name}', exist_ok=True)

        # Save prediction results
        np.savetxt(f'./MultiBgolearn/{model_name}/predictions.csv', map_bac(y_pred, ori_y), delimiter=',')
        
        # Generate scatter plots for each target
        for i in range(y.shape[1]):
            plot_performance(y.iloc[:, i], y_pred[:, i], ori_y.iloc[:, i], model_name, target_index=i)
    
    # Recommend the best model based on R2 score
    best_model = max(r2_scores, key=r2_scores.get)
    print(f"The best model is {best_model} with an R2 score of {r2_scores[best_model]:.4f}")
    return best_model
 
def map_bac(y_pred, ori_y):
    """
    Rescales the predicted values `y_pred` back to the original range of `ori_y` for each target.

    :param y_pred: np.ndarray, predicted values with the same shape as ori_y (n_samples, n_targets)
    :param ori_y: pd.DataFrame, original target values (n_samples, n_targets)
    
    :return: np.ndarray, rescaled predicted values
    """
    _pre = copy.deepcopy(y_pred)
    # Rescale each target column of y_pred to the original range of ori_y
    for i in range(_pre.shape[1]):
        # Get the min and max values for the original target
        ori_min = ori_y.iloc[:, i].min()
        ori_max = ori_y.iloc[:, i].max()
        
        # Rescale y_pred values for the i-th target to the original range
        _pre[:, i] = _pre[:, i] * (ori_max - ori_min) + ori_min
    
    return _pre
    
def plot_performance(_y, _y_pred,ori_y, model_name, target_index):

    """
    Plots the predicted vs true values for a specific target and saves the plot.

    :param _y: true target values (1D for a specific target)
    :param _y_pred: predicted target values (1D for a specific target)
    :param ori_y : targets before normalized
    :param model_name: name of the model
    :param target_index: index of the target being plotted
    """
    plt.figure(figsize=(8, 8))
    sns.set(style="whitegrid")

    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 20,
        'axes.titlesize': 24,
        'legend.fontsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'axes.linewidth': 2,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
        'axes.grid': True,
        'grid.alpha': 0.8,
        'grid.linestyle': '--',
        'grid.linewidth': 1,
        'figure.figsize': (12, 12)
    })

    y = _y  * (ori_y.max() - ori_y.min()) + ori_y.min()
    y_pred= _y_pred  * (ori_y.max() - ori_y.min()) + ori_y.min()

    # Scatter plot of true vs predicted values for the specific target
    sns.scatterplot(x=y, y=y_pred, marker='o', s=150, edgecolor='k', color='#1f77b4', alpha=0.8)

    gap = (y.max() - y.min()) * 0.05
    # Plot y=x reference line
    plt.plot([y.min() - gap, y.max() + gap], [y.min() - gap, y.max() + gap], '--', color='k', linewidth=2.5)

    # Set title and labels
    plt.title(f'{model_name} Target {target_index+1} Predictions vs True Values', fontsize=14)
    plt.xlabel('True Values', fontsize=20)
    plt.ylabel('Predicted Values', fontsize=20)

    # Add text annotations for Correlation Coefficient and RMSE
    plt.text(0.05, 0.95, 'Correlation Coefficient: {:.2f}'.format(np.corrcoef(y, y_pred)[0, 1]), 
             fontsize=18, color='black', transform=plt.gca().transAxes)
    plt.text(0.05, 0.9, 'RMSE: {:.2f}'.format(np.sqrt(mean_squared_error(y, y_pred))), 
             fontsize=18, color='black', transform=plt.gca().transAxes)

    # Set axis limits and aspect ratio
    plt.xlim(y.min() - gap, y.max() + gap)
    plt.ylim(y.min() - gap, y.max() + gap)
    plt.gca().set_aspect('equal', adjustable='box')

    # Save plot as an image
    plt.savefig(f'./MultiBgolearn/{model_name}/scatter_target_{target_index+1}.png', dpi=500, bbox_inches='tight')




def pre_model(X, y, VS, model_name, bootstrap_times=5):
    """
    Predicts using the specified model. If it's a Gaussian Process, it directly returns
    the mean and standard deviation of the predictions. For other models, it uses 
    bootstrapping to return the mean and variance of the predictions.

    :param X: features (multi-columns) for training
    :param y: targets (multi-columns) for training
    :param VS: validation set features (multi-columns) to predict
    :param model_name: the name of the model to use (should match keys in the `models` dict)
    :param bootstrap_times: number of bootstrap samples to use for non-Gaussian models
    :return: vs_mean, vs_vars - mean and variance of predictions
    """
    
    # Dictionary of models to choose from
    models = {
        'RandomForest': MultiOutputRegressor(RandomForestRegressor()),
        'GradientBoosting': MultiOutputRegressor(GradientBoostingRegressor()),
        'LinearRegression': MultiOutputRegressor(LinearRegression()),
        'Lasso': MultiOutputRegressor(Lasso()),
        'Ridge': MultiOutputRegressor(Ridge()),
        'SVR': MultiOutputRegressor(SVR()),
        'GaussianProcess': MultiOutputRegressor(GaussianProcessRegressor(alpha=0.1))
    }
    
    # Select the model from the dictionary
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found in available models: {list(models.keys())}")
    else:
        print(f"Model {model_name} is applied as an available model")

    best_model = models[model_name]
    
    if isinstance(best_model.estimator, GaussianProcessRegressor):
        # If the model is Gaussian Process
        y_pred, y_std = best_model.predict(VS, return_std=True)
        vs_mean = y_pred
        vs_vars = y_std ** 2  # variance is the square of the standard deviation
    else:
        # For other models, apply bootstrap sampling
        y_preds = []
        for i in range(bootstrap_times):
            # Resample the training data with replacement for bootstrapping
            X_bootstrap, y_bootstrap = resample(X, y)
            best_model.fit(X_bootstrap, y_bootstrap)
            y_pred = best_model.predict(VS)
            y_preds.append(y_pred)

        # Stack predictions and compute mean and variance
        y_preds = np.stack(y_preds, axis=0)
        vs_mean = np.mean(y_preds, axis=0)
        vs_vars = np.var(y_preds, axis=0)

    return vs_mean, vs_vars







def Monte_Carlo(y, vs_mean, vs_var, times=10):
    """
    Perform Monte Carlo sampling from a multivariate Gaussian distribution 
    with given mean and variance.

    :param y: current targets
    :param vs_mean: mean of virtual data point
    :param vs_var: variance of virtual data point
    :param times: number of Monte Carlo samples
    :return: list of sampled points
    """
    samples = []
    for _ in range(times):
        y_sample = multivariate_normal.rvs(mean=vs_mean, cov=np.diag(vs_var))
        samples.append(y_sample)
    return samples




def calculate_lebesgue_measure(pareto_front, max_search):
    """
    Calculate the Lebesgue measure (hypervolume) of the Pareto front.

    :param pareto_front: numpy array of Pareto front points
    :param max_search: boolean indicating if the problem is maximization or minimization
    :return: Lebesgue measure (hypervolume)
    """
    pareto_front = np.array(pareto_front)
    
    if pareto_front.size == 0:
        raise ValueError("Pareto front is empty.")
    
    # Determine the reference point based on max_search
    if max_search:
        pareto_front *= -1
   
    ref_point = np.max(pareto_front, axis=0) + 0.1
         
    hv = pg.hypervolume(pareto_front)
    hypervolume = hv.compute(ref_point)
    
    return hypervolume



def get_pareto_front(y, max_search):
    """
    Given a set of points, return the points that are on the Pareto front.

    :param y: numpy array of target points (2D array where each row is a point)
    :param max_search: boolean indicating the direction of optimization. 
                       True for maximization, False for minimization.
    :return: Pareto front points as a numpy array
    """
    y = np.array(y)  # Ensure y is a numpy array

    # Convert the problem to maximization if it's a minimization problem
    if not max_search:
        y = -y  # Negate the values to convert minimization to maximization

    pareto_front = []
    num_points = y.shape[0]

    for i in range(num_points):
        point = y[i]
        dominated = False
        for j in range(num_points):
            if i != j:
                other_point = y[j]
                if all(other_point >= point) and any(other_point > point):
                    dominated = True
                    break
        if not dominated:
            pareto_front.append(point)

    pareto_front = np.array(pareto_front)

    # If it was a minimization problem, negate the points back to original
    if not max_search:
        pareto_front = -pareto_front

    return pareto_front
