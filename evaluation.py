import nevergrad as ng
import ioh
from nevergrad.optimization.optimizerlib import (
    CMA,
    BFGS, 
    Cobyla, 
    PSO, 
    DE,
)
import iohinspector
import polars as pl
from scipy.stats import rankdata
import warnings

class NG_Evaluator():
    def __init__(self, optimizer, budget: int = 2000):
        self.alg = optimizer
        self.budget = budget
    
    def __call__(self, func):
        parametrization = ng.p.Array(shape=(func.meta_data.n_variables,)).set_bounds(-5, 5)
        optimizer = eval(f"{self.alg}")(
            parametrization=parametrization, budget=int(self.budget)
        )
        optimizer.minimize(func)

algs = ['CMA', 'DE', 'PSO', 'BFGS', "Cobyla"]

def run_benchmark(problem_suite, budget = 5000):
    for name, problem in problem_suite.items():
        for alg in algs:
            logger = ioh.logger.Analyzer(root = "data", folder_name=f"{name}_{alg}_logs", algorithm_name=alg)
            problem.attach_logger(logger)
            optimizer = NG_Evaluator(alg, budget)
            for _ in range(5):
                optimizer(problem)
                problem.reset()
            logger.close()
        
def get_Friedman_val(dt_perf):
    # Get friedman ranks for each function
    friedman_ranks = []

    for func_name in dt_perf['function_name'].unique():
        func_data = dt_perf.filter(pl.col('function_name') == func_name)
        
        # Collect all scores for this function organized by run
        max_runs = func_data.group_by('algorithm_name').agg(pl.col('best_y').count()).select(pl.col('best_y').max())[0,0]
        
        # For each run, rank algorithms
        for run_idx in range(max_runs):
            run_scores = []
            run_algs = []
            
            for alg_name in func_data['algorithm_name'].unique():
                alg_scores = func_data.filter(pl.col('algorithm_name') == alg_name)['best_y'].to_numpy()
                if run_idx < len(alg_scores):
                    run_scores.append(alg_scores[run_idx])
                    run_algs.append(alg_name)
            
            # Rank algorithms for this run (lower score = lower rank = better)
            ranks = rankdata(run_scores, method='average')
            
            for alg_name, rank in zip(run_algs, ranks):
                friedman_ranks.append({
                    'function_name': func_name,
                    'algorithm_name': alg_name,
                    'run': run_idx,
                    'rank': rank
                })

    # Average ranks per algorithm per function
    friedman_ranks_df = pl.DataFrame(friedman_ranks)
    friedman_avg_ranks = friedman_ranks_df.group_by(['function_name', 'algorithm_name']).agg(
        pl.col('rank').mean().alias('rank')
    )
    friedman_iqr = friedman_avg_ranks.group_by('algorithm_name').agg(
        pl.col('rank').quantile(0.75).alias('q75'),
        pl.col('rank').quantile(0.25).alias('q25')
    ).with_columns(
        (pl.col('q75') - pl.col('q25')).alias('iqr')
    )
    return friedman_iqr['iqr'].mean()


def evaluate_suite(suite):
    run_benchmark(suite, budget=5000)
    dm = iohinspector.DataManager()
    dm.add_folder("data")
    dt_perf = dm.overview[['algorithm_name', 'function_name', 'best_y']]
    # as_val = get_AS_val(dt_perf)
    friedman_val = get_Friedman_val(dt_perf)
    return friedman_val

def create_suite(problems, meta_dims):
    probs_ioh = {}
    for k in problems.keys():
        probs_ioh[k] = ioh.wrap_problem(problems[k], k, ioh.ProblemClass.REAL, meta_dims[k], lb=-5, ub=5)
    return probs_ioh

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Warning)
    from problem_suite import problems, meta_dims
    import os

    if os.path.exists("data") and os.listdir("data"):
        raise RuntimeError("The 'data' directory already exists and contains files. Please remove it before running.")
    
    if len(problems) != 25:
        print("Warning: Number of problems is not 25")

    try:
        suite = create_suite(problems, meta_dims)
        v1 = evaluate_suite(suite)
        print(f"Performance value: {v1} (to be maximized)")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")