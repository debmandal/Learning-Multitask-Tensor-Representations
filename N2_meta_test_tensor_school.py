import numpy as np
import argparse
import pickle
from tensorly.decomposition import parafac
from tensorly.tenalg import multi_mode_dot
from tensorly.tenalg import mode_dot
from tensorly.tenalg import khatri_rao
from algo1 import algo1
from algo2 import algo2
from generate_data import generate_training_data
from generate_data import generate_responses
from generate_data import generate_A_tensor

def least_squares(A1, A2, X, Y0, R, r):
    # Get the CP decomposition of A
    #W, factors = parafac(A, r, normalize_factors=True)
    #print(W)
    #A1 = factors[0]
    #A2 = factors[1]
    #A3 = factors[2]

    # Construct \hat{V}
    Y_prod = Y0.T @ A2
    Y_prod = np.reshape(Y_prod, (Y_prod.shape[0], 1)).T
    X_prod = X @ A1
    kr_prod = khatri_rao([Y_prod, X_prod])
    #V = kr_prod @ np.diag(W)

    #inverse_term = (A3 @ V.T) @ (V @ A3.T)
    wt = np.linalg.pinv(kr_prod) @ R
    # Z = np.linalg.pinv(inverse_term) @ (A3 @ V.T @ R)
    return wt


if __name__ == '__main__':
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", help="Std. dev. of the noise.")
    parser.add_argument('--output_dir', help="Path to output the generated tensors to.")
    parser.add_argument('--seed', help="Seed for the randomness of data generation.")
    parser.add_argument('--A_and_task_dir', help="Path for the underlying tensor A and Y0, Z0.")
    parser.add_argument('--num_trials', help="Number of Trials.")
    parser.add_argument('--method', help="Type of method: I (Tensor Regression), II (Method of Moments)")
    parser.add_argument("--iters", help="Number of iterations for grad. desc.")
    parser.add_argument("--lambd", help="Value of hyperparameter lambda.")
    parser.add_argument('--eta', help="Eta (learning rate) parameter.")
    parser.add_argument("--r", help="CP Rank of the underlying tensor A.")
    # Parse args (otherwise set defaults)
    args = parser.parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "meta_test_results/school/"
    if args.A_and_task_dir:
        A_and_task_dir = args.A_and_task_dir
    else:
        A_and_task_dir = "data/school/"
    if args.seed:
        seed = int(args.seed)
    else:
        seed = 11
    if args.num_trials:
        num_trials = int(args.num_trials)
    else:
        num_trials = 20
    if args.method:
        method = args.method
    else:
        method = 'I'
    if args.sigma:
        sigma = float(args.sigma)
    else:
        sigma = 1.0
    if args.iters:
        iterations = int(args.iters)
    else:
        iterations = 200
    if args.lambd:
        lambd = float(args.lambd)
    else:
        lambd = 0.01
    if args.eta:
        eta = float(args.eta)
    else:
        eta = 0.1
    if args.r:
        r = int(args.r)
    else:
        r = 15

    #run tensor regression to get A
    X = pickle.load(open('./data/school/X.pkl', 'rb'))
    Y = pickle.load(open('./data/school/Y.pkl', 'rb'))
    R = pickle.load(open('./data/school/R.pkl', 'rb'))
    task_function = pickle.load(open('./data/school/task_function.pkl', 'rb'))
    
    task_function = task_function.astype('int')
    #print(task_function)
    print(X.shape)
    print(Y.shape)
    print(R.shape)
    #run algo1
    d1 = 26
    d2 = 50
    T = 50

    if method == 'I':
        Y_ti = Y[task_function]
        cov_X = np.einsum('bi,bo->bio', X, Y_ti)
        # Perform algorithm 1 to get estimated A
        eps = 0.01
        est_A1, est_A2 = algo1(d1, d2, T, R, X, Y, cov_X, eta, eps, r, lambd, task_function, iterations)
    
        print('Done running Tensor Regression')
    elif method == 'II':
        est_A1, est_A2 = algo2(R, X, Y, task_function, r)
        
    mse_all = np.zeros((num_trials,9))
    

    for trial in range(num_trials):
        for N2 in range(20,200,20):
            


            #load data
            X2 = pickle.load(open('./data/school/' + 'X0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'rb'))
            Y0 = pickle.load(open('./data/school/' + 'Y0.pkl','rb'))
            #Z0 = pickle.load(open('./data/school/' + 'Z0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'rb'))
            R = pickle.load(open('./data/school/' + 'R0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'rb'))

            # Need to get A^3^TZ_0 from A to perform least squares on new task
            est_wt = least_squares(est_A1, est_A2,  X2, Y0, R, r)
            #print(np.sum(np.square(est_wt - np.matmul(factors[2].T , Z0))))

            # load evaluation dataset
            X0eval = pickle.load(open('./data/school/X0eval.pkl','rb'))
            R0eval = pickle.load(open('./data/school/X0eval.pkl','rb'))

            # Find avg. error over all X
            Y_prod = Y0.T @ est_A2
            Y_prod = np.reshape(Y_prod, (Y_prod.shape[0], 1)).T
            X_prod = np.matmul(X0eval, est_A1)
            kr_prod = khatri_rao([Y_prod, X_prod])
            est_R = kr_prod @ est_wt
            
            MSE = np.sum(np.square(R0eval - est_R))
            MSE = MSE / X0eval.shape[0]
            mse_all[trial][(N2-20)//20] = MSE

    avgerr = np.mean(mse_all, axis=0)
    stderr = np.std(mse_all, axis=0)/num_trials

    print('(' + str(avgerr[0]) , end = '')
    for i in range(1,9):
        print(", " + str(avgerr[i]), end='')
    print(')')

    print('(' + str(stderr[0]) , end = '')
    for i in range(1,9):
        print(", " + str(stderr[i]), end='')
    print(')')
    


