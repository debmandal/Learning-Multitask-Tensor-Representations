import argparse
import numpy as np
import pickle
from tensorly.decomposition import parafac
from tensorly.tenalg import multi_mode_dot
from tensorly.tenalg import mode_dot
from tensorly.tenalg import khatri_rao
from generate_data import generate_training_data
from generate_data import generate_responses
from generate_data import generate_A_tensor
from numpy.linalg import norm
import tensorly as tl

def generate_new_task(d1, d2, d3, N2, A, seed=42):
    # Seed the randomness

    # Generate user feature vectors X
    user_mu = 0
    user_sigma = 1/np.sqrt(d1)
    X = user_sigma * np.random.randn(N2, d1) + user_mu # From N(0, 1/sqrt(d1))

    # Generate test task (Y, Z)
    Y_mu = 0
    Y_sigma = 1/np.sqrt(d2)
    Z_mu = 0
    Z_sigma = 1/np.sqrt(d3)
    Y0 = Y_sigma * np.random.randn(d2) + Y_mu
    Z0 = Z_sigma * np.random.randn(d3) + Z_mu

    # Generate the responses
    noise = np.random.normal(0, 1, (N2))
    R = [multi_mode_dot(A, [X[u], Y0, Z0], modes=[0,1,2]) for u in range(N2)]
    R = np.asarray(R) + noise
    return X, Y0, Z0, R


if __name__ == "__main__":
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", help="First dimension (d1).")
    parser.add_argument("--d2", help="Second dimension (d2).")
    parser.add_argument("--d3", help="Third dimension (d3).")
    parser.add_argument("--N", help= "Number of users/examples (N).")
    parser.add_argument("--T", help="Number of tasks (T).")
    parser.add_argument("--r", help="CP Rank of the underlying tensor A.")
    parser.add_argument('--seed', help="Seed for the randomness of data generation.")
    parser.add_argument('--save_data', help="Specify where you would like to save the data.")
    parser.add_argument('--num_trials', help="Number of Trials.")

    # Parse args (otherwise set defaults)
    args = parser.parse_args()
    if args.d1:
        d1 = int(args.d1)
    else:
        d1 = 100
    if args.d2:
        d2 = int(args.d2)
    else:
        d2 = 50
    if args.d3:
        d3 = int(args.d3)
    else:
        d3 = 50
    if args.N:
        N = int(args.N)
    else:
        N = 1000
    if args.T:
        T = int(args.T)
    else:
        T = 100
    if args.r:
        r = int(args.r)
    else:
        r = 10
    if args.seed:
        seed = args.seed
    else:
        seed = 42
    if args.save_data:
        save_data = args.save_data
    else:
        save_data = './data/synthetic/'
    if args.num_trials:
        num_trials = int(args.num_trials)
    else:
        num_trials = 20
    #if args.load_data:
    #    A = pickle.load(open(args.load_data + "A.pkl", "rb"))
    #    X = pickle.load(open(args.load_data + "X.pkl", "rb"))
    #    Y = pickle.load(open(args.load_data + "Y.pkl", "rb"))
    #    Z = pickle.load(open(args.load_data + "Z.pkl", "rb"))   # NOTE: the contents of Z are never actually used, only shape
    #    R = pickle.load(open(args.load_data + "R.pkl", "rb"))
    #    task_function = pickle.load(open(args.load_data + "task_function.pkl", "rb"))
    #else:
    print("Generating synthetic data...")
    # Generate A tensor
    A = generate_A_tensor(d1, d2, d3, r)

    # Generate synthetic training data
    X, Y, Z = generate_training_data(d1, d2, d3, N, T)
    noise = np.random.normal(0, 1, (N, T))
    R = generate_responses(A, X, Y, Z, T)
    R += noise

    task_function = np.random.randint(0, T, size=N)
    R = [R[i][task_function[i]] for i in range(N)]
    
    pickle.dump(A, open(save_data + "A.pkl", "wb"))
    pickle.dump(X, open(save_data + "X.pkl", "wb"))
    pickle.dump(Y, open(save_data + "Y.pkl", "wb"))
    pickle.dump(Z, open(save_data + "Z.pkl", "wb"))
    pickle.dump(R, open(save_data + "R.pkl", "wb"))
    pickle.dump(task_function, open(save_data + "task_function.pkl", "wb"))

     #generate test-data for different N2 sizes and trials
    output_dir = save_data
    for trial in range(num_trials):
        for N2 in range(20,220,20):
            #generate data
            X2, Y0, Z0, R = generate_new_task(d1, d2, d3, N2,A)
            #save data
            pickle.dump(X2, open(output_dir + 'X2_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'wb'))
            pickle.dump(Y0, open(output_dir + 'Y0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'wb'))
            pickle.dump(Z0, open(output_dir + 'Z0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'wb'))
            pickle.dump(R, open(output_dir + 'R_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'wb'))
   

    # ALGO 1
    #print("Executing Algorithm 1...")
    #true_B = mode_dot(A, Z, mode=2)
    

    #save est_A
    #pickle.dump(est_A1, open(estimated_data + "est_A1.pkl", "wb"))

    # ALGO 2
    #print("Executing Algorithm 2...")

    #save est_A2
    #pickle.dump(est_A2, open(estimated_data + "est_A2.pkl", "wb"))

    # Save A, est_A, X, Y, Z, task_function, and R

    

#    true_B = mode_dot(A, Z, mode=2)
#    Y_ti = Y[task_function]
#    cov_X = np.einsum('bi,bo->bio', X, Y_ti)

    # Perform algorithm 1 to get estimated A
#    eps = 0.01
#    B, A1, A2 = algo1(true_B, A, R, X, Y, Z, cov_X, T, eta, eps, r, lambd, task_function, iterations)


    #N,_ = X.shape
    #_,d3 = Z.shape
    #Ri = [R[i][task_function[i]] for i in range(N)]
    #A1, A2 = algo2(Ri, X, Y, task_function, r, d3, A)
    #save est_A2
#    pickle.dump(A1, open(A_and_task_dir + "est_A1.pkl", "wb"))
#    pickle.dump(A2, open(A_and_task_dir + "est_A2.pkl", "wb"))

       # Save A, est_A, X, Y, Z, task_function, and R

