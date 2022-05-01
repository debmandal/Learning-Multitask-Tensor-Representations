import numpy as np
import argparse
import pickle
from tensorly.decomposition import parafac
from tensorly.tenalg import multi_mode_dot
from tensorly.tenalg import mode_dot
from tensorly.tenalg import khatri_rao
from N2_meta_test_matrix_synthetic import MetaLR_w_MOM, gen_train_model, LR
from generate_data import generate_training_data
from generate_data import generate_responses
from generate_data import generate_A_tensor
from tensorly import unfold

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
    parser.add_argument('--output_dir', help="Path to output the generated tensors to.")
    parser.add_argument('--seed', help="Seed for the randomness of data generation.")
    parser.add_argument('--A_and_task_dir', help="Path for the underlying tensor A and Y0, Z0.")
    parser.add_argument('--num_trials', help="Number of Trials.")
    parser.add_argument('--method', help="Method (Type of matrix model).")
    parser.add_argument("--r", help="CP Rank of the underlying tensor A.")
    # Parse args (otherwise set defaults)
    args = parser.parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "meta_test_results/synthetic"
    if args.A_and_task_dir:
        A_and_task_dir = args.A_and_task_dir
    else:
        A_and_task_dir = "data/synthetic/"
    if args.seed:
        seed = args.seed
    else:
        seed = 42
    if args.num_trials:
        num_trials = args.num_trials
    else:
        num_trials = 20
    if args.method:
        method = args.method
    else:
        method = 'I'
    if args.r:
        r = int(args.r)
    else:
        r = 15

    X = pickle.load(open('./data/school/X.pkl', 'rb'))
    Y = pickle.load(open('./data/school/Y.pkl', 'rb'))
    R = pickle.load(open('./data/school/R.pkl', 'rb'))
    task_function = pickle.load(open('./data/school/task_function.pkl', 'rb'))
    
    task_function = task_function.astype('int')
    #print(task_function)

    #run algo1
    d1 = 26
    d2 = 50
    T = 50

    Y_ti = Y[task_function]
    cov_X = np.einsum('bi,bo->bio', X, Y_ti)



    #est_A1, est_A2 = algo2(R, X, Y, task_function, r)
    if method == 'I': #dimension of recovered matrix is d1d2xd3
        cov_X = np.einsum('bi,bo->bio', X, Y_ti)
        #cov_X = np.concatenate((X, Y_ti), axis=1)
        cov_X = unfold(cov_X,0)

        # Now we can generate the training data
        # In our paper, alphas are equivalent to the Z vectors in d3
        train_data = gen_train_model(cov_X, R, task_function,T)
        # get B using MoM method from TJJ
        B = MetaLR_w_MOM(train_data, r )
        #B = MetaLR_w_FO(train_data, d3 )
    elif method == 'II': #dimension of recovered matrix is (d1+d2)xd3
        cov_X = np.concatenate((X,Y_ti), axis=1)
        train_data = gen_train_model(cov_X, R, task_function,T)
        B = MetaLR_w_MOM(train_data, r )

    
    mse_all = np.zeros((num_trials,9))

    for trial in range(num_trials):
        for N2 in range(20,200,20):
            #generate data
            #X2, Y0, Z0, R = generate_new_task(d1, d2, d3, N2,A)
            #save data
            #pickle.dump(X2, open(output_dir + 'X2_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'wb'))
            #pickle.dump(Y0, open(output_dir + 'Y0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'wb'))
            #pickle.dump(Z0, open(output_dir + 'Z0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'wb'))
            #pickle.dump(R, open(output_dir + 'R_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'wb'))

            #load data
            X2 = pickle.load(open('./data/school/' + 'X0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'rb'))
            Y0 = pickle.load(open('./data/school/' + 'Y0.pkl','rb'))
            #Z0 = pickle.load(open('./data/school/' + 'Z0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'rb'))
            R = pickle.load(open('./data/school/' + 'R0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'rb'))

            if method == 'I':
                cov_X0 = np.einsum('bo,i->boi', X2, Y0)
                cov_X0 = unfold(cov_X0, 0)
            elif method == 'II':
                cov_X0 = np.concatenate((X2,np.tile(Y0, (N2,1))), axis=1)
            elif method == 'III':
                cov_X0 = X2
            #get estimate of Z0

            X_low = cov_X0 @ B
            alpha_LR = LR((X_low, R))
            beta_LR = B @ alpha_LR
            

            # load evaluation dataset
            X0eval = pickle.load(open('./data/school/X0eval.pkl','rb'))
            R0eval = pickle.load(open('./data/school/X0eval.pkl','rb'))
            #estimate with beta_LR
            if method == 'I':
                cov_Xtest = np.einsum('bo,i->boi', X0eval, Y0)
                cov_Xtest = unfold(cov_Xtest, 0)
            elif method =='II':
                cov_Xtest = np.concatenate((X0eval,np.tile(Y0, (X0eval.shape[0], 1))), axis=1)
            elif method == 'III':
                cov_Xtest = X0eval
            est_R = cov_Xtest @ beta_LR
            
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