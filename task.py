import pandas as pd
import numpy as np


class matrix: 
     
                    def __init__(self,filename):
                        # Initialize the matrix class with either a filename or a NumPy array.
                        if isinstance(filename, str):  # If data is a string (filename)
                            self.load_from_csv(filename)
                        elif isinstance(filename, np.ndarray):  # If data is already a NumPy array
                            self.array_2d = filename
            
                    # class methods
                    
                    def load_from_csv(self, filename): 
                        try:
                            df = pd.read_csv(filename,header =None)  
                            self.array_2d = df.to_numpy()   # converting the data into array_2d matrix
                            
                        except Exception as e:
                            print(e)
                            
                            


                    def standardise(self):
                                
                        # Initializing an empty matrix to hold the standardized values of the original matrix
                        D_std = np.zeros(self.array_2d.shape)
                        # Iterate over each column of original matrix and calculate the column avg, max and min values
                        for j in range(self.array_2d.shape[1]): # here shape of 2d array is (178,13). Thus selecting 13 columns
                                # Calculating the column Average, max, and min
                                avg = np.mean(self.array_2d[:, j])  # [:,j] -> selects all the rows and j columns
                                maxi = np.max(self.array_2d[:, j])
                                mini = np.min(self.array_2d[:, j])
                            # standardized values are replaced columnwise
                                
                                for i in range(self.array_2d.shape[0]):
                                        D_std[i, j] = (self.array_2d[i, j] - avg) / (maxi - mini)
                    
                        return matrix(D_std) #returning the output as an object of the class matrix
                    
                    
                    
                    
                    def get_distance(self,other_matrix,row_i):
                        y = self.array_2d[row_i] # getting the row_i of the matrix calling this method
                        distance_matrix = []
                        # calculating the euclidean distance between given matrix and other matrix
                        for x in other_matrix:
                                distance = np.sum((x - y)**2)
                                distance_matrix.append(distance)
                        dist =  np.array(distance_matrix).reshape(-1,1)
                        return matrix(dist)
                    
                    
                    
                    
                    def get_weighted_distance(self,other_matrix,weights,row_i):
                        y = self.array_2d[row_i] # getting the row_i of the matrix calling this method
                        distance_matrix = []
                        # calculating the weighted euclidean distance between data matrix and other matrix
                        for x in other_matrix:
                                distance = np.sum(weights * ((x - y)**2))
                                distance_matrix.append(distance)
                        dist =  np.array(distance_matrix).reshape(-1,1)
                        return matrix(dist)
                    
                    
                    
                    
                    def get_count_frequency(self):
                        if self.array_2d.shape[1] != 1:  # checking if the matrix calling this method has only a single column
                            return 0
                        else:
                            count_dictionary = {}   # initializing an empty dictionary
                            one_dim_array = self.array_2d.flatten()
                            for i in one_dim_array:
                                if i not in count_dictionary:
                                        count_dictionary[i] = 1  # if the element is already not in the dict, it creates a new key:value pair 
                                else:                        # with element and its frequency of occurence
                                        count_dictionary[i] += 1 # if element is already there in dict, it increments the count by 1
                        return count_dictionary
        
# Functions

        
def get_initial_weights(m):
    np.random.seed(42) # setting the seed, to reproduce the same output every time
    random_numbers = np.random.rand(1,m)  # rand() method creates a 2D array (1 row, m columns) filled with random values between 0 & 1
    initial_weights = random_numbers/np.sum(random_numbers) # normalizing by didving every element in the matrix with the total sum
    return initial_weights



def get_centroids(data_matrix,S,K):
    n, m = data_matrix.array_2d.shape  # Number of rows and columns in the data matrix
    centroids = np.empty((K, m)) # create an empty matrix of dimension(K,m)
    for k in range(K):
        indices = np.where(S == k) # Get all row indices from S matrix where the corresponding row in S equals k 
        # get the mean of corresponding rows(according to row indices) from data matrix and update the centroids matrix
        centroids[k] = np.mean(data_matrix.array_2d[indices[0]],axis=0) 
    return centroids



    
def get_separation_within(data_matrix,S,centroids,K):
    m = data_matrix.array_2d.shape[1]
    a_j = np.zeros((1, m))
    for k in range(K):
        cluster_indices = np.where(S.array_2d == k)[0]
        for j in range(m):
            sum_distances = np.zeros((1,m))
            for i in cluster_indices:
                    distance = data_matrix.get_distance(centroids[k],i)
                    sum_distances += np.sum(distance.array_2d.reshape(1,m),axis=0)
        a_j += sum_distances
    return a_j
 


def get_separation_between(data_matrix,S,centroids,K):
    m = data_matrix.array_2d.shape[1]
    b_j = np.zeros((1, m))
    overall_means = matrix(np.mean(data_matrix.array_2d,axis=0))
    for k in range(K):
        cluster_indices = np.where(S.array_2d == k)
        n_k = len(cluster_indices)
        distance = overall_means.get_distance(centroids[k],0).array_2d.reshape(1,m)
        b_j += (n_k * distance)
    return b_j
    


        
def get_groups(data_matrix,K):
    n,m = data_matrix.array_2d.shape
    data_std = data_matrix.standardise() # standardising th data matrix to calculate distance
    weights = get_initial_weights(m) # m --> no. of cols 
    random_indices = np.random.choice(n, K)   # selecting random centroids from the data_matrix
    centroids = data_std.array_2d[random_indices, :]
    S = np.zeros((n,1))
    new_S = np.zeros((n,1)) # creating another S matrix to comparet the new and previous S matrix
    
    # calculating the weighted euclidean distances and updating S matrix with index of shortest distance(clustering the data by its index)
    for row_i in range(n):
            distances = data_std.get_weighted_distance(centroids,weights,row_i)
            new_S[row_i] = np.argmin(distances.array_2d)
            if np.array_equal(S, new_S):
                break
            S = new_S
            centroids = get_centroids(data_std,S,K) #updating the centroids
            a_j = get_separation_within(data_std,S,centroids,K)
            b_j = get_separation_between(data_std,S,centroids,K)
    return matrix(S)

def get_new_weights(m,centroids,old_weight_vector,S,K):
    pass

def run_test() :
    m = matrix("Data.csv") 
    for k in range(2,11):
        for i in range(20):
            S = get_groups(m, k) 
            print(str(k)+'='+str(S.get_count_frequency()))