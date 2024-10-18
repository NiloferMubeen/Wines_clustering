#IMPORTING ALL THE REQUIRED LIBRARIES
import pandas as pd
import numpy as np

# Creating class matrix and its methods
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
                        df = pd.read_csv(filename,header =None)  #using pandas to read the data 
                        self.array_2d = df.to_numpy()   # converting the data into array_2d matrix
                        
                    except Exception as e:
                        print(e)


            def standardise(self):
                        
                    # Initializing an empty matrix to hold the standardized values of the original matrix
                    D_std = np.zeros(self.array_2d.shape)
                    
                    # Iterate over each column of original matrix and calculate the column avg, max and min values
                    
                    for j in range(self.array_2d.shape[1]): # looping through all the columns
                        
                        
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
                    # calculating the euclidean distance between data matrix and every centroids (other matrix)
                    for x in other_matrix:
                            distance = np.sum((x - y)**2)
                            distance_matrix.append(distance)
                    dist =  np.array(distance_matrix)
                    return dist 
            
            
            def get_weighted_distance(self,other_matrix,weights,row_i):
                y = self.array_2d[row_i] # getting the row_i of the matrix calling this method
                distance_matrix = []
                # calculating the weighted euclidean distance between data matrix and other matrix
                for x in other_matrix:
                        distance = np.sum(weights * ((x - y)**2)) 
                        distance_matrix.append(distance)
                dist =  np.array(distance_matrix)
                return dist
            
            
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

        
def get_initial_weights(m): # m -> number of columns in data matrix
    random_numbers = np.random.rand(1,m)  # rand() method creates a 2D array (1 row, m columns) filled with random values between 0 & 1
    initial_weights = random_numbers/np.sum(random_numbers) # normalizing by dividing every element in the matrix with the total sum
    return initial_weights   


def get_centroids(data_matrix,S,K):
    n, m = data_matrix.array_2d.shape  
    # create an empty matrix of dimension(K,m)
    centroids = np.empty((K, m)) 
    for k in range(K):
        indices = np.where(S == k)[0]# Get all row indices from S matrix where the corresponding row in S equals k 
        
        # get the mean of corresponding rows(according to row indices) from data matrix and update the centroids matrix
        centroids[k] = np.mean(data_matrix.array_2d[indices],axis=0) #updating the centroids
    return centroids


    
def get_separation_within(data_matrix,S,centroids,K):
    n,m = data_matrix.array_2d.shape
    a_j = np.zeros((1, m))  #initializing an empty matrix of shape(1,m) 
    for k in range(K):
        cluster_indices = np.where(S.array_2d == k)[0] # getting the indices where the elements of S matrix is equal to k
        #calculating the distance between every data point and their corresponding clusters
        for i in cluster_indices:
                    distance = data_matrix.get_distance(centroids[k],i) 
                    
                    a_j += distance # sum of all the cluster distances
    return a_j
 

def get_separation_between(data_matrix,S,centroids,K):
    
    m = data_matrix.array_2d.shape[1]
    b_j = np.zeros((1, m)) 
    
    # calculates the mean of of all the columns
    overall_means = matrix(np.mean(data_matrix.array_2d,axis=0).reshape(1,m)) 
    for k in range(K):
        cluster_indices = np.where(S.array_2d == k)
        n_k = len(cluster_indices) # number of data points assigned to a cluster
        distance = overall_means.get_distance(centroids[k],0) # here row is 0 as overall means has only one row
        b_j += (n_k * distance)
    return b_j
    

        
def get_groups(data_matrix,K):
    n,m = data_matrix.array_2d.shape
    
    # standardising th data matrix to calculate distance
    data_std = data_matrix.standardise() 
    
    # getting the initialweights for 'm'columns
    weights = get_initial_weights(m)  
    
    # initiallyselecting random centroids from the data_matrix
    random_indices = np.random.choice(n, K)   
    centroids = data_std.array_2d[random_indices, :]
    
    # creating S (cluster assignment matrix) and new_S to compare the new and previous S matrix
    S = matrix(np.zeros((n,1)))
    new_S = matrix(np.zeros((n,1))) 

    # calculating the weighted euclidean distances and updating S matrix with index of shortest distance(clustering the data by its index)
    for row_i in range(n):
                    distances = data_std.get_weighted_distance(centroids,weights,row_i)
                    new_S.array_2d[row_i] = np.argmin(distances)
                    if np.array_equal(S.array_2d, new_S.array_2d):  #checks if the new and the old S matrix are the same after the updation
                        break
                    
                    # if not equal, new_s is copied to the old s matrix with updated centroids and weights
                    S.array_2d = new_S.array_2d.copy() 
                    
                    #updating the centroids
                    centroids = get_centroids(data_std,S,K) 
                    
                     # updating weights
                    weights = get_new_weights(data_std,centroids,weights,S,K)
                    
    return S 

def get_new_weights(data_matrix,centroids,old_weight_vector,S,K):
    
    m = old_weight_vector.shape[1]
    
    # calculating separation within clusters 
    a_j = get_separation_within(data_matrix,S,centroids,K)
    
    # calculating separation between clusters
    b_j = get_separation_between(data_matrix,S,centroids,K)
    
    new_weights = np.zeros((1, m))
    
    b_over_a = np.sum(b_j/a_j)
    # Loop over each feature j to calculate the new weight
    for j in range(m):
        bj_over_aj = b_j[0, j] / a_j[0, j] 
        new_weights[0, j] += 0.5 * (old_weight_vector[0, j] + (bj_over_aj / b_over_a))
    return new_weights


def run_test() :
    m = matrix("Data.csv") 
    for k in range(2,11):
        for i in range(20):
            S = get_groups(m, k) 
            print(str(k)+'='+str(S.get_count_frequency()))
            
print(run_test())