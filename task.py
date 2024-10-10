import pandas as pd
import numpy as np


class Matrix: 
     
        def __init__(self,filename):
            
            self.array_2d = None   # class Matrix initialized with attribute array_2d
            if filename is not None:
                self.load_from_csv(filename)

        
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
        
            return D_std
        
    
    
        def get_distance(self,other_matrix,row_i):
                
                y = self.array_2d[row_i] # getting the row_i of the matrix calling this method
                
                distance_matrix = []
                
                # calculating the euclidean distance between given matrix and other matrix
                
                for x in other_matrix:
                    distance = np.sum((x - y)**2)
                    distance_matrix.append(distance)
                return np.array(distance_matrix).reshape(other_matrix.shape[0],1)
        
        
        
        def get_weighted_distance(self,other_matrix,weights,row_i):
            
            y = self.array_2d[row_i] # getting the row_i of the matrix calling this method
            
            distance_matrix = []
            
            # calculating the wieghted euclidean distance between given matrix and other matrix
            
            for x in other_matrix:
                distance = np.sum(weights * ((x - y)**2))
                distance_matrix.append(distance)
            return np.array(distance_matrix).reshape(other_matrix.shape[0],1) # reshaping it to 2d array
        
        
        
        def get_count_frequency(self):
        
            if self.array_2d.shape[1] != 1:  # checking if the matrix calling this method has only a single column
                return 0
            
            else:
                count_dictionary = {}   # initializing an empty dictionary
                one_dim_array = self.array_2d.flatten()
                
                for i in one_dim_array:
                    if i not in count_dictionary:
                        count_dictionary[i] = 1   # if the element is already not in the dict, it creates a new key:value pair 
                                                # with element and its frequency of occurence
                    else:
                        count_dictionary[i] += 1 # if element is already there in dict, it increments the count by 1
            return count_dictionary
        
# Functions

def get_initial_weights(m):
    np.random.seed(42) # setting the seed, to reproduce the same output every time
    
    random_numbers = np.random.rand(1,m)  # rand() method creates a 2D array (1 row, m columns) filled with random values between 0 & 1
    
    initial_weights = random_numbers/np.sum(random_numbers) # normalizing by didving every element in the matrix with the total sum
    
    return initial_weights
        