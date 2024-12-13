import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

if __name__ == "__main__":

    figsize = (8, 4)
    figure, ax = plt.subplots(figsize=figsize)

    file1 = open("./ProbLeft-Q", "rb")
    file2 = open("./ProbLeft-D-Q-average", "rb")
    file3 = open("./ProbLeft-MMQL", "rb")     
    file4 = open("./ProbLeft-WSQL", "rb")   
    file5 = open("./ProbLeft-LSEQL", "rb")    
    file6 = open("./ProbLeft-BZQL", "rb")
    num_iter = 1000
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    arr3 = np.load(file3)  
    arr4 = np.load(file4)  
    arr5 = np.load(file5)  
    arr6 = np.load(file6)
    mean1 = np.mean(arr1, axis=1)
    mean2 = np.mean(arr2, axis=1)
    mean3 = np.mean(arr3, axis=1)  
    mean4 = np.mean(arr4, axis=1)  
    mean5 = np.mean(arr5, axis=1)  
    mean6 = np.mean(arr6, axis=1)
    x = range(100)

    plt.plot(x, mean3, 'g-x', markevery=10, label='MMQL ($ w = 0$)')  
    plt.plot(x, mean4, 'k-*', markevery=10, label='WSQL ($ w = 0.1$)')  
    plt.plot(x, mean5, 'm-s', markevery=10, label='LSEQL ($ w = 1$)') 
    plt.plot(x, mean6, 'y--', markevery=10, label='BZQL')
    plt.plot(x, mean1, 'r--', markevery=10, label='QL')
    plt.plot(x, mean2, 'b-o', markevery=10, label='DQL')

    # Customize legend with title
    legend = plt.legend(loc='best', fancybox=True, shadow=True, prop={'size': 10},)
    legend.get_title().set_fontname('Times New Roman')

    # Customize ticks
    plt.tick_params(labelsize=12)  # Reduce label size
    

    # Set font properties for x and y labels
    font = {'family': 'Times New Roman', 'size': 12}  # Non-bold font
    plt.xlabel('Number of Episodes', fontdict=font)
    plt.ylabel('Probability of a left action', fontdict=font)
    plt.tight_layout()
    plt.savefig('BIAS_PLOT.pdf', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
