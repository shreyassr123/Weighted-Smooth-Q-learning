import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

if __name__ == "__main__":

    figsize = (8, 4)
    figure, ax = plt.subplots(figsize=figsize)

    # Open the files
    file1 = open("./ProbLeft-Q", "rb")
    file2 = open("./ProbLeft-D-Q-average", "rb")
    file3 = open("./ProbLeft-WSQL", "rb")
    file4 = open("./ProbLeft-MMQL", "rb")
    file5 = open("./ProbLeft-LSEQL", "rb")
    file6 = open("./ProbLeft-BZQL", "rb")
    
    # Load the arrays
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    arr3 = np.load(file3)
    arr4 = np.load(file4)
    arr5 = np.load(file5)
    arr6 = np.load(file6)
    
    # Compute means
    mean1 = np.mean(arr1, axis=1)
    mean2 = np.mean(arr2, axis=1)
    mean3 = np.mean(arr3, axis=1)
    mean4 = np.mean(arr4, axis=1)
    mean5 = np.mean(arr5, axis=1)
    mean6 = np.mean(arr6, axis=1)

    x = range(len(mean1))  # Ensure x matches the length of means

    # Add horizontal line at y=0
    plt.axhline(y = 0, color='grey', linestyle='dotted')

    # Plot the data
    plt.plot(x, mean1, 'r-', label='QL', linewidth=1.5)
    plt.plot(x, mean2, 'b--', label='DQL', linewidth=1.5)
    plt.plot(x, mean3, 'k-.', label='WSQL ($w = 0.5$)', linewidth=1.5)
    plt.plot(x, mean4, 'r:', label='MMQL ($w = 0$)', linewidth=1.5)  # Add check2
    plt.plot(x, mean5, 'm-', label='LSEQL ($w = 1$)', linewidth=1.5)  # Add check3
    plt.plot(x, mean6, 'y-', label='BZQL ', linewidth=1.5) 
    # Customize legend
    legend = plt.legend(loc='best', fancybox=True, shadow=True, prop={'size': 10})
    legend.get_title().set_fontname('Times New Roman')

    # Customize ticks
    plt.tick_params(labelsize=10)

    # Set font properties for x and y labels
    font = {'family': 'Times New Roman', 'size': 12}  # Non-bold font
    plt.xlabel('Number of Episodes', fontdict=font)
    plt.ylabel('$\\max_a Q(a)$', fontdict=font)
    
    plt.tight_layout()
    plt.savefig('Roulette_plot.pdf', dpi=600, bbox_inches='tight')
    plt.show()
