import pickle
import active
import sys

def main(directory, y_len = 2500, x_len = 100000):
    #directory = 'exp_proton-beam_unf/'
    y_len = int(y_len)
    x_len = int(x_len)

    axis = [0, x_len, 0, y_len]
    
    f = open(directory + 'jc.pkl')
    cde = pickle.load(f)
    f.close()
    active.plot_fill(cde, 'US-Crowd', cl = 'black', ls= '-', marker = 'x')


    f = open(directory + 'je.pkl')
    cde = pickle.load(f)
    f.close()
    active.plot_fill(cde, 'US-Expert', cl = 'red', ls = '-', marker = '^')


    f = open(directory + 'cde.pkl')
    cde = pickle.load(f)
    f.close()
    active.plot_fill(cde, 'US-Crowd+Expert', cl = 'blue', ls = '-', marker = 'o')



    f = open(directory + 'dec5_ip.pkl')
    cde = pickle.load(f)
    f.close()
    print axis
    active.plot_fill(cde, 'Decision Theory', cl = 'green', ls = '-', marker = 's', axis = axis)

    #f = open(directory + 'dec5_ip.pkl')
    #cde = pickle.load(f)
    #f.close()
    #active.plot_fill(cde, 'Decision Theory(5) + IP', cl = 'green', ls = '-')



    #f = open('dec5_ip.pkl')
    #dec = pickle.load(f)
    #f.close()
    #active.plot_fill(dec, 'Dec Theory(5) + Interpoplate', cl = 'green', ls = ':')



    active.plt.legend(loc='upper right')
    active.plt.xlabel('Label Cost')
    active.plt.ylabel('True Loss')

    active.plt.savefig(directory + 'fig.png')

    #active.plt.annotate('us-crowd', xy=(2000, 750),  xycoords='data',
    #                xytext=(0, -50), textcoords='offset points',
    #                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color = 'gray'), color = 'black',
    #                )
                    
                    
    #active.plt.annotate('us-expert', xy=(5000, 2200),  xycoords='data',
    #                xytext=(0, 50), textcoords='offset points',
    #                arrowprops=dict(arrowstyle="->", color = 'gray'), color = 'red',
    #                )
                    
    #active.plt.annotate('us-crowd+expert', xy=(10000, 1100),  xycoords='data',
    #                xytext=(0, 130), textcoords='offset points', color = 'blue',
    #                arrowprops=dict(arrowstyle="->", color = 'gray')
    #                )
                    
    #active.plt.annotate('decision theory', xy=(20000, 350),  xycoords='data',
    #                xytext=(0, -30), textcoords='offset points', color = 'green',
    #                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color = 'gray')
    #                )
    
    
if __name__ == "__main__":
   main(*sys.argv[1:])
