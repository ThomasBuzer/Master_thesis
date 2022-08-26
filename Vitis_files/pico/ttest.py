import sys
import numpy as np
import trsfile
import matplotlib.pyplot as plt
from tqdm import tqdm

class ttest(object):
    def __init__(self, Ns):
        self.Ns = Ns
        self.Nt = np.zeros(2, dtype=np.int)
        self.Nt_total = 0
        self.t_sum = np.zeros((2, Ns), dtype=np.float32)
        self.t_sum2 = np.zeros((2, Ns), dtype=np.float32)
        self.t_var = np.empty((2,Ns), dtype=np.float32)
        self.tstat = np.zeros((Ns), dtype=np.float32)

    def ttest_trsfile(self, traceset, disable_tqdm=False):
        #Var[X] = E[X**2] - E[X]**2
        for trace in tqdm(traceset[self.Nt_total:], disable=disable_tqdm):
            t = trace.samples
            self.t_sum[int(trace.parameters.serialize()[0])] += t.astype(np.float32)
            self.t_sum2[int(trace.parameters.serialize()[0])] += t*t.astype(np.float32)
            self.Nt[int(trace.parameters.serialize()[0])] += 1
        self.Nt_total = self.Nt[0] + self.Nt[1]

        for i in range(2):
            self.t_var[i] = self.t_sum2[i]/self.Nt[i] - (self.t_sum[i]/self.Nt[i])**2

        # Tstat = (E[X] - E[Y])/(sqtr(Var[X]/Nx + Var[Y]/Ny))
        self.tstat = (self.t_sum[0]/self.Nt[0] - self.t_sum[1]/self.Nt[1])/np.sqrt((self.t_var[0]/self.Nt[0]) + (self.t_var[1]/self.Nt[1]))
        print("t-value MAX : ", max(max(self.tstat), -min(self.tstat)))
        return self.tstat


if __name__=="__main__":
    traceFileName = sys.argv[1]
    traceset = trsfile.open(traceFileName, 'r')

    # tstat = ttest_trsfile(traceset)
    ttest_object = ttest(traceset.get_header(trsfile.Header.NUMBER_SAMPLES))
    tstat = ttest_object.ttest_trsfile(traceset)

    fig, ax1 = plt.subplots()
    ax1.set_ylabel("t-values")
    plot_1 = ax1.plot(tstat, label='t-stat')
    ax1.plot(4.5*np.ones(len(tstat)), color='red')
    ax1.plot(-4.5*np.ones(len(tstat)), color='red')
    ax2 = ax1.twinx()
    plot_2 = ax2.plot(traceset.get_header(trsfile.Header.SCALE_Y)*traceset[0].samples, color='gray', alpha=0.5, label='power trace')
    ax2.set_ylabel("V")
    #add legend
    lns = plot_1 + plot_2
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc=0)
    plt.show()

    traceset.close()
