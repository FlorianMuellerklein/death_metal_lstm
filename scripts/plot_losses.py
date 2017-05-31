import matplotlib.pyplot as plt
import seaborn as sns
import pickle

LSTM = pickle.load(open('../plots/lstm_1024.pkl', 'rb'))
miLSTM = pickle.load(open('../plots/milstm_1024.pkl', 'rb'))
ln_miLSTM = pickle.load(open('../plots/ln_milstm_1024.pkl', 'rb'))


plt.figure()
plt.plot(LSTM, label='LSTM')
plt.plot(miLSTM, label='miLSTM')
#plt.plot(ln_miLSTM, label='ln_miLSTM')
plt.legend(loc=1, prop={'size':20})
plt.ylim([0.75, 2.5])
plt.xlabel('Weight Update x100', size=20)
plt.ylabel('Cross Entropy Loss', size=20)
plt.title('miLSTM vs LSTM \n Character Level Sequence Modeling', size=30)
plt.show()