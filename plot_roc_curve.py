import pylab as pl
import numpy as np

pl.clf()

precision_rf = np.array([0.71039924, 0.91129032, 1])
recall_rf = np.array([1, 0.94676521, 0])
auc_rf = 0.860343339644
pl.plot(recall_rf, precision_rf, label='RF, AUC: 0.860343339644, Accuracy: 0.89670952889')

precision_svm = np.array([0.71039924, 0.87210411, 1])
recall_svm = np.array([1, 0.94024424, 0])
auc_svm = 0.80099948297
pl.plot(recall_svm, precision_svm, label='SVM, AUC: 0.80099948297, Accuracy: 0.859593463979')

precision_logr = np.array([0.71039924, 0.86374399, 1])
recall_logr = np.array([1, 0.92269691, 0])
auc_logr = 0.782822038867
pl.plot(recall_logr, precision_logr, label='Logistic Regression, AUC: 0.782822038867, Accuracy: 0.841681172441')

precision_nb = np.array([0.71039924, 0.93472543, 1])
recall_nb = np.array([1, 0.46350235, 0])
auc_nb = 0.692051708954
pl.plot(recall_nb, precision_nb, label='NB, AUC: 0.692051708954, Accuracy: 0.595878488405')


pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])

pl.title('Precision-Recall curve')
pl.legend(loc="lower left")
pl.show()