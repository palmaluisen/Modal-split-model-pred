#Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from itertools import cycle
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

class Logit_model_pred:
    def __init__(self, pandasResults, V, df, modos):
        self.pandasResults = pandasResults
        self.V = V
        self.df = df
        self.modos = modos
                
    def probs_pred_MNL(self):
                
        betas = self.pandasResults[["Value"]]

        # Crear listas de parámetros y regresores
        params_regres = []
        for i in self.V.keys():
            new_list = []
            for j in range(len(str(self.V[i]).split("+"))):
                a = str(self.V[i]).replace(" ","").replace("(","").replace(")","").split("+")[j].split("*")
                for k in range(len(a)):
                    new_list.append(a[k].split("init=")[0])
            params_regres.append(new_list)

        # Crear funciones de utilidad
        fun_util = []
        for i in range(len(params_regres)):
            params_list = []
            regres_list = []
            for j in range(len(params_regres[i])):
                if params_regres[i][j] in list(betas.index):
                    # Add to parameters list
                    params_list.append((betas["Value"][params_regres[i][j]]))
                    regres_list.append(1)
                
                else:
                    # Add to regressor list
                    regres_list.pop() #Pop the last 1 added to regres_list, because it means not a constant.
                    regres_list.append((self.df[params_regres[i][j]]))

            fun_util.append(list(np.dot(params_list, regres_list))) 

        # Crear matrices de utilidades 
        matrices_utilidades =[np.array(l) for l in [fun_util]]
        matrices_utilidades = matrices_utilidades[0].T

        # Cálculo de probabilidades 
        probs = np.divide(np.exp(matrices_utilidades),
                          np.sum(np.exp(matrices_utilidades), axis=1)[:, np.newaxis])
        
        probs = pd.DataFrame(probs, columns = self.modos)
        self.probs = probs

    def probs_pred_NL(self, nests):

        betas = self.pandasResults[["Value"]]

        # Crear listas de parámetros y regresores
        params_regres = []
        for i in self.V.keys():
            new_list = []
            for j in range(len(str(self.V[i]).split("+"))):
                a = str(self.V[i]).replace(" ","").replace("(","").replace(")","").split("+")[j].split("*")
                for k in range(len(a)):
                    new_list.append(a[k].split("init=")[0])
            params_regres.append(new_list)

        # Crear funciones de utilidad
        fun_util = []
        for i in range(len(params_regres)):
            params_list = []
            regres_list = []
            for j in range(len(params_regres[i])):
                if params_regres[i][j] in list(betas.index):
                    # Add to parameters list
                    params_list.append((betas["Value"][params_regres[i][j]]))
                    regres_list.append(1)
                
                else:
                    # Add to regressor list
                    regres_list.pop() #Pop the last 1 added to regres_list, because it means not a constant.
                    regres_list.append((self.df[params_regres[i][j]]))

            fun_util.append(list(np.dot(params_list, regres_list))) 

        # Crear matrices de utilidades 
        matrices_utilidades =[np.array(l) for l in [fun_util]]
        matrices_utilidades = matrices_utilidades[0].T

        # Tratamiento de Nests
        ## Lista de listas de los modos que contiene cada Nest: por ejemplo> [[1], [0, 2]]: Nest_0: modo 1, Nest_1: modos 0 y 2
        modos_nest = [list(nests[i])[1] for i in range(len(nests))]

        ## Valores de lambdas por Nests
        lambdas_pos = [int(list(nests[i])[0]) if isinstance(list(nests[i])[0], float) else self.pandasResults.loc[str(list(nests[i])[0]).split("(init")[0],"Value"]  for i in range(len(nests))]

        # Creación de diferentes listas y arrays:
        # mode_in_nest: Nest al que pertenece cada modo
        # lambdas: lambda de cada modo
        # numerator_sum: array (N, modos) con los numeradores de cada N
        # denominator_sum: array (N, modos) con los denominadores de cada N

        mode_in_nest = []
        lambdas = [] 
        numerator_sum = []
        denominator_sum =[]
        for i in self.V.keys():
            for ix,j in enumerate(modos_nest):
                if i in j:
                    mode_in_nest.append(ix)
                    lambdas.append(lambdas_pos[ix])
                    numerator_sum.append(sum([np.exp(matrices_utilidades[:,k]/lambdas_pos[ix]) for k in j])**(lambdas_pos[ix]-1))
                    denominator_sum.append(sum([np.exp(matrices_utilidades[:,k]/lambdas_pos[ix]) for k in j])**(lambdas_pos[ix]))

        numerator_sum = np.array(numerator_sum).T
        denominator_sum = np.array(denominator_sum).T

        ## Lista para decidir si sumar o no los elementos de las listas del denominador
        sum_or_not = [True if i not in mode_in_nest[:ix] else False for ix,i in enumerate(mode_in_nest)]

        # Lista de denominadores de cada N
        total_denom = []
        for i in denominator_sum:
            td = []
            for ixj, j in enumerate(i):
                if sum_or_not[ixj]==True:
                    td.append(i[ixj])
            total_denom.append(sum(td))

        # Probs base
        probs_base = np.exp(matrices_utilidades/lambdas)/(np.array(total_denom)[:, np.newaxis])

        # Probs nested logit
        probs = np.multiply(probs_base, numerator_sum)

        probs = pd.DataFrame(probs, columns = self.modos)
        self.probs = probs

    def classification_report(self):
        y_true = [i for i in self.df.CHOICE]
        #y_pred = [np.argmax([self.Proba_Car[i], self.Proba_Bus[i], self.Proba_BP[i]]) for i in range(len(self.Proba_Car))]
        y_pred = [np.argmax(self.probs.iloc[i]) for i in range(self.probs.shape[0])]
        target_names = self.modos
        print(classification_report(y_true, y_pred, target_names=target_names)) #

    def confusion_matrix_display(self):
        y_true = [i for i in self.df.CHOICE]
        y_pred = [np.argmax(self.probs.iloc[i]) for i in range(self.probs.shape[0])]
        target_names = self.modos
        cm = confusion_matrix(y_true, y_pred, labels= [0,1,2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names) 
        disp.plot(cmap=plt.cm.BuGn)
        plt.show()       

    def ROC_curve(self):
        y_score = [self.probs.iloc[i] for i in range(len(self.probs))] #np.array([[,1-i] for i in range(len(self.Proba_Car))])
        y_test=pd.get_dummies(self.df.CHOICE).values
        n_classes = len(self.modos)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        lw=2
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], [y_score[idx][i] for idx in range(len(y_score))])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        def gen_lista_colores(cantidad):
            colores = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink', 'gray', 'black']
            return (colores * (cantidad // len(colores) + 1))[:cantidad]

        colors = cycle(gen_lista_colores(n_classes))

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-class data')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()
                
    # def Precision_Recall_curve(self):
    #     from sklearn.metrics import precision_recall_curve
    #     from sklearn.metrics import PrecisionRecallDisplay
    #     y_true = [1-i for i in self.df.CHOICE]
    #     y_score = self.Proba_Car

    #     prec, recall, _ = precision_recall_curve(y_test, y_score)
    #     pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, ).plot()

    def Comparative_plot(self, based_on, umbral = 0.5):
        
        def gen_lista_marcadores(cantidad):
            marcadores = ["o","v","l","s","p","*","d","D"]
            return (marcadores * (cantidad // len(marcadores) + 1))[:cantidad]

        def gen_lista_colores(cantidad):
            colores = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink', 'gray', 'black']
            return (colores * (cantidad // len(colores) + 1))[:cantidad]

        # Lista con elementos restantes a based_on de la lista de modos 
        list_last_elements = [i for ix, i in enumerate(self.modos) if ix!=based_on]

          #df_comparativo = pd.DataFrame({"Obs": self.df.CHOICE, "Pred_CAR": self.Proba_Car, "Pred_BUS": self.Proba_Bus, "Pred_BP": self.Proba_BP})
        df_comparativo = self.probs.copy()
        df_comparativo.columns = ["Pred_" + i for i in list(df_comparativo.columns)]
        df_comparativo["Obs"] = self.df.CHOICE       

        df_comparativo = df_comparativo.iloc[df_comparativo.iloc[:, based_on].argsort()]
        scatter_x = np.array(range(len(self.df.CHOICE)))
        scatter_y = np.array(df_comparativo.iloc[:,based_on])
        group = np.array(df_comparativo.Obs)
        
        # Dictionary of colors and markers
        ## Color and marker lists
        clr = ["darkblue", "#00ff00"] + gen_lista_colores(len(self.modos))
        t = ["x", "+"] + gen_lista_marcadores(len(self.modos))
        
        cdict = {based_on:"magenta"}
        tdict = {based_on:"o"}
        for ix, i in enumerate([i for ix, i in enumerate(range(len(self.modos))) if ix!=based_on]):              
            cdict[i] = clr[ix]
            tdict[i] = t[ix]

        names = ["Prob"+self.modos[based_on]+"_Real"+i for i in self.modos]

        fig, ax = plt.subplots(figsize=(11,5))
        plt.grid()
        plt.title(f"Basadas en {self.modos[based_on]}")
        plt.ylabel("Probabilidades")
        plt.plot([0,df_comparativo.shape[0]],[umbral,umbral],color="darkred", label="Umbral")
        for g in np.unique(group):
            ix = np.where(group == g)
            ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], marker=tdict[g], label = names[g])
        ax.legend()

        for i in list_last_elements:
            plt.scatter(range(len(self.df.CHOICE)), df_comparativo["Pred_"+i], label="Prob_"+i, marker=".", alpha=0.5)
                
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()