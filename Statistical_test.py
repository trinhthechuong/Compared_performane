import scikit_posthocs as sp
from typing import Union, List, Tuple
import numpy as np
from matplotlib import colors
from matplotlib.axes import SubplotBase
from matplotlib.colorbar import ColorbarBase, Colorbar
from matplotlib.colors import ListedColormap
from pandas import DataFrame
from seaborn import heatmap
from sklearn.dummy import DummyClassifier, DummyRegressor
from itertools import combinations
from math import factorial
import numpy as np
from scipy.stats import t
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

class statistical_test:

    def __init__(self, results, model,Data_train = None , X_train = None ,y_train = None):
        if  type(Data_train) == DataFrame:
            self.X_train = Data_train.drop([y_name], axis = 1)
            self.y_train = Data_train[y_name]
        else:
            self.X_train = X_train
            self.y_train = y_train
        self.results = results
        self.names = model
        self.cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        #self.scoring = input("please input the metric that you want to compare:")
        self.make_dataframe()
        #self.posthoc()
        self.n_train = len(list(self.cv.split(self.X_train, self.y_train))[0][0])
        self.n_test = len(list(self.cv.split(self.X_train, self.y_train))[0][1])




    def make_dataframe(self):
      #Create pandas Data Frame for statical test
        base_estimator = DummyRegressor()
        scores_0 = cross_val_score(base_estimator,self.X_train, self.y_train,cv=self.cv)
        df_score = pd.DataFrame(scores_0).transpose()
        df_score.insert(0, "Name", ["Dummy"], True)
        for i in range(len(self.results)):
          scores = self.results[i]
          demo = pd.DataFrame(scores).transpose()
          demo.insert(0, "Name", [self.names[i]], True)
          df_score = pd.concat([df_score,demo],ignore_index=True)
        df_score = df_score.drop(0,axis=0)
        df_score = df_score.reset_index(drop = True)
        self.model_scores = df_score.set_index("Name")
        return self.model_scores


    def statiscal_dataframe(self):
        n_comparisons = factorial(len(self.model_scores)) / (
    factorial(2) * factorial(len(self.model_scores) - 2))
        pairwise_t_test = []

        for model_i, model_k in combinations(range(len(self.model_scores)), 2):
          model_i_scores = self.model_scores.iloc[model_i].values
          model_k_scores = self.model_scores.iloc[model_k].values
          differences = model_i_scores - model_k_scores
          n = differences.shape[0]
          self.df = n-1

          t_stat, p_val = self.compute_corrected_ttest(differences, self.df, self.n_train, self.n_test)
          p_val *= n_comparisons  # implement Bonferroni correction
    # Bonferroni can output p-values higher than 1
          p_val = 1 if p_val > 1 else p_val
          pairwise_t_test.append([self.model_scores.index[model_i], self.model_scores.index[model_k], t_stat, p_val])

        self.pairwise_comp_df = pd.DataFrame(
    pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]).round(3)
        return self.pairwise_comp_df


    def bayesian_df(self):
        self.pairwise_bayesian = []
        rope_interval = [-0.01, 0.01]

        for model_i, model_k in combinations(range(len(self.model_scores)), 2):
          model_i_scores = self.model_scores.iloc[model_i].values
          model_k_scores = self.model_scores.iloc[model_k].values
          differences = model_i_scores - model_k_scores
          t_post = t(self.df, loc=np.mean(differences), scale=self.corrected_std(differences, self.n_train, self.n_test))
          self.worse_prob = t_post.cdf(rope_interval[0])
          self.better_prob = 1 - t_post.cdf(rope_interval[1])
          self.rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])

          self.pairwise_bayesian.append([self.worse_prob, self.better_prob, self.rope_prob])

        self.pairwise_bayesian_df = pd.DataFrame(self.pairwise_bayesian, columns=["worse_prob", "better_prob", "rope_prob"]).round(3)
        self.better_prob = self.pairwise_bayesian_df["better_prob"]
        self.worse_prob  = self.pairwise_bayesian_df["worse_prob"]
        self.rope_prob  = self.pairwise_bayesian_df["rope_prob"]
        return self.pairwise_bayesian_df

    def bayesian_comparison(self):
        self.statiscal_dataframe()
        self.bayesian_df()
        self.pairwise_comp_df = self.pairwise_comp_df.join(self.pairwise_bayesian_df)
        display(self.pairwise_comp_df)
        return self.pairwise_comp_df

    def wilcoxon_comparison(self):
        a = np.stack(self.results)
        df_metrics = pd.DataFrame(a.T, columns = self.names)
        df_melt = pd.melt(df_metrics.reset_index(), id_vars=['index'], value_vars=df_metrics.columns)
        df_melt.columns = ['index', 'Method', 'Scores']
        pvalue_df = sp.posthoc_wilcoxon(df_melt, val_col='Scores', group_col='Method', p_adjust='holm')
        return pvalue_df
        
        
    def posthoc_comparison(self, title = "Comparison heatmap", save = False):
   #Create dataframe for Posthocs
        self.posthoc_title = title
        a = np.stack(self.results)
        self.df_metrics = pd.DataFrame(a.T, columns = self.names)
        self.df_melt = pd.melt(self.df_metrics.reset_index(), id_vars=['index'], value_vars=self.df_metrics.columns)
        self.df_melt.columns = ['index', 'Method', 'Scores']
        self.pvalue_df = sp.posthoc_wilcoxon(self.df_melt, val_col='Scores', group_col='Method', p_adjust='holm')
        plt.figure(figsize = (16,8))
        ax = plt.axes()
        heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        self.sign_plot(self.pvalue_df, **heatmap_args)
        ax.set_title(self.posthoc_title,fontsize = 20, weight ='semibold')
        if save == "True":
            plt.show()
            title_fig = self.posthoc_title+".png"
            plt.savefig(title_fig, dpi = 600)
        else:
            plt.show()



    def sign_plot(
        self,
        x: Union[List, np.ndarray, DataFrame],
        g: Union[List, np.ndarray] = None,
        flat: bool = False,
        labels: bool = True,
        cmap: List = None,
        cbar_ax_bbox: List = None,
        ax: SubplotBase = None,
        **kwargs) -> Union[SubplotBase, Tuple[SubplotBase, Colorbar]]:

        for key in ['cbar','vmin','vmax','center']:
          if key in kwargs:
            del kwargs

        if isinstance(x, DataFrame):
          df = x.copy()
        else:
          x = np.array(x)
          g = g or np.arange(x.shape[0])
          df = DataFrame(np.copy(x), index=g, columns=g)

        dtype = df.values.dtype

        if not np.issubdtype(dtype, np.integer) and flat:
          raise ValueError("X should be a sign_array or DataFrame of integers")
        elif not np.issubdtype(dtype, np.floating) and not flat:
          raise ValueError("X should be an array or DataFrame of float p values")

        if not cmap and flat:
          cmap = ['1', '#fbd7d4', '#1a9641']

        elif not cmap and not flat:
          cmap = ['1', '#fbd7d4', '#005a32', '#238b45', '#a1d99b']

        if flat:
          np.fill_diagonal(df.values, -1)
          hax = heatmap(df, vmin=-1, vmax=1, cmap=ListedColormap(cmap),
                      cbar=False, ax=ax, **kwargs)
          if not labels:
            hax.set_xlabel('')
            hax.set_ylabel('')
            hax.set_title(self.posthoc_title,fontsize = 50, weight ='semibold')
          return hax

        else:
          df[(x < 0.001) & (x >= 0)] = 1
          df[(x < 0.01) & (x >= 0.001)] = 2
          df[(x < 0.05) & (x >= 0.01)] = 3
          df[(x >= 0.05)] = 0

          np.fill_diagonal(df.values, -1)

          if len(cmap) != 5:
            raise ValueError("Cmap list must contain 5 items")

          hax = heatmap(
              df, vmin=-1, vmax=3, cmap=ListedColormap(cmap), center=1,
            cbar=False, ax=ax, **kwargs)
          if not labels:
            hax.sex_xlabel("")
            hax.sex_ylabel("")
            hax.set_title(self.posthoc_title,fontsize = 50, weight ='semibold')
          cbar_ax = hax.figure.add_axes(cbar_ax_bbox or [0.95, 0.35, 0.04, 0.3])
          cbar = ColorbarBase(cbar_ax, cmap=(ListedColormap(cmap[2:] + [cmap[1]])), norm=colors.NoNorm(),
                            boundaries=[0, 1, 2, 3, 4])
          cbar.set_ticks(np.linspace(0.3, 3, 4))
          cbar.set_ticklabels(['p < 0.001', 'p < 0.01', 'p < 0.05', 'NS'])
          cbar.outline.set_linewidth(1)
          cbar.outline.set_edgecolor('0.5')
          cbar.ax.tick_params(size=0)
        
          return hax, cbar


    def corrected_std(self,differences, n_train, n_test):
      kr = len(differences)
      corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
      corrected_std = np.sqrt(corrected_var)
      return corrected_std


    def compute_corrected_ttest(self,differences, df, n_train, n_test):
      mean = np.mean(differences)
      std = self.corrected_std(differences, n_train, n_test)
      t_stat = mean / std
      p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
      return t_stat, p_val

    def boxplot_comparision(self, show_pvalue = False, score = "Score",title = "Comparison bopxlot",save=False):
        AUC_internal = pd.DataFrame(self.results).T


        AUC_internal.columns = self.names
        df_melt = pd.melt(AUC_internal.reset_index(), id_vars=['index'], value_vars=AUC_internal.columns)
        df_melt.columns = ['index', 'Model', score]
        #sns.set_style("whitegrid")
        subcat_palette = sns.dark_palette("#8BF", reverse=True, n_colors=1)
        stat_results = []
        for model_i, model_k in combinations(self.names,2):
            stat_results.append(wilcoxon(AUC_internal[model_i].astype('float'), AUC_internal[model_k].astype('float'), alternative="two-sided"))


        pvalues = [result.pvalue for result in stat_results]
        #print ("p value:",pvalues)
        pairs = list(combinations(self.names, 2))
        plotting_parameters = {
            'data':    df_melt,
            'x':       'Model',
            'y':       score,
            'palette': subcat_palette[1:]
        }



        sns.set_style("whitegrid")
        plt.figure(figsize = (30,15))




        ax = sns.boxplot(x='Model', y=score, data=df_melt, showmeans=True ,meanprops={"marker":"d",
                                   "markerfacecolor":"white", 
                                   "markeredgecolor":"black",
                                  "markersize":"10"})

        mean = round(AUC_internal.mean(),3)
        data = np.array(mean)   
        ser = pd.Series(data, index =AUC_internal.columns)

        dict_columns = {'Mean':mean,}
        df = pd.DataFrame(dict_columns)

        vertical_offset = df["Mean"].median()*0.008

        for xtick in ax.get_xticks():
            ax.text(xtick,ser[xtick]+ vertical_offset,ser[xtick], 
            horizontalalignment='center',fontsize = 30,color='w',weight='semibold')

        if show_pvalue == True:
            annotator = Annotator(ax, pairs, **plotting_parameters)
            annotator.configure(text_format="simple", verbose = 0,fontsize = 30)
            annotator.set_pvalues_and_annotate(pvalues)
        else:
            pass


        ax.set_ylabel(score, fontsize = 30, weight = "semibold")
        ax.set_xlabel(None)
        ax.set_xticklabels(labels = AUC_internal.columns, rotation = "horizontal" ,fontsize = 15)
        ax.set_title(title,fontsize = 50, weight ='semibold')
        if save == "True":
            plt.show()
            title_fig = title+".png"
            plt.savefig(title_fig, dpi = 600)
        else:
            plt.show()

        #self.bayesian_comparision()
        #self.posthoc()
        
import numpy as np
from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu, normaltest, wilcoxon
from statannot import add_stat_annotation
import seaborn as sns
import matplotlib.pyplot as plt

