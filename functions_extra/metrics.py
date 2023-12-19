import pandas as pd
import numpy as np 
from scipy.stats import chi2_contingency
from itertools import product


class Relations:
    def __init__(self, dfs: pd.DataFrame, resp: str):
        self.dfs = dfs.fillna("Nulo").copy()
        self.resp = resp

    def get_woe(self, vars: list, target: str):   
        resp = target 
        out_dict = {}
        out_dict_iv = {}
        out_dict_count = {}
        out_dict_fraud = {}
        l_bom = len(self.dfs[self.dfs[resp]==0])+.5
        l_mau = len(self.dfs[self.dfs[resp]==1])+.5
        for v in vars:
            dict_woe = {}
            dict_woe_count = {}
            dict_woe_fraud = {}
            list_iv = []
            for y in self.dfs[v].unique():
                bons = (len(self.dfs.loc[
                    ((self.dfs[v]==y)&(self.dfs[resp]==0)), 
                    resp])+.5)
                maus = (len(self.dfs.loc[
                    ((self.dfs[v]==y)&(self.dfs[resp]==1)), 
                    resp])+.5)
                
                dict_woe[y] = np.log((maus/l_mau)/(bons/l_bom))
                dict_woe_count[str(y)] = [bons-.5 + maus-.5]
                dict_woe_fraud[str(y)] = [maus-.5]

                list_iv.append(dict_woe[y]*((maus/l_mau)-(bons/l_bom)))
            out_dict[str(v)] = dict_woe
            out_dict_iv[v] = np.sum(list_iv)
            out_dict_count[str(v)] = dict_woe_count
            out_dict_fraud[str(v)] = dict_woe_fraud
        return out_dict, out_dict_iv, out_dict_count, out_dict_fraud

    def get_cramers_V(self, var1: list, var2: list) :
        prod = product(var1, var2)

        out = pd.DataFrame()
        for a,b in prod:
            crosstab =np.array(pd.crosstab(self.dfs[a], self.dfs[b], rownames=None, colnames=None)) # Cross table building
            stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
            obs = np.sum(crosstab) # Number of observations
            mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
            result = np.sqrt(stat/(obs*mini))
            out.loc[a,b] = result

        return out