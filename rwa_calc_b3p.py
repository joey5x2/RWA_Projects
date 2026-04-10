# %%
import os
import pandas as pd
import numpy as np

is_number = lambda x: isinstance(x, (int, float, complex, np.number)) and not pd.isna(x)

class CalcEAD():

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.df.columns = df.columns.str.strip()
        self.df.replace(["N/A", "NA", "-", "—", "", " "], np.nan, inplace=True)

        numeric_cols = ['Principal', 'Market Value', 'Haircut','Scen A MPOR', 'Scen B MPOR','SA EAD']
        for col in numeric_cols:
            self.df[col] = (
                        self.df[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)      
                        .str.replace("$", "", regex=False)      
                        .str.replace("–", "-", regex=False)
                        .str.strip()                            
                    )
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)

        text_cols = [
            'Buy or Sell Indicator', 'LRM Flag', 'Dsft Base Conc',
            'Haircut Eligible Status', 'Stale prc flg 2days',
            'Illiquid Flag', 'Netting Set ID', 'Security ID',
            'Exposure Currency', 'Collateral Currency', 'Agr Settlement Ccy code'
        ]
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()

        illiquid_sets = (self.df[self.df["Illiquid Flag"] == "Y"]["Netting Set ID"].unique())
        self.df["Has Illiquid In Set"] = self.df["Netting Set ID"].isin(illiquid_sets)
        self.df['WFB Flag'] = 'Y'

        self.df['Netting Set ID'] = self.df['Netting Set ID'].where(
            self.df['WFB Flag'] == 'Y',
            self.df['Source Txns ID']
        )

        self.df = pd.concat([self.df,self.df.apply(self.calc_txn_components,axis=1)],axis=1)
        self.df_sec_addon = self._compute_sec_addon()
        self.df_fx_addon = self._compute_fx_addon()
        self.df_rwa_summary = self._compute_rwa_summary()



    def calc_txn_components(self,row):
        
        if row['Sft Transaction Type'] == 'MARGIN_LOAN':
            scen_b_mpor = 10
            scen_a_mpor = (20 if (row['Large Netting Set Flag'] == 'Y' or row['Has Illiquid In Set']) else 10) * (2 if row['Margin Dispute Flag']=='Y' else 1)
        elif row['Sft Transaction Type'] == 'SFT':
            scen_b_mpor = 5
            scen_a_mpor = (20 if (row['Large Netting Set Flag'] == 'Y' or row['Has Illiquid In Set']) else 5) * (2 if row['Margin Dispute Flag']=='Y' else 1)

        if row['Buy or Sell Indicator'] == 'B':
            trade_exposure = row['Principal']
            trade_collateral = row['Market Value']
        elif row['Buy or Sell Indicator'] == 'S':
            trade_exposure = row['Market Value']
            trade_collateral = row['Principal']
        else:
            trade_exposure = 'DQ'
            trade_collateral = 'DQ'

        if row['Buy or Sell Indicator'] == 'S':
            eligible_collateral  = trade_collateral
        elif row['LRM Flag']=='N' or row['Dsft Base Conc']=='NETTING-INELIGIBLE' or row['Haircut Eligible Status']=='Ineligible' or row['Stale prc flg 2days']=='Y':
            eligible_collateral = 0
        else:
            eligible_collateral = trade_collateral

        if row['Buy or Sell Indicator'] == 'B' and row['Illiquid Flag'] == 'Y':
            collateral_scen_b = 0
        else: 
            collateral_scen_b = eligible_collateral

        if row['Buy or Sell Indicator'] == 'S':
            sec_addon_exposure = row['Market Value']
            sec_addon_coll_scen_a = 0
            sec_addon_coll_scen_b = 0
        else:
            sec_addon_exposure = 0
            sec_addon_coll_scen_a = eligible_collateral
            sec_addon_coll_scen_b = collateral_scen_b

        net_exposure_scen_a_inelligible = (sec_addon_exposure)*row['Haircut']
        net_exposure_scen_b_inelligible = (sec_addon_exposure)*row['Haircut']

        net_exposure_scen_a_elligible = (sec_addon_exposure - sec_addon_coll_scen_a)*row['Haircut']*np.sqrt(scen_a_mpor/10)
        net_exposure_scen_b_elligible = (sec_addon_exposure - sec_addon_coll_scen_b)*row['Haircut']*np.sqrt(scen_b_mpor/10)


        gross_exposure_scen_a_inelligible = (sec_addon_exposure )*np.abs(row['Haircut'])
        gross_exposure_scen_b_inelligible = (sec_addon_exposure )*np.abs(row['Haircut'])

        gross_exposure_scen_a_elligible = (sec_addon_exposure + sec_addon_coll_scen_a)*row['Haircut']*np.sqrt(scen_a_mpor/10)
        gross_exposure_scen_b_elligible = (sec_addon_exposure + sec_addon_coll_scen_b)*row['Haircut']*np.sqrt(scen_b_mpor/10)

        return pd.Series({
            'Scen A MPOR - Recalc': scen_a_mpor,
            'Scen B MPOR - Recalc': scen_b_mpor,
            'Trade Level Exposure': trade_exposure,
            'Trade Level Collateral': trade_collateral,
            'Trade Level Collateral - Scen A': eligible_collateral,
            'Trade Level Collateral - Scen B': collateral_scen_b,
            'Net Exp - Scen A Inelligible': net_exposure_scen_a_inelligible, 
            'Net Exp - Scen B Inelligible': net_exposure_scen_b_inelligible,
            'Net Exp - Scen A Elligible': net_exposure_scen_a_elligible,
            'Net Exp - Scen B Elligible': net_exposure_scen_b_elligible,
            'Gross Exp - Scen A Inelligible': gross_exposure_scen_a_inelligible, 
            'Gross Exp - Scen B Inelligible': gross_exposure_scen_b_inelligible,
            'Gross Exp - Scen A Elligible': gross_exposure_scen_a_elligible,
            'Gross Exp - Scen B Elligible': gross_exposure_scen_b_elligible
        })

    def _compute_sec_addon(self):
        df = self.df
        df_eff_haircut = df[['Netting Set ID','Trade Level Exposure',	'Trade Level Collateral']].pivot_table(
            index=['Netting Set ID'],
            values=['Trade Level Exposure',	'Trade Level Collateral'],
            aggfunc='sum'
        ).reset_index()
        df_eff_haircut['Effective Haircut'] = (df_eff_haircut['Trade Level Collateral'] - df_eff_haircut['Trade Level Exposure']) / df_eff_haircut['Trade Level Exposure']

        ead_eff_floor = df[['Netting Set ID','Trade Level Exposure','Trade Level Collateral', 'Haircut']]
        ead_eff_floor['Exposure Num']=ead_eff_floor['Trade Level Exposure'] / (1+ead_eff_floor['Haircut'])
        ead_eff_floor['Collateral Num']=ead_eff_floor['Trade Level Collateral'] / (1+ead_eff_floor['Haircut'])
        df_eff_floor = ead_eff_floor[['Netting Set ID', 'Trade Level Exposure', 'Trade Level Collateral', 'Exposure Num', 'Collateral Num']].pivot_table(
            index=['Netting Set ID'],
            values=['Trade Level Exposure',	'Trade Level Collateral', 'Exposure Num', 'Collateral Num'],
            aggfunc='sum'
        ).reset_index()
        df_eff_floor['Effective Floor'] = (df_eff_floor['Exposure Num']/df_eff_floor['Trade Level Exposure']) / (df_eff_floor['Collateral Num']/df_eff_floor['Trade Level Collateral']) -1

        df_eligibility = df_eff_haircut[['Netting Set ID', 'Effective Haircut']].merge(df_eff_floor[['Netting Set ID', 'Effective Floor']], on='Netting Set ID', how='left')
        df_eligibility['Eligibility'] =  df_eligibility['Effective Haircut'] > df_eligibility['Effective Floor']
        df_N = df_eligibility[['Netting Set ID', 'Eligibility']].merge(df[['Netting Set ID', 'Security ID','Trade Level Exposure', 'Trade Level Collateral']],on='Netting Set ID', how='left')
        df_N["count_flag"] = np.where(df_N["Eligibility"] == False, df_N["Trade Level Exposure"] != 0, (df_N["Trade Level Exposure"] != 0) | (df_N["Trade Level Collateral"] != 0))

        df_N["N"] = df_N.groupby("Netting Set ID")["count_flag"].transform("sum")
        df_N_eligibility = df_N.drop(columns="count_flag")[['Netting Set ID', 'Eligibility', 'N']].drop_duplicates()

        df_sec_level_addon = df[['Netting Set ID',  'Net Exp - Scen A Inelligible',
       'Net Exp - Scen B Inelligible', 'Net Exp - Scen A Elligible',
       'Net Exp - Scen B Elligible', 'Gross Exp - Scen A Inelligible',
       'Gross Exp - Scen B Inelligible', 'Gross Exp - Scen A Elligible',
       'Gross Exp - Scen B Elligible']]
        df_sec_level_addon_eligibility = df_N_eligibility.merge(df_sec_level_addon,  on='Netting Set ID', how='left')

        df_sec_level_addon_notelig = df_sec_level_addon_eligibility[df_sec_level_addon_eligibility['Eligibility'] == False][['Netting Set ID', 'N','Net Exp - Scen A Inelligible',
       'Net Exp - Scen B Inelligible', 'Gross Exp - Scen A Inelligible',
       'Gross Exp - Scen B Inelligible']]
        df_sec_addon_notelig = df_sec_level_addon_notelig.pivot_table(
            index=['Netting Set ID','N'],
            values=['Net Exp - Scen A Inelligible',
       'Net Exp - Scen B Inelligible', 'Gross Exp - Scen A Inelligible',
       'Gross Exp - Scen B Inelligible'],
            aggfunc='sum'
        ).reset_index()
        df_sec_addon_notelig['Sec Addon - Scen A'] = 0.4 * np.abs(df_sec_addon_notelig['Net Exp - Scen A Inelligible']) + 0.6 * df_sec_addon_notelig['Gross Exp - Scen A Inelligible']/np.sqrt(df_sec_addon_notelig['N'])
        df_sec_addon_notelig['Sec Addon - Scen B'] = 0.4 * np.abs(df_sec_addon_notelig['Net Exp - Scen B Inelligible']) + 0.6 * df_sec_addon_notelig['Gross Exp - Scen B Inelligible']/np.sqrt(df_sec_addon_notelig['N'])
        df_sec_addon_notelig_final = df_sec_addon_notelig[['Netting Set ID','Sec Addon - Scen A',
       'Sec Addon - Scen B']]
        
        df_sec_level_addon_elig = df_sec_level_addon_eligibility[df_sec_level_addon_eligibility['Eligibility'] == True][['Netting Set ID', 'N','Net Exp - Scen A Elligible',
       'Net Exp - Scen B Elligible', 'Gross Exp - Scen A Elligible', 'Gross Exp - Scen B Elligible']]
        df_sec_addon_elig = df_sec_level_addon_elig.pivot_table(
            index=['Netting Set ID','N'],
            values=['Net Exp - Scen A Elligible', 'Net Exp - Scen B Elligible', 'Gross Exp - Scen A Elligible', 'Gross Exp - Scen B Elligible'],
            aggfunc='sum'
        ).reset_index()
        df_sec_addon_elig['Sec Addon - Scen A'] = 0.4 * np.abs(df_sec_addon_elig['Net Exp - Scen A Elligible']) + 0.6 * df_sec_addon_elig['Gross Exp - Scen A Elligible']/np.sqrt(df_sec_addon_elig['N'])
        df_sec_addon_elig['Sec Addon - Scen B'] = 0.4 * np.abs(df_sec_addon_elig['Net Exp - Scen B Elligible']) + 0.6 * df_sec_addon_elig['Gross Exp - Scen B Elligible']/np.sqrt(df_sec_addon_elig['N'])
        df_sec_addon_elig_final = df_sec_addon_elig[['Netting Set ID','Sec Addon - Scen A',
       'Sec Addon - Scen B']]
        
        df_sec_addon = pd.concat([df_sec_addon_notelig_final,df_sec_addon_elig_final])

        return df_sec_addon

    def _compute_fx_addon(self):
        df = self.df

        df_fx_addon = pd.merge(
            df.pivot_table(index=['Netting Set ID','Exposure Currency'],values=['Trade Level Exposure'],aggfunc='sum').reset_index(),
            df.pivot_table(index=['Netting Set ID','Collateral Currency'],values=['Trade Level Collateral - Scen A','Trade Level Collateral - Scen B'],aggfunc='sum').reset_index(),
            how='outer',
            left_on=['Netting Set ID','Exposure Currency'],
            right_on=['Netting Set ID','Collateral Currency']
        )

        df_fx_addon = pd.merge(
            pd.merge(
            pd.merge(df_fx_addon,
            df[['Netting Set ID','Agr Settlement Ccy code']].drop_duplicates(),
            how='left'),
            df[['Netting Set ID','Scen A MPOR - Recalc']].drop_duplicates(),
            how='left'),
            df[['Netting Set ID','Scen B MPOR - Recalc']].drop_duplicates(),
            how='left'
        )

        for column_name in ['Trade Level Exposure','Trade Level Collateral - Scen A','Trade Level Collateral - Scen B']:
            df_fx_addon[column_name] = df_fx_addon[column_name].fillna(0)

        df_fx_addon['FX Addon Net Amount - Scen A'] = abs(df_fx_addon['Trade Level Exposure']-df_fx_addon['Trade Level Collateral - Scen A'])*0.08*np.sqrt(df_fx_addon['Scen A MPOR - Recalc']/10)
        df_fx_addon['FX Addon Net Amount - Scen B'] = abs(df_fx_addon['Trade Level Exposure']-df_fx_addon['Trade Level Collateral - Scen B'])*0.08*np.sqrt(df_fx_addon['Scen B MPOR - Recalc']/10)

        return df_fx_addon
    

    def _compute_rwa_summary(self):
        df = self.df
        df_sec_addon = self.df_sec_addon
        df_fx_addon = self.df_fx_addon
        
        df_rwa_summary = df.pivot_table(
            index=['Netting Set ID'],
            values=['Trade Level Exposure','Trade Level Collateral - Scen A','Trade Level Collateral - Scen B','SA EAD'],
            aggfunc={'Trade Level Exposure': 'sum','Trade Level Collateral - Scen A': 'sum','Trade Level Collateral - Scen B': 'sum','SA EAD': 'mean'}
        )

        df_rwa_summary = pd.merge(
            df_rwa_summary,
            df_sec_addon.pivot_table(index=['Netting Set ID'],values=['Sec Addon - Scen A', 'Sec Addon - Scen B'],aggfunc='sum'),
            left_index=True,
            right_index=True
        )

        df_rwa_summary = pd.merge(
            df_rwa_summary,
            df_fx_addon.pivot_table(index=['Netting Set ID'],values=['FX Addon Net Amount - Scen A','FX Addon Net Amount - Scen B'],aggfunc='sum'),
            left_index=True,
            right_index=True
        )

        df_rwa_summary = df_rwa_summary.fillna(0)

        df_rwa_summary['EAD - Scen A'] = df_rwa_summary.apply(lambda row: max(0,row['Trade Level Exposure']-row['Trade Level Collateral - Scen A']+row['Sec Addon - Scen A']+row['FX Addon Net Amount - Scen A']),axis=1)
        df_rwa_summary['EAD - Scen B'] = df_rwa_summary.apply(lambda row: max(0,row['Trade Level Exposure']-row['Trade Level Collateral - Scen B']+row['Sec Addon - Scen B']+row['FX Addon Net Amount - Scen B']),axis=1)
        # df_rwa_summary['EAD'] = df_rwa_summary.apply(lambda row: min(row['EAD - Scen A'],row['EAD - Scen B']),axis=1)
        df_rwa_summary['Chosen Scenario'] = np.where(df_rwa_summary['EAD - Scen A'] <= df_rwa_summary['EAD - Scen B'],'A','B')

        result = []
        for _, row in df_rwa_summary.iterrows():
            scen = row['Chosen Scenario']
            data = {
                'Netting Set ID': _,
                # 'Chosen Scenario': scen,
                'Exposure': row['Trade Level Exposure'],
                'Collateral': row[f'Trade Level Collateral - Scen {scen}'],
                'Sec Addon': row[f'Sec Addon - Scen {scen}'],
                'FX Addon': row[f'FX Addon Net Amount - Scen {scen}'],
                'EAD': row[f'EAD - Scen {scen}'],
                'SA EAD': row['SA EAD']
            }
            result.append(data)

        df_final = pd.DataFrame(result)
        return df_final[['Netting Set ID','Exposure','Collateral','Sec Addon','FX Addon','EAD', 'SA EAD']]

# %%
if __name__ == "__main__":
    df_old = pd.read_csv("c:/Users/katli/RWA Explain/RWA_Projects/Data1 v5.csv")
    df_new = pd.read_csv("c:/Users/katli/RWA Explain/RWA_Projects/Data2 v5.csv")
    calc_old = CalcEAD(df_old)
    calc_new = CalcEAD(df_new)
# %%
