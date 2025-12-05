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

        if row['Haircut Eligible Status']=='Ineligible':
            sec_addon_scen_a = (sec_addon_exposure - sec_addon_coll_scen_a)*row['Haircut']
            sec_addon_scen_b = (sec_addon_exposure - sec_addon_coll_scen_b)*row['Haircut']
        else:
            sec_addon_scen_a = (sec_addon_exposure - sec_addon_coll_scen_a)*row['Haircut']*np.sqrt(scen_a_mpor/10)
            sec_addon_scen_b = (sec_addon_exposure - sec_addon_coll_scen_b)*row['Haircut']*np.sqrt(scen_b_mpor/10)

        return pd.Series({
            'Scen A MPOR - Recalc': scen_a_mpor,
            'Scen B MPOR - Recalc': scen_b_mpor,
            'Trade Level Exposure': trade_exposure,
            'Trade Level Collateral': trade_collateral,
            'Trade Level Collateral - Scen A': eligible_collateral,
            'Trade Level Collateral - Scen B': collateral_scen_b,
            'Sec Addon - Scen A': sec_addon_scen_a,
            'Sec Addon - Scen B': sec_addon_scen_b
        })

    def _compute_sec_addon(self):
        df = self.df
        df_sec_addon = df.pivot_table(
            index=['Netting Set ID','Security ID'],
            values=['Sec Addon - Scen A', 'Sec Addon - Scen B'],
            aggfunc='sum'
        ).reset_index()
        df_sec_addon['Sec Addon Net Amount - Scen A'] = abs(df_sec_addon['Sec Addon - Scen A'])
        df_sec_addon['Sec Addon Net Amount - Scen B'] = abs(df_sec_addon['Sec Addon - Scen B'])
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
            df_sec_addon.pivot_table(index=['Netting Set ID'],values=['Sec Addon Net Amount - Scen A','Sec Addon Net Amount - Scen B'],aggfunc='sum'),
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

        df_rwa_summary['EAD - Scen A'] = df_rwa_summary.apply(lambda row: max(0,row['Trade Level Exposure']-row['Trade Level Collateral - Scen A']+row['Sec Addon Net Amount - Scen A']+row['FX Addon Net Amount - Scen A']),axis=1)
        df_rwa_summary['EAD - Scen B'] = df_rwa_summary.apply(lambda row: max(0,row['Trade Level Exposure']-row['Trade Level Collateral - Scen B']+row['Sec Addon Net Amount - Scen B']+row['FX Addon Net Amount - Scen B']),axis=1)
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
                'Sec Addon': row[f'Sec Addon Net Amount - Scen {scen}'],
                'FX Addon': row[f'FX Addon Net Amount - Scen {scen}'],
                'EAD': row[f'EAD - Scen {scen}'],
                'SA EAD': row['SA EAD']
            }
            result.append(data)

        df_final = pd.DataFrame(result)
        return df_final[['Netting Set ID','Exposure','Collateral','Sec Addon','FX Addon','EAD', 'SA EAD']]

# %%
if __name__ == "__main__":
    df_old = pd.read_csv('data/Data1 v4.csv')
    df_new = pd.read_csv('data/Data2 v4.csv')
    calc_old = CalcEAD(df_old)
    calc_new = CalcEAD(df_new)