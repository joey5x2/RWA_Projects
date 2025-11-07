
import pandas as pd
import numpy as np
from typing import Optional, List

is_number = lambda x: isinstance(x, (int, float, complex))

class CalcEAD():

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # Strip leading/trailing spaces from column names
        self.df.columns = df.columns.str.lstrip(' ')
        self.df.columns = df.columns.str.rstrip(' ')

        # Clean numeric inputs
        for column_name in ['Principal', 'Market Value']:
            self.df[column_name] = self.df[column_name].apply(
                lambda x: 0 if not is_number(x) else x
            )

        # Per-trade components
        self.df = pd.concat(
            [self.df, self.df.apply(self.calc_txn_components, axis=1)],
            axis=1
        )

        # Add-on components
        self.df_sec_addon = self._compute_sec_addon()
        self.df_fx_addon = self._compute_fx_addon()

        # Final RWA summary (this is what the Streamlit app uses)
        self.df_rwa_summary = self._compute_rwa_summary()

    def calc_txn_components(self, row):
        # Trade-level exposure & collateral
        if row['Buy or Sell Indicator'] == 'B':
            trade_exposure = row['Principal']
            trade_collateral = row['Market Value']
        elif row['Buy or Sell Indicator'] == 'S':
            trade_exposure = row['Market Value']
            trade_collateral = row['Principal']
        else:
            trade_exposure = 'DQ'
            trade_collateral = 'DQ'

        # Eligible collateral
        if row['Buy or Sell Indicator'] == 'S':
            eligible_collateral = trade_collateral
        elif (
            row['LRM Flag'] == 'N'
            or row['Dsft Base Conc'] == 'NETTING-INELIGIBLE'
            or row['Haircut Eligible Status'] == 'Ineligible'
            or row['Stale prc flg 2days'] == 'Y'
        ):
            eligible_collateral = 0
        else:
            eligible_collateral = trade_collateral

        # Scenario B collateral
        if row['Buy or Sell Indicator'] == 'B' and row['Illiquid Flag'] == 'Y':
            collateral_scen_b = 0
        else:
            collateral_scen_b = eligible_collateral

        # Sec addon legs
        if row['Buy or Sell Indicator'] == 'S':
            sec_addon_exposure = row['Market Value']
            sec_addon_coll_scen_a = 0
            sec_addon_coll_scen_b = 0
        else:
            sec_addon_exposure = 0
            sec_addon_coll_scen_a = eligible_collateral
            sec_addon_coll_scen_b = collateral_scen_b

        # Sec addon formulas
        if row['Haircut Eligible Status'] == 'Ineligible':
            sec_addon_scen_a = (sec_addon_exposure - sec_addon_coll_scen_a) * row['Haircut']
            sec_addon_scen_b = (sec_addon_exposure - sec_addon_coll_scen_b) * row['Haircut']
        else:
            sec_addon_scen_a = (
                (sec_addon_exposure - sec_addon_coll_scen_a)
                * row['Haircut']
                * np.sqrt(row['Scen A MPOR'] / 10)
            )
            sec_addon_scen_b = (
                (sec_addon_exposure - sec_addon_coll_scen_b)
                * row['Haircut']
                * np.sqrt(row['Scen B MPOR'] / 10)
            )

        return pd.Series(
            {
                'Trade Level Exposure': trade_exposure,
                'Trade Level Collateral': trade_collateral,
                'Trade Level Collateral - Scen A': eligible_collateral,
                'Trade Level Collateral - Scen B': collateral_scen_b,
                'Sec Addon - Scen A': sec_addon_scen_a,
                'Sec Addon - Scen B': sec_addon_scen_b,
            }
        )

    def _compute_sec_addon(self):
        df = self.df
        df_sec_addon = (
            df.pivot_table(
                index=['Netting Set ID', 'Security ID'],
                values=['Sec Addon - Scen A', 'Sec Addon - Scen B'],
                aggfunc='sum',
            )
            .reset_index()
        )
        df_sec_addon['Sec Addon Net Amount - Scen A'] = df_sec_addon['Sec Addon - Scen A'].abs()
        df_sec_addon['Sec Addon Net Amount - Scen B'] = df_sec_addon['Sec Addon - Scen B'].abs()
        return df_sec_addon

    def _compute_fx_addon(self):
        df = self.df

        # Base exposures & collateral by currency
        df_fx_addon = pd.merge(
            df.pivot_table(
                index=['Netting Set ID', 'Exposure Currency'],
                values=['Trade Level Exposure'],
                aggfunc='sum',
            ).reset_index(),
            df.pivot_table(
                index=['Netting Set ID', 'Collateral Currency'],
                values=['Trade Level Collateral - Scen A', 'Trade Level Collateral - Scen B'],
                aggfunc='sum',
            ).reset_index(),
            how='outer',
            left_on=['Netting Set ID', 'Exposure Currency'],
            right_on=['Netting Set ID', 'Collateral Currency'],
        )

        # Join settlement ccy + MPORs
        df_fx_addon = pd.merge(
            pd.merge(
                pd.merge(
                    df_fx_addon,
                    df[['Netting Set ID', 'Agr Settlement Ccy code']].drop_duplicates(),
                    how='left',
                ),
                df[['Netting Set ID', 'Scen A MPOR']].drop_duplicates(),
                how='left',
            ),
            df[['Netting Set ID', 'Scen B MPOR']].drop_duplicates(),
            how='left',
        )

        # Fill NaNs for numeric calc
        for column_name in [
            'Trade Level Exposure',
            'Trade Level Collateral - Scen A',
            'Trade Level Collateral - Scen B',
        ]:
            df_fx_addon[column_name] = df_fx_addon[column_name].fillna(0)

        # FX addon formulas
        df_fx_addon['FX Addon Net Amount - Scen A'] = (
            (df_fx_addon['Trade Level Exposure'] - df_fx_addon['Trade Level Collateral - Scen A'])
            .abs()
            * 0.08
            * np.sqrt(df_fx_addon['Scen A MPOR'] / 10)
        )
        df_fx_addon['FX Addon Net Amount - Scen B'] = (
            (df_fx_addon['Trade Level Exposure'] - df_fx_addon['Trade Level Collateral - Scen B'])
            .abs()
            * 0.08
            * np.sqrt(df_fx_addon['Scen B MPOR'] / 10)
        )

        return df_fx_addon

    def _compute_rwa_summary(self):
        df = self.df
        df_sec_addon = self.df_sec_addon
        df_fx_addon = self.df_fx_addon

        # Base netting-set aggregation
        df_rwa_summary = df.pivot_table(
            index=['Netting Set ID'],
            values=[
                'Trade Level Exposure',
                'Trade Level Collateral - Scen A',
                'Trade Level Collateral - Scen B',
            ],
            aggfunc='sum',
        )

        # Securities addon at netting-set level
        df_rwa_summary = pd.merge(
            df_rwa_summary,
            df_sec_addon.pivot_table(
                index=['Netting Set ID'],
                values=['Sec Addon Net Amount - Scen A', 'Sec Addon Net Amount - Scen B'],
                aggfunc='sum',
            ),
            left_index=True,
            right_index=True,
        )

        # FX addon at netting-set level
        df_rwa_summary = pd.merge(
            df_rwa_summary,
            df_fx_addon.pivot_table(
                index=['Netting Set ID'],
                values=['FX Addon Net Amount - Scen A', 'FX Addon Net Amount - Scen B'],
                aggfunc='sum',
            ),
            left_index=True,
            right_index=True,
        )

        # EAD by scenarios
        df_rwa_summary['EAD - Scen A'] = df_rwa_summary.apply(
            lambda row: max(
                0,
                row['Trade Level Exposure']
                - row['Trade Level Collateral - Scen A']
                + row['Sec Addon Net Amount - Scen A']
                + row['FX Addon Net Amount - Scen A'],
            ),
            axis=1,
        )
        df_rwa_summary['EAD - Scen B'] = df_rwa_summary.apply(
            lambda row: max(
                0,
                row['Trade Level Exposure']
                - row['Trade Level Collateral - Scen B']
                + row['Sec Addon Net Amount - Scen B']
                + row['FX Addon Net Amount - Scen B'],
            ),
            axis=1,
        )

        # Final EAD as min across scenarios
        df_rwa_summary['Reported EAD'] = df_rwa_summary.apply(
            lambda row: min(row['EAD - Scen A'], row['EAD - Scen B']),
            axis=1,
        )

        # ✅ Make 'Netting Set ID' an explicit column (not just index)
        if 'Netting Set ID' not in df_rwa_summary.columns:
            # index is currently Netting Set ID
            df_rwa_summary = df_rwa_summary.reset_index()
        else:
            # already a column; just ensure clean index
            df_rwa_summary = df_rwa_summary.reset_index(drop=True)

        return df_rwa_summary