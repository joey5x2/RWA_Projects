# import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass
from abc import abstractmethod


class Subscriber:
    def __init__(self, netting_id:str) -> None:
        self.res_df = pd.DataFrame()
        self.netting_id = netting_id
    def post_res(self, trade_res:dict):
        self.res_df = pd.concat([self.res_df, pd.Series(trade_res)], axis=1)
    def output_results(self):
        return self.res_df.T
    def reset(self):
        self.res_df = pd.DataFrame()


@dataclass
class CSA:
    margin_id: str
    linked_netting_id: str
    is_margined: str 
    NICA: float
    VM: float=0.0
    VMT: float=0.0
    MTA: float=0.0
    System_EAD: float=0.0

    def get_colleteral(self):
        if self.is_margined == "Margined":
            return self.VM + self.NICA
        else:
            return self.NICA


@dataclass
class Trade:
    trade_id: str
    linked_netting_id: str
    linked_csa: CSA
    longshort: str
    underlying_ticker: str
    asset_class: str
    asset_category: str
    asset_type: str
    mv: float
    trade_notional: float
    maturity: float
    system_MPOR: float
    MPOR: float
    sup_param: dict
    ircr_start_date: float
    ircr_end_date: float
    is_option: bool
    op_P: float
    op_K: float
    op_T: float
    op_CallPut: str
    is_cdo: bool
    cdo_att_point: float
    cdo_det_point: float
    is_vol: bool
    is_basis: bool
    system_delta: float
    is_client_facing: bool
    margin_freq: int 
    is_LNS: bool 
    is_difficult_replace: bool 
    is_illiquid: bool
    is_margin_dispute: bool 
    
    
    subscriber: Subscriber

    def get_trade_addon(self):
        adjusted_notional = self.get_adjusted_notional()
        mf = self.get_maturity_factor()
        sup_delta = self.get_delta_adjustment()
        sf = self.get_supervisory_factor()
        addon = adjusted_notional * mf * sup_delta * sf
        print(f"trade id: {self.trade_id} -> adj notional: {adjusted_notional}; system_MPOR: {self.system_MPOR}; calculated MPOR: {self.MPOR}; maturity factor: {mf}; supervisory delta: {sup_delta}; system delta: {self.system_delta};\
              supervisory factor: {sf}, addon: {addon}, model: 'SA-CCR'")
        self.subscriber.post_res({'Type': 'Trade', 'Trade_ID': self.trade_id, 'Model':'SA-CCR', 'Underlying_Ticker': self.underlying_ticker, 'Adjusted_Notional': adjusted_notional, 'System_MPOR': self.system_MPOR, 'Calculated_MPOR': self.MPOR,\
                                  'Maturity_Factor': mf, 'Supervisory_Delta': sup_delta, 'System_Delta': self.system_delta, 'Supervisory_Factor': sf, 'Trade_Addon': addon})
        return addon

    def get_adjusted_notional(self):
        if self.asset_class == "Interest rate" or self.asset_class == "Credit, single name" or self.asset_class == "Credit, index":
            supervisory_duration = max((np.exp(-0.05*self.ircr_start_date/250) - np.exp(-0.05*self.ircr_end_date/250)) / 0.05, 0.04)
            adjusted_notional = supervisory_duration * self.trade_notional
        else:
            adjusted_notional = self.trade_notional

        return adjusted_notional

    def get_maturity_factor(self) -> float:
        if self.linked_csa.is_margined == "Margined":
            # Margined
            mf = 3/2 * np.sqrt(self.get_mpor()/250)
        else:
            # Unmargined
            mf = np.sqrt(min(max(self.maturity, 10),250)/250)

        return mf

    def get_option_sigma(self) -> float:
        sigma = self.sup_param[self.asset_class][self.asset_category][self.asset_type]['Option_volatility']
        return sigma

    def get_delta_adjustment(self) -> float:
        if self.is_option:
            if self.asset_class == "Interest Rate":
                lambda_factor = max(-min(self.op_P, self.op_K) + 0.001, 0.0)
            else:
                lambda_factor = max(-1.1*min(self.op_P, self.op_K), 0.0)
            option_sigma = self.get_option_sigma()
            if (self.longshort == "Long") & (self.op_CallPut == "Call"):
                delta = norm.cdf((np.log((self.op_P+lambda_factor)/(self.op_K+lambda_factor)) + 0.5*option_sigma**2*self.op_T/250) / (option_sigma*np.sqrt(self.op_T/250)))
            elif (self.longshort == "Long") & (self.op_CallPut == "Put"):
                delta = -norm.cdf(-(np.log((self.op_P+lambda_factor)/(self.op_K+lambda_factor)) + 0.5*option_sigma**2*self.op_T/250) / (option_sigma*np.sqrt(self.op_T/250)))
            elif (self.longshort == "Short") & (self.op_CallPut == "Call"):
                delta = -norm.cdf((np.log((self.op_P+lambda_factor)/(self.op_K+lambda_factor)) + 0.5*option_sigma**2*self.op_T/250) / (option_sigma*np.sqrt(self.op_T/250)))
            else: 
                delta = norm.cdf(-(np.log((self.op_P+lambda_factor)/(self.op_K+lambda_factor)) + 0.5*option_sigma**2*self.op_T/250) / (option_sigma*np.sqrt(self.op_T/250)))
        elif self.is_cdo:
            if self.longshort == "Long":
                delta = 15 / ((1 + 14 * self.cdo_att_point) * (1 + 14 * self.cdo_det_point))
            else:
                delta = - 15 / ((1 + 14 * self.cdo_att_point) * (1 + 14 * self.cdo_det_point))
        else:
            if self.longshort == "Long":
                delta = 1.0
            else:
                delta = -1.0
        return delta

    def get_correlation_factor(self) -> float:
        rho = self.sup_param[self.asset_class][self.asset_category][self.asset_type]['Correlation_factor']
        return rho
    
    def get_supervisory_factor(self) -> float:
        sf = self.sup_param[self.asset_class][self.asset_category][self.asset_type]['Supervisory_factor']
        if self.is_vol:
            sf = sf * 5
        if self.is_basis:
            sf = sf * 0.5
        return sf
    
    def get_mpor(self) -> int:
        self.MPOR = (2 if self.is_margin_dispute else 1) * (20 if self.is_LNS == "Y" or self.is_difficult_replace or self.is_illiquid else (5 if self.is_client_facing else 10)) + self.margin_freq - 1
        return self.MPOR
        # return self.system_MPOR
    
class HedgingSet:
    def __init__(self, name:str, subscriber: Subscriber):
        self.trade_list = []
        self.name = name
        self.subscriber = subscriber
    @abstractmethod
    def add_trade(self):
        pass
    @abstractmethod
    def get_hedging_set_amount(self):
        pass

class HedgingSetIR(HedgingSet):
    def __init__(self, name:str, subscriber: Subscriber):
        super().__init__(name, subscriber)
        self.bucket1 = [] # less than 1 year
        self.bucket2 = [] # 1 - 5 years
        self.bucket3 = [] # more than 5 years
    
    def add_trade(self, one_trade:Trade):
        self.trade_list.append(one_trade)
        if one_trade.ircr_end_date < 250:
            self.bucket1.append(one_trade)
        elif 250 <= one_trade.ircr_end_date < 250*5:
            self.bucket2.append(one_trade)
        else:
            self.bucket3.append(one_trade)
    
    def get_hedging_set_amount(self, use_offset=True):
        if not self.trade_list:
            return 0
        D_b1 = sum([trade.get_trade_addon() for trade in self.bucket1])
        D_b2 = sum([trade.get_trade_addon() for trade in self.bucket2])
        D_b3 = sum([trade.get_trade_addon() for trade in self.bucket3])

        if use_offset:
            amount = np.sqrt(D_b1**2+D_b2**2+D_b3**2 + 1.4*D_b1*D_b2 + 1.4*D_b2*D_b3 + 0.6*D_b1*D_b3)
        else:
            amount = abs(D_b1) + abs(D_b2) + abs(D_b3)
        print(f"IR Hedging Set: {self.name}, hedging set amount: {amount}")
        self.subscriber.post_res({'Type': 'IR Hedging Set', 'ID': self.name, 'Model':'SA-CCR', 'HedingSet_amount': amount})

        return amount

class HedgingSetFX(HedgingSet):
    def __init__(self, name:str, subscriber: Subscriber):
        super().__init__(name, subscriber)
    def add_trade(self, one_trade: Trade):
        self.trade_list.append(one_trade)
    def get_hedging_set_amount(self):
        if not self.trade_list:
            return 0
        effective_notional = sum([trade.get_trade_addon() for trade  in self.trade_list])
        amount = abs(effective_notional)
        print(f"FX Hedging Set: {self.name}, hedging set amount: {amount}")
        self.subscriber.post_res({'Type': 'FX Hedging Set', 'ID': self.name, 'Model':'SA-CCR', 'HedingSet_amount': amount})
        return amount

class HedgingSetCR(HedgingSet):
    def __init__(self, name:str, subscriber: Subscriber):
        super().__init__(name, subscriber)
        self.entity_buckets = dict()
    def add_trade(self, one_trade: Trade):
        self.trade_list.append(one_trade)
        self.entity_buckets.setdefault(one_trade.underlying_ticker, ([], one_trade.get_correlation_factor()))[0].append(one_trade)
    def get_hedging_set_amount(self):
        if not self.trade_list:
            return 0
        entity_summary = {entity_name: (sum([trade.get_trade_addon() for trade in et_trades]),  rho) for entity_name, (et_trades, rho) in self.entity_buckets.items()}
        print("###", entity_summary)
        sum_one = 0
        sum_two = 0
        for entity_name, (et_addon, rho) in entity_summary.items():
            sum_one += rho * et_addon
            sum_two += (1 - rho ** 2) * et_addon ** 2
        amount = np.sqrt(sum_one ** 2 + sum_two)
        print(f"CR Hedging Set: {self.name}, hedging set amount: {amount}")
        self.subscriber.post_res({'Type': 'CR Hedging Set', 'ID': self.name, 'Model':'SA-CCR', 'HedingSet_amount': amount})
        return amount

class HedgingSetEQ(HedgingSet):
    def __init__(self, name:str, subscriber: Subscriber):
        super().__init__(name, subscriber)
        self.entity_buckets = dict()
    def add_trade(self, one_trade: Trade):
        self.trade_list.append(one_trade)
        self.entity_buckets.setdefault(one_trade.underlying_ticker, ([], one_trade.get_correlation_factor()))[0].append(one_trade)
    def get_hedging_set_amount(self):
        if not self.trade_list:
            return 0
        entity_summary = {entity_name: (sum([trade.get_trade_addon() for trade in et_trades]),  rho) for entity_name, (et_trades, rho) in self.entity_buckets.items()}
        sum_one = 0
        sum_two = 0
        for entity_name, (et_addon, rho) in entity_summary.items():
            sum_one += rho * et_addon
            sum_two += (1 - rho ** 2) * et_addon ** 2
        amount = np.sqrt(sum_one ** 2 + sum_two)
        print(f"EQ Hedging Set: {self.name}, hedging set amount: {amount}")
        self.subscriber.post_res({'Type': 'EQ Hedging Set', 'ID': self.name, 'Model':'SA-CCR', 'HedingSet_amount': amount})
        return amount

class HedgingSetCM(HedgingSet):
    def __init__(self, name:str, subscriber: Subscriber):
        super().__init__(name, subscriber)
        self.entity_buckets = dict()
    def add_trade(self, one_trade: Trade):
        self.trade_list.append(one_trade)
        self.entity_buckets.setdefault(one_trade.underlying_ticker, ([], one_trade.get_correlation_factor()))[0].append(one_trade)
    def get_hedging_set_amount(self):
        if not self.trade_list:
            return 0
        entity_summary = {entity_name: (sum([trade.get_trade_addon() for trade in et_trades]),  rho) for entity_name, (et_trades, rho) in self.entity_buckets.items()}
        sum_one = 0
        sum_two = 0
        for entity_name, (et_addon, rho) in entity_summary.items():
            sum_one += rho * et_addon
            sum_two += (1 - rho ** 2) * et_addon ** 2
        amount = np.sqrt(sum_one ** 2 + sum_two)
        print(f"CM Hedging Set: {self.name}, hedging set amount: {amount}")
        self.subscriber.post_res({'Type': 'CM Hedging Set', 'ID': self.name, 'Model':'SA-CCR', 'HedingSet_amount': amount})
        return amount



# define netting set classes -----------------
class Netting_set:
    def __init__(self, netting_id:str, csa:CSA, subscriber: Subscriber):
        self.netting_id = netting_id
        self.csa = csa
        self.subscriber = subscriber
        self.ir_hedging_sets = dict()
        self.fx_hedging_sets = dict()
        # self.cr_hedging_set = HedgingSetCR(name="CR_Single_Set", subscriber=self.subscriber)
        # self.eq_hedging_set = HedgingSetEQ(name="EQ_Single_Set", subscriber=self.subscriber)
        self.cr_hedging_sets = dict()
        self.eq_hedging_sets = dict()
        self.cm_hedging_sets = dict()
        self.trade_list = []
    
    def add_trade(self, one_trade: Trade):
        self.trade_list.append(one_trade)
        # Volatility trade
        if one_trade.is_vol and not one_trade.is_basis:
            if one_trade.asset_class == "Interest rate":
                self.ir_hedging_sets.setdefault(one_trade.underlying_ticker+'_vol', HedgingSetIR(name=one_trade.underlying_ticker+'_vol', subscriber=self.subscriber)).add_trade(one_trade)
            elif one_trade.asset_class == "Exchange rate":
                self.fx_hedging_sets.setdefault(one_trade.underlying_ticker+'_vol', HedgingSetFX(name=one_trade.underlying_ticker+'_vol', subscriber=self.subscriber)).add_trade(one_trade)
            elif (one_trade.asset_class == "Credit, single name") or (one_trade.asset_class == "Credit, index"):
                self.cr_hedging_sets.setdefault('CR_Vol_Set', HedgingSetCR(name='CR_Vol_Set', subscriber=self.subscriber)).add_trade(one_trade)
            elif (one_trade.asset_class == "Equity, single name") or (one_trade.asset_class == "Equity, index"):
                self.eq_hedging_sets.setdefault('EQ_Vol_Set', HedgingSetEQ(name='EQ_Vol_Set', subscriber=self.subscriber)).add_trade(one_trade)
            elif one_trade.asset_class == "Commodity":
                self.cm_hedging_sets.setdefault(one_trade.asset_category+'_vol', HedgingSetCM(name=one_trade.asset_category, subscriber=self.subscriber)).add_trade(one_trade)
            else:
                raise ValueError("Wrong Asset Class Name")

        elif one_trade.is_basis and not one_trade.is_vol:
        # Basis Trade
            if one_trade.asset_class == "Interest rate":
                self.ir_hedging_sets.setdefault(one_trade.underlying_ticker+'_basis', HedgingSetIR(name=one_trade.underlying_ticker+'_basis', subscriber=self.subscriber)).add_trade(one_trade)
            elif one_trade.asset_class == "Exchange rate":
                self.fx_hedging_sets.setdefault(one_trade.underlying_ticker+'_basis', HedgingSetFX(name=one_trade.underlying_ticker+'_basis', subscriber=self.subscriber)).add_trade(one_trade)
            elif (one_trade.asset_class == "Credit, single name") or (one_trade.asset_class == "Credit, index"):
                self.cr_hedging_sets.setdefault(one_trade.underlying_ticker+'_basis', HedgingSetCR(name=one_trade.underlying_ticker+'_basis', subscriber=self.subscriber)).add_trade(one_trade)
            elif (one_trade.asset_class == "Equity, single name") or (one_trade.asset_class == "Equity, index"):
                self.eq_hedging_sets.setdefault(one_trade.underlying_ticker+'_basis', HedgingSetEQ(name=one_trade.underlying_ticker+'_basis', subscriber=self.subscriber)).add_trade(one_trade)
            elif one_trade.asset_class == "Commodity":
                self.cm_hedging_sets.setdefault(one_trade.underlying_ticker+'_basis', HedgingSetCM(name=one_trade.underlying_ticker+'_basis', subscriber=self.subscriber)).add_trade(one_trade)
            else:
                raise ValueError("Wrong Asset Class Name")

        elif not one_trade.is_basis and not one_trade.is_vol:
            if one_trade.asset_class == "Interest rate":
                self.ir_hedging_sets.setdefault(one_trade.underlying_ticker, HedgingSetIR(name=one_trade.underlying_ticker, subscriber=self.subscriber)).add_trade(one_trade)
            elif one_trade.asset_class == "Exchange rate":
                self.fx_hedging_sets.setdefault(one_trade.underlying_ticker, HedgingSetFX(name=one_trade.underlying_ticker, subscriber=self.subscriber)).add_trade(one_trade)
            elif (one_trade.asset_class == "Credit, single name") or (one_trade.asset_class == "Credit, index"):
                self.cr_hedging_sets.setdefault('CR_Regular_Set', HedgingSetCR(name='CR_Regular_Set', subscriber=self.subscriber)).add_trade(one_trade)
            elif (one_trade.asset_class == "Equity, single name") or (one_trade.asset_class == "Equity, index"):
                self.eq_hedging_sets.setdefault('EQ_Regular_Set', HedgingSetEQ(name='EQ_Regular_Set', subscriber=self.subscriber)).add_trade(one_trade)
            elif one_trade.asset_class == "Commodity":
                self.cm_hedging_sets.setdefault(one_trade.asset_category, HedgingSetCM(name=one_trade.asset_category, subscriber=self.subscriber)).add_trade(one_trade)
            else:
                raise ValueError("Wrong Asset Class Name")
        
        else:
            raise ValueError("Wrong trade type (Vol/Basis/Regular)")
    
    def get_addon(self):
        print(self.cr_hedging_sets)
        # IR
        if len(self.ir_hedging_sets) != 0:
            IR_addons = sum([hs.get_hedging_set_amount() for hs in self.ir_hedging_sets.values()])
        else:
            IR_addons = 0
        # FX
        if len(self.fx_hedging_sets) != 0:
            FX_addons = sum([hs.get_hedging_set_amount() for hs in self.fx_hedging_sets.values()])
        else:
            FX_addons = 0
        # CR
        if len(self.cr_hedging_sets) != 0:
            CR_addons = sum([hs.get_hedging_set_amount() for hs in self.cr_hedging_sets.values()])
        else:
            CR_addons = 0
        # EQ
        if len(self.eq_hedging_sets) != 0:
            EQ_addons = sum([hs.get_hedging_set_amount() for hs in self.eq_hedging_sets.values()])
        else:
            EQ_addons = 0
        # CM
        if len(self.cm_hedging_sets) != 0:
            CM_addons = sum([hs.get_hedging_set_amount() for hs in self.cm_hedging_sets.values()])
        else:
            CM_addons = 0

        netting_set_addon = IR_addons + FX_addons + CR_addons + EQ_addons + CM_addons
        return netting_set_addon

    def get_V(self):
        V = sum([trade.mv for trade in self.trade_list])
        return V

    def get_RC(self):
        V = self.get_V()
        C = self.csa.get_colleteral()
        if self.csa.is_margined == "Margined":
            RC = max(V - C, self.csa.VMT + self.csa.MTA - self.csa.NICA, 0)
        else:
            RC = max(V - C, 0)
        return RC

    def compute_multiplier(self, Addon):
        V = self.get_V()
        C = self.csa.get_colleteral()
        multiplier = min(1, 0.05+0.95*np.exp((V-C)/(1.9*Addon)))
        return multiplier

    def get_EAD(self):
        RC = self.get_RC()
        PFE_addon = self.get_addon()
        multiplier = self.compute_multiplier(PFE_addon)
        EAD = 1.4 * (RC + multiplier * PFE_addon)
        print(f"RC:{RC}, PFE_addon:{PFE_addon}, EAD: {EAD}")
        self.subscriber.post_res({'Type': 'Netting Set', 'ID': self.netting_id, 'Model':'SA-CCR', 'Replacement_Cost': RC, 'PFE_addon': PFE_addon, 'multiplier': multiplier, 'EAD': EAD})
        return EAD

def sup_para_df2dict(parameters_df_raw:pd.DataFrame) -> dict:
    parameters_df = parameters_df_raw.copy()
    parameters_df = parameters_df.fillna('N/A')
    params_dict = dict()
    for ind, row in parameters_df.iterrows():
        params_dict.setdefault(row['Asset_class'], {}).setdefault(row['Category'], {}).setdefault(row['Type'], {})['Option_volatility'] = row['Option_volatility']
        params_dict.setdefault(row['Asset_class'], {}).setdefault(row['Category'], {}).setdefault(row['Type'], {})['Correlation_factor'] = row['Correlation_factor']
        params_dict.setdefault(row['Asset_class'], {}).setdefault(row['Category'], {}).setdefault(row['Type'], {})['Supervisory_factor'] = row['Supervisory_factor']
    return params_dict



def feed_data_excel(excel_name:str):
    trades_df = pd.read_excel(excel_name, sheet_name="trades_inputs")
    trades_df[["Category", "Type"]] = trades_df[["Category", "Type"]].fillna("N/A")
    csa_df = pd.read_excel(excel_name, sheet_name="csa_inputs")
    parameters_df = pd.read_excel(excel_name, sheet_name="parameters")
    parameters_dict = sup_para_df2dict(parameters_df)
    # Load all netting sets
    NettingSet_dict = dict()
    all_nettings = trades_df['Netting_ID'].unique().tolist()
    for nid in all_nettings:
        print("=" * 80)
        print(f"netting id: {nid}")
        # Load CSA data
        n_csa = csa_df[csa_df['Netting_ID'] == nid].squeeze()
        this_CSA = CSA(margin_id=n_csa["Margin_ID"],
                       linked_netting_id=n_csa["Netting_ID"],
                       is_margined=n_csa["Margin_status"],
                       NICA=n_csa["NICA"],
                       VM=n_csa["VM"],
                       VMT=n_csa["VMT"],
                       MTA=n_csa["MTA"],
                       System_EAD=n_csa["System_EAD"])
        print(this_CSA)
        # Declare Subscriber
        one_observer = Subscriber(nid)
        # Construct Netting Set
        this_netting_set = Netting_set(netting_id=nid, csa=this_CSA, subscriber=one_observer)
        # Load Trades Data
        n_trades_df = trades_df[trades_df['Netting_ID'] == nid]
        for ind, trade_row in n_trades_df.iterrows():
            one_trade = Trade(trade_id=trade_row["Trade_ID"],
                              linked_netting_id=trade_row["Netting_ID"],
                              linked_csa=this_CSA,
                              longshort=trade_row["Long_Short"],
                              underlying_ticker=trade_row["Underlying_ticker"],
                              asset_class=trade_row["Asset_class"],
                              asset_category=trade_row["Category"],
                              asset_type=trade_row["Type"],
                              mv=trade_row["Market_value"],
                              trade_notional=trade_row["Trade_notional"],
                              maturity=trade_row["Maturity"],
                              system_MPOR=trade_row["System_MPOR"],
                              MPOR=float("nan"),
                              sup_param=parameters_dict,
                              ircr_start_date=trade_row["start_date"],
                              ircr_end_date=trade_row["end_date"],
                              is_option=trade_row["is_option"],
                              op_P=trade_row["Underlying_price"],
                              op_K=trade_row["Strike_price"],
                              op_T=trade_row["Option_maturity"],
                              op_CallPut=trade_row["Call_Put"],
                              is_cdo=trade_row["is_CDO_tranches"],
                              cdo_att_point=trade_row["Attachment_Point"],
                              cdo_det_point=trade_row["Detachment_Point"],
                              is_vol = trade_row['is_volatility_trade'],
                              is_basis = trade_row['is_basis_trade'],
                              system_delta= trade_row['System_delta'],
                              is_client_facing= trade_row['is_client_facing'],
                              margin_freq= trade_row['margin frequency (BD)'], 
                              is_LNS= trade_row['is_large_netting_set'], 
                              is_difficult_replace= trade_row['ns_has_difficult_to_replace_trade'], 
                              is_illiquid= trade_row['ns_has_illiquid_collateral'],
                              is_margin_dispute= trade_row['margin_dispute'], 
                              subscriber=one_observer)
            this_netting_set.add_trade(one_trade)
        NettingSet_dict[nid] = this_netting_set
    return NettingSet_dict


def Cal_EAD(netting_sets:dict):
    res_df = dict()
    ead_summary = pd.DataFrame()
    for ns_id, ns in netting_sets.items():
        ns.subscriber.reset()
        ns.get_EAD()
        # print("NS EAD:", ns.get_EAD())
        df = ns.subscriber.output_results()
        df.reset_index(drop=True, inplace=True)
        # print(df)
        res_df[ns_id] = df
        ns_row = df.loc[df["Type"] == "Netting Set"]
        ns_row['System_EAD'] = ns.csa.System_EAD
        # print(ns_row)
        ead_summary = pd.concat([ead_summary, ns_row], axis=0)
       
        
    ead_summary = ead_summary[["Type", "ID", "Model", "Replacement_Cost", "PFE_addon", "multiplier", "EAD", "System_EAD"]]
    ead_summary.reset_index(drop=True, inplace=True)
    
    return res_df, ead_summary



if __name__ == '__main__':
    all_netting_sets = feed_data_excel('inputs_data1_v4.xlsx')
    # write_results_excel(all_netting_sets, 'outputs.xlsx')
    res_all, ead_summary = Cal_EAD(all_netting_sets)
    print(ead_summary)
    res_trade = pd.DataFrame()
    for k, v in res_all.items():
        df = v.loc[v["Type"] == "Trade"]
        df["Netting Set"] = k
        res_trade = pd.concat([res_trade, df], axis=0)
    res_trade.reset_index(drop=True, inplace=True)
        
        
        




    

    
    
    
