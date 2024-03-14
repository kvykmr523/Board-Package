#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:29:24 2024

@author: kavya
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
import plotly.graph_objects as go
import matplotlib.pyplot as plt 
from matplotlib import colors as mcolors
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PyQt5.QtWebKitWidgets import QWebView

'''
# Overview
The purpose of this Python Script is to:
- Import project data from various sources (e.g. various Reforecast System Tables and the DealPath System)
- Clean up the data and make it 'Python Friendly'
- Perform some calculations and data manipulations
- Create Factor variables for several data elements that can be used in subsequent charting and reporting
 Combine all of the data into the dfProjAll dataframe
'''
# Set up directory paths
root_directory = "/Users/kavya/Dropbox/Board Package Material/CC R Portfolio"
os.chdir(root_directory
         )
data_extract_directory = os.path.join(root_directory, "00 Data Extracts")
csv_files_directory = os.path.join(root_directory, "01 CSV Files")
xlsx_files_directory = os.path.join(root_directory, "02 xlsx Files")
df_files_directory = os.path.join(root_directory, "03 df Files")
xml_files_directory = os.path.join(root_directory, "04 xml Files")
documentation_directory = os.path.join(root_directory, "05 documentation")

# Input CSV files from 01 CSV Files Folder 
iREF_Projects = os.path.join(csv_files_directory, "R_ProjectSummary.csv")
REF_Project = pd.read_csv(iREF_Projects)

iREF_Dealpath = os.path.join(csv_files_directory, "R_Dealpath.csv")
REF_Dealpath = pd.read_csv(iREF_Dealpath)

iREF_Concentrate = os.path.join(csv_files_directory, "R_CashFlowDetail.csv")
REF_Concentrate = pd.read_csv(iREF_Concentrate)

iREF_Summary = os.path.join(csv_files_directory, "R_CashFlowSummary.csv")
REF_Summary = pd.read_csv(iREF_Summary)

# Output files 
oxl_ProjAll = os.path.join(xlsx_files_directory, "dfProjAll.xlsx")
odf_ProjAll = os.path.join(df_files_directory, "dfProjAll.pkl")
oht_EHF1 = os.path.join(df_files_directory, "Portfolio_EHF1.html")
oht_EHF2 = os.path.join(df_files_directory, "Portfolio_EHF2.html")

# Define Global Variables
# Set the date ranges for the last month of the Quarter used in Concentration Balance Reporting
iQtr1 = (datetime(2023, 9, 1), datetime(2023, 9, 30))
iQtr2 = (datetime(2023, 12, 1), datetime(2023, 12, 31))
iQtr3 = (datetime(2024, 3, 1), datetime(2024, 3, 30))

# Working strings for EHF1 and EHF2
iEHF1 = "EHF1"
iEHF2 = "EHF2"

# Set the order of how Builders appear (see f_builder_3letter factor).
# Note: These are the 3Letter codes set by this program during data import
BuilderOrder = ["LEN", "MTH", "HOV", "NHC", "CHM", "LGI", "SDH", "LSE", "TOL", "EMP", "PHM"]

#Dataframe Cleanup 
# DataFrame CleanUp

# Drop unnecessary columns
columns_to_drop = ['createTimestamp', 
                   'hssId', 
                   'clawbackBuilderPercent',
                   'clawbackGrossMarginPercent',
                   'clawbackIRR',
                   'optionTermGuaranteePercent',
                   'lastWorkDay',
                   'clawbackFloor',
                   'takedownAtCOE',
                   'clawback',
                   'hiatus',
                   'optionFeeFloor',
                   'optionFeeCap',
                   'siteDayOfMonth',
                   'targetIRR',
                   'optionFeeInitial',
                   'accessId',
                   'builderId',
                   'fundSubId',
                   'optionDayOfMonth']




dfProject = REF_Project.drop(columns=columns_to_drop)

# Rename selected columns
# Rename selected columns - (just personal preference)
dfProject = dfProject.rename(columns={'id': 'project_id',
                                      'name': 'proj_name',
                                      'termStart': 'd_term_start',
                                      'actualsThruDate': 'd_actuals_thru',
                                      'builderId[Builder]': 'builder',
                                      'name[FundSubsidiary]': 'fundSub',
                                      'name[MetropolitanArea]': 'msa', 
                                      'term' : 'opt_term'})

dfProject['opt_term'] = pd.to_numeric(dfProject['opt_term'])
dfProject['d_term_start'] = pd.to_datetime(dfProject['d_term_start'])

dfProject['d_term_end'] = dfProject['d_term_start'] + dfProject['opt_term'].apply(lambda x:relativedelta(months = x))

date_columns = dfProject.filter(like='d_').columns
dfProject[date_columns] = dfProject[date_columns].apply(pd.to_datetime)

# Filter rows where fund_sub is not "In Process"
dfProject = dfProject[dfProject['fundSub'] != "In Process"]

# Create fundid column
dfProject['fundid'] = dfProject['fundSub'].apply(lambda x: 'EHF1' if 'EHF1' in x else ('EHF2' if 'EHF2' in x else 'EHFX'))

# Create builder_3letter column
dfProject['builder_3letter'] = dfProject['builder']

# Reorder columns
dfProject = dfProject[['project_id', 'proj_name', 'fundid', 'builder_3letter', 'city', 'state', 'msa', 
                       'numLotTypes', 'numLots', 'd_term_start', 'opt_term', 'd_term_end', 
                       'status', 'actualsThruPeriod', 'd_actuals_thru']]

# Control Totals for iREF_Projects by Fund
dfProjectCT1_fund = dfProject.groupby('fundid').agg(tot_lots=('numLots', 'sum'),
                                                    project_count=('project_id', 'count')).reset_index()

# Add totals row
dfProjectCT1_fund.loc['Total'] = dfProjectCT1_fund.sum(numeric_only=True)

# Control Totals for iREF_Projects by Fund and by Fund-Builder
dfProjectCT2 = dfProject.groupby(['fundid', 'builder_3letter']).agg(tot_lots=('numLots', 'sum'),
                                                                  project_count=('project_id', 'count')).reset_index()

# Add totals row
dfProjectCT2.loc['Total'] = dfProjectCT2.sum(numeric_only=True)

# Rename columns
dfProjectCT1_fund.columns = ['Fund ID', 'Total Lots', '# of Projects']
dfProjectCT2.columns = ['Fund ID', 'Builder ID', 'Total Lots', '# of Projects']


# Dataframe CleanUp
dfDealpath = REF_Dealpath.rename(columns={
    'REF_ID': 'project_id',
    'Deal Name': 'dealpath_name',
    'bld_Gross Margin': 'hb_margin_dollars',
    'bld_Gross Margin % of Home Revenue': 'hb_margin_percent'
}).drop(columns=['dealpath_name', 'hb_margin_dollars']).assign(
    hb_margin_percent=lambda x: x['hb_margin_percent'] * 100
)

# Join into dfProject
dfProject = dfProject.merge(
    dfDealpath,
    on="project_id",
    how="inner",
    suffixes=("_X", "_Y"),
    validate="one_to_many"
)

# Drop the duplicated key columns
dfProject = dfProject.loc[:,~dfProject.columns.duplicated()]

# Display the DataFrame information
print(dfProject.info())

# Group by fundid and calculate total lots and project count
dfProjectCT1_fundid = dfProject.groupby('fundid').agg(
    tot_lots=('numLots', 'sum'),
    project_count=('project_id', 'count')
).reset_index()

# Add total row using sum() function
total_row_fund = dfProjectCT1_fundid.sum(numeric_only=True)
total_row_fund['fundid'] = 'Total'
# Add total row using adorn_totals() function
dfProjectCT1_fundid = pd.concat([dfProjectCT1_fundid, total_row_fund.to_frame().transpose()], ignore_index=True)

# Rename columns
dfProjectCT1_fundid.columns = ['Fund ID', 'Total Lots', '# of Projects']

# Add footer line
dfProjectCT1_fundid.loc[len(dfProjectCT1_fundid)] = ['Total', dfProjectCT1_fundid['Total Lots'].sum(), dfProjectCT1_fundid['# of Projects'].sum()]

# Convert DataFrame to a formatted table using tabulate
table_fundid = tabulate(dfProjectCT1_fundid, headers='keys', tablefmt='pretty')

# Print the table
print("iREF_DealPath Control Totals by Fund:")
print(table_fundid)
print()

# Group by fundid and builder_3letter and calculate total lots and project count
dfProjectCT1_builder = dfProject.groupby(['fundid', 'builder_3letter']).agg(
    tot_lots=('numLots', 'sum'),
    project_count=('project_id', 'count')
).reset_index()


# Rename columns
dfProjectCT1_builder.columns = ['Fund ID', 'Builder ID', 'Total Lots', '# of Projects']

# Add footer line
dfProjectCT1_builder.loc[len(dfProjectCT1_builder)] = ['Total', '', dfProjectCT1_builder['Total Lots'].sum(), dfProjectCT1_builder['# of Projects'].sum()]

# Convert DataFrame to a formatted table using tabulate
table_builder = tabulate(dfProjectCT1_builder, headers='keys', tablefmt='pretty')

# Print the table
print("iREF_DealPath Control Totals by Fund and Builder:")
print(table_builder)


# CleanUp Dataframe
dfConcentrate = REF_Concentrate.rename(columns={'julian_date': 'myDate', 'id': 'project_id', 'calc_ConcBalancePlusTermFee': 'concentration_bal'})

dfConcentrate['myDate'] = pd.to_datetime(dfConcentrate['myDate'])
dfConcentrate = dfConcentrate[(dfConcentrate['myDate'] .between(iQtr1[0], iQtr1[1])) |
                              (dfConcentrate['myDate'].between(iQtr2[0], iQtr2[1])) |
                              (dfConcentrate['myDate'].between(iQtr3[0], iQtr3[1]))]


dfConcentrate['myYQ'] = pd.to_datetime(dfConcentrate['myDate']).dt.to_period('Q')

# Step 4: Negate concentration_bal and create myYQname
dfConcentrate['concentration_bal'] = -dfConcentrate['concentration_bal']

# Step 5: Create myYQname based on myDate within iQtr1, iQtr2, or iQtr3
conditions = [
    (dfConcentrate['myDate'].between(iQtr1[0], iQtr1[1])),
    (dfConcentrate['myDate'].between(iQtr2[0], iQtr2[1])),
    (dfConcentrate['myDate'].between(iQtr3[0], iQtr3[1]))
]
choices = ["cb_Qtr1", "cb_Qtr2", "cb_Qtr3"]

# Apply conditions to create myYQname
dfConcentrate['myYQname'] = np.select(conditions, choices, default= "UNKNOWN")

# Drop columns 'name', 'myDate', 'myYQ'
dfConcentrate = dfConcentrate.drop(columns=['name', 'myDate', 'myYQ'])

# Pivot wider
dfConcentrate = dfConcentrate.pivot_table(index='project_id', columns='myYQname', values='concentration_bal', fill_value=0)

# Replace NaN values in specific columns with 0
dfConcentrate['cb_Qtr1'] = dfConcentrate['cb_Qtr1'].fillna(0).astype(int)
dfConcentrate['cb_Qtr2'] = dfConcentrate['cb_Qtr2'].fillna(0).astype(int)
dfConcentrate['cb_Qtr3'] = dfConcentrate['cb_Qtr3'].fillna(0).astype(int)

# Left join dfConcentrate to dfProject
dfProject = dfProject.merge(dfConcentrate, on='project_id', how='left')

# Control totals for dfProjectCT1 by fundid
dfProjectCT1_fundid = dfProject.groupby('fundid').agg(
    tot_lots=('numLots', 'sum'),
    project_count=('project_id', 'nunique'),
    tot_cbQ1=('cb_Qtr1', 'sum'),
    tot_cbQ2=('cb_Qtr2', 'sum'),
    tot_cbQ3=('cb_Qtr3', 'sum')
).reset_index()

# Add total row
dfProjectCT1_fundid.loc['Total'] = dfProjectCT1_fundid.sum(numeric_only=True)

# Display the DataFrame
print(tabulate(dfProjectCT1_fundid, headers='keys', tablefmt='pretty'))

# Control totals for dfProjectCT1 by fundid and builder_3letter
dfProjectCT1_fundid_builder = dfProject.groupby(['fundid', 'builder_3letter']).agg(
    tot_lots=('numLots', 'sum'),
    project_count=('project_id', 'nunique'),
    tot_cbQ1=('cb_Qtr1', 'sum'),
    tot_cbQ2=('cb_Qtr2', 'sum'),
    tot_cbQ3=('cb_Qtr3', 'sum')
).reset_index()

# Add total row
dfProjectCT1_fundid_builder.loc['Total'] = dfProjectCT1_fundid_builder.sum(numeric_only=True)

# Display the DataFrame
print(tabulate(dfProjectCT1_fundid_builder, headers='keys', tablefmt='pretty'))

# Clean up DataFrame
dfxSummary = REF_Summary.rename(columns={'projectId': 'project_id',
                                          'type': 'type',
                                          'Land': 'land',
                                          'Site': 'site',
                                          'MUD': 'mud',
                                          'projectCosts': 'project_costs',
                                          'unleveraged': 'unleveraged',
                                          'first_takedown_period': 'first_takedown_period',
                                          'last_takedown_period': 'last_takedown_period',
                                          'numUnpurchased': 'num_unpurchased',
                                          'lots_in_final_tdown': 'lots_in_final_tdown',
                                          'hit_70': 'hit_70'})

# Select required columns
dfxSummary = dfxSummary[['project_id', 'type', 'land', 'site', 'mud', 'project_costs',
                         'unleveraged', 'first_takedown_period', 'last_takedown_period',
                         'num_unpurchased', 'lots_in_final_tdown', 'hit_70']]


# Create a Proforma Table (SummaryP) and an Actuals Table (SummaryF)
REF_SummaryP = dfxSummary[dfxSummary['type'] == "P"]
REF_SummaryF = dfxSummary[dfxSummary['type'] == "F"]

# Then join these two tables so all columns are in one "row" (dfxSummary)
dfSummary = pd.merge(REF_SummaryP, REF_SummaryF, on="project_id", suffixes=('.P', '.A'), how="inner")

# Join REF dframes into dfProjAll and prepare for Charting    
dfProjAll = pd.merge(dfProject, dfSummary, on="project_id", suffixes=('.1', '.2'), how="inner")

# Convert the start date to a "date type" so it can be formatted for factoring
dfProjAll['OrigQtr'] = pd.to_datetime(dfProjAll['d_term_start']).dt.to_period('Q')
dfProjAll['f_OrigQtr'] = pd.to_datetime(dfProjAll['d_term_start']).dt.to_period('Q').astype(str)
dfProjAll['OrigMth'] = pd.to_datetime(dfProjAll['d_term_start']).dt.to_period('M')
dfProjAll['f_OrigMth'] = pd.to_datetime(dfProjAll['d_term_start']).dt.to_period('M').astype(str)

# Define factor breaks and create factors for different columns
dfProjAll['f_builder_3letter'] = pd.Categorical(dfProjAll['builder_3letter'], categories=BuilderOrder)
dfProjAll['f_state'] = pd.Categorical(dfProjAll['state'])
dfProjAll['f_msa'] = pd.Categorical(dfProjAll['msa'])

dfProjAll['f_optterm'] = pd.cut(dfProjAll['opt_term'], bins=[0, 18, 24, 36, np.inf], 
                                 labels=["<= 18", "19-24", "25-36", "> 36"], right=False)


# For f_ProjLots
bins = [0, 100, 150, 300, 450, 600, np.inf]
labels = ["<= 100", "101-150", "151-300", "301-450", "451-600", "> 600"]
dfProjAll['f_ProjLots'] = pd.cut(dfProjAll['numLots'], bins=bins, labels=labels, right=True)

# For f_UnpurLots
bins = [0, 100, 150, 300, 450, 600, np.inf]
labels = ["<= 100", "101-150", "151-300", "301-450", "451-600", "> 600"]
dfProjAll['f_UnpurLots'] = pd.cut(dfProjAll['num_unpurchased.A'], bins=bins, labels=labels, right=True)

# For f_1st_takedown.P
bins = [0, 6, 9, 12, 15, 18, np.inf]
labels = ["<=6", "7-9", "10-12", "13-15", "16-18", ">18"]
dfProjAll['f_1st_takedown.P'] = pd.cut(dfProjAll['first_takedown_period.P'], bins=bins, labels=labels, right=True)

# For f_hb_margin
bins = [0, 20, 22, 24, 26, 30, np.inf]
labels = ["<=20%", "20.1%-22.0%", "22.1%-24.0%", "24.1%-26.0%", "26.1%-30%", ">30.0%"]
dfProjAll['f_hb_margin'] = pd.cut(dfProjAll['hb_margin_percent'], bins=bins, labels=labels, right=True)


# Calculate derived columns
dfProjAll['lb_margin_pct.P'] = (dfProjAll['unleveraged.P'] / (dfProjAll['project_costs.P'] * -1)) * 100
dfProjAll['lb_margin_pct.A'] = (dfProjAll['unleveraged.A'] / (dfProjAll['project_costs.A'] * -1)) * 100

dfProjAll['project_costs.P'] = -np.round(dfProjAll['project_costs.P'] / 1000)
dfProjAll['project_costs.A'] = -np.round(dfProjAll['project_costs.A'] / 1000)

dfProjAll['land.P'] = -np.round(dfProjAll['land.P'] / 1000)
dfProjAll['land.A'] = -np.round(dfProjAll['land.A'] / 1000)

dfProjAll['site.P'] = -np.round(dfProjAll['site.P'] / 1000)
dfProjAll['site.A'] = -np.round(dfProjAll['site.A'] / 1000)

dfProjAll['mud.P'] = -np.round(dfProjAll['mud.P'] / 1000)
dfProjAll['mud.A'] = -np.round(dfProjAll['mud.A'] / 1000)

dfProjAll['unleveraged.P'] = -np.round(dfProjAll['unleveraged.P'] / 1000)
dfProjAll['unleveraged.A'] = -np.round(dfProjAll['unleveraged.A'] / 1000)

dfProjAll['cb_Qtr1'] = -np.round(dfProjAll['cb_Qtr1'] / 1000)
dfProjAll['cb_Qtr2'] = -np.round(dfProjAll['cb_Qtr2'] / 1000)
dfProjAll['cb_Qtr3'] = -np.round(dfProjAll['cb_Qtr3'] / 1000)

# Create additional factors
dfProjAll['f_lb_margin.P'] = pd.cut(dfProjAll['lb_margin_pct.P'], bins=[0, 5, 8, 10, np.inf], 
                                     labels=["<= 5.0%", "5.1%-8.0%", "8.1%-10.0%", ">10.0%"], right=False)

dfProjAll['f_lb_margin.A'] = pd.cut(dfProjAll['lb_margin_pct.A'], bins=[0, 5, 8, 10, np.inf], 
                                     labels=["<= 5.0%", "5.1%-8.0%", "8.1%-10.0%", ">10.0%"], right=False)

dfProjAll['f_ASP'] = pd.cut(dfProjAll['wavg_ASP'], 
                             bins=[0, 300000, 450000, 600000, 750000, np.inf],                                
                             labels=["<=$300k", "$300k-$450k", "$450k-$600k", "$600k-$750k", ">$750k"],  
                             right=True)

# Calculate remaining term
dfProjAll['remain_term'] = dfProjAll['opt_term'] - dfProjAll['actualsThruPeriod']


dfProjAll['f_remain_term'] = pd.cut(dfProjAll['remain_term'], bins=[0, 12, 18, 24, 36, np.inf], 
                                     labels=["<= 12", "13-18", "19-24", "25-36", ">36"], right=False)

dfProjAll['w_recordcount'] = 1

dfProjAll['f_fundid'] = pd.Categorical(dfProjAll['fundid'], ordered=True)  # Assuming fundid is an ordinal factor

# For f_hit_70.P and f_hit_70.A
dfProjAll['f_hit_70.P'] = pd.cut(dfProjAll['hit_70.P'], bins=[0, 12, 18, 24, 30, 36, np.inf],                
                                  labels=["<=12", "13-18", "19-24", "25-30", "31-36", ">36"],  
                                  right=True)

dfProjAll['f_hit_70.A'] = pd.cut(dfProjAll['hit_70.A'], bins=[0, 12, 18, 24, 30, 36, np.inf],                
                                  labels=["<=12", "13-18", "19-24", "25-30", "31-36", ">36"],  
                                  right=True)

# For f_remain_term
dfProjAll['remain_term'] = dfProjAll['opt_term'] - dfProjAll['actualsThruPeriod']

dfProjAll['f_remain_term'] = pd.cut(dfProjAll['remain_term'], bins=[0, 12, 18, 24, 36, np.inf],                
                                     labels=["<=12", "13-18", "19-24", "25-36", ">36"],  
                                     right=True)

# For f_bookstatus
dfProjAll['f_bookstatus'] = pd.Categorical(dfProjAll['Book Status'])  # Assuming book_status is a categorical variable

# For f_msa_tier
dfProjAll['msa_tier'] = np.where(dfProjAll['msa'].str.contains(": Core"), "Core",
                                  np.where(dfProjAll['msa'].str.contains(": T1"), "T1",
                                           np.where(dfProjAll['msa'].str.contains(": T2"), "T2",
                                                    np.where(dfProjAll['msa'].str.contains(": T3"), "T3", "XX"))))

dfProjAll['f_msa_tier'] = pd.Categorical(dfProjAll['msa_tier'])

# Create a column to indicate if project is closed or active
dfProjAll['closed'] = np.where(dfProjAll['num_unpurchased.A'] == 0, "Closed", "Active")



# For dfProjectCT1 without builder_3letter grouping
dfProjectCT2_fundId = dfProjAll.groupby('fundid').agg(
    tot_lots=('numLots', 'sum'),
    project_count=('project_id', 'nunique'),
    tot_cbQ1=('cb_Qtr1', 'sum'),
    tot_cbQ2=('cb_Qtr2', 'sum'),
    tot_cbQ3=('cb_Qtr3', 'sum')
).reset_index()

# Add totals row
dfProjectCT2_fundId.loc['Total'] = dfProjectCT2_fundId.sum(numeric_only=True)

# For dfProjectCT1 with builder_3letter grouping
dfProjectCT2_builder = dfProjAll.groupby(['fundid', 'builder_3letter']).agg(
    tot_lots=('numLots', 'sum'),
    project_count=('project_id', 'nunique'),
    tot_cbQ1=('cb_Qtr1', 'sum'),
    tot_cbQ2=('cb_Qtr2', 'sum'),
    tot_cbQ3=('cb_Qtr3', 'sum')
).reset_index()

# Add totals row
dfProjectCT2_builder.loc['Total'] = dfProjectCT2_builder.sum(numeric_only=True)


# Create dfFundTotals
dfFundTotals = dfProjAll.groupby(['fundid', 'builder_3letter']).agg(
    CountOrig=('w_recordcount', 'sum'),
    CountClosed=('w_recordcount', lambda x: np.sum(x[dfProjAll['closed'] != 'Active'])),
    CountActive=('w_recordcount', lambda x: np.sum(x[dfProjAll['closed'] == 'Active'])),
    
    LotsOrig=('numLots', 'sum'),
    LotsConveyClosed=('numLots', lambda x: np.sum(x[dfProjAll['closed'] != 'Active'])),
    LotsConveyActive=('numLots', lambda x: np.sum((x - dfProjAll['num_unpurchased.A'])[dfProjAll['closed'] == 'Active'])),
    LotsActive=('num_unpurchased.A', 'sum'),

    LandOrig=('land.P', 'sum'),
    ContractSumOrig=('project_costs.P', lambda x: np.sum(x - dfProjAll['land.P'])),
    LBMargin_Orig=('unleveraged.P', 'sum'),
    LBRev_Orig=('project_costs.P', lambda x: np.sum(x + dfProjAll['unleveraged.P'])),

    CostOrig=('project_costs.P', 'sum'),
    CostClosed=('project_costs.P', lambda x: np.sum(x[dfProjAll['closed'] != 'Active'])),
    CostActive=('project_costs.P', lambda x: np.sum(x[dfProjAll['closed'] == 'Active'])),

    ConBalActive=('cb_Qtr1', lambda x: np.sum(x[dfProjAll['closed'] == 'Active']))
).reset_index()



# Subset dfFundTotals for fundid == "EHF1"
dfEHF1Totals = dfFundTotals[dfFundTotals['fundid'] == 'EHF1']
dfEHF1Totals.loc['Total'] = dfEHF1Totals.sum(numeric_only=True)
dfEHF1Totals.loc[dfEHF1Totals.index[-1], ('fundid', 'builder_3letter')] = 'Total'

dfEHF1Totals = dfEHF1Totals.rename(columns={'fundid': 'Fund',
                                          'builder_3letter': 'Builder',
                                          'CountOrig': 'Originations',
                                          'CountClosed': 'Closed',
                                          'CountActive': 'Active'
                                        })


dfEHF2Totals = dfFundTotals[dfFundTotals['fundid'] == 'EHF2']
dfEHF2Totals.loc['Total'] = dfEHF2Totals.sum(numeric_only=True)
dfEHF2Totals.loc[dfEHF2Totals.index[-1], ('fundid', 'builder_3letter')] = 'Total'
dfEHF2Totals = dfEHF2Totals.rename(columns={'fundid': 'Fund',
                                          'builder_3letter': 'Builder',
                                          'CountOrig': 'Originations',
                                          'CountClosed': 'Closed',
                                          'CountActive': 'Active'
                                        })

dfChart1Data = dfProjAll.loc[:, ['fundid', 'project_id', 'builder_3letter', 'proj_name', 'city', 'state', 'msa',
                                  'msa_tier', 'OrigMth', 'OrigQtr', 'project_costs.P', 'w_recordcount', 'numLots',
                                  'num_unpurchased.A', 'cb_Qtr1', 'cb_Qtr2', 'cb_Qtr3', 'closed']+ 
                             [col for col in dfProjAll.columns if col.startswith('f_') and col != 'f_lots_asp']]
 
                 

# Creating a Pandas Excel writer using xlsxwriter as the engine'
excel_path = "/Users/kavya/Dropbox/Board Package Material/dfProjAllTEST.xlsx"
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    dfProjAll.to_excel(writer, sheet_name='dfProjAll', index=False)
    dfChart1Data.to_excel(writer, sheet_name='dfChart1Data', index=False)
    dfEHF1Totals.to_excel(writer, sheet_name='dfEHF1Totals', index=False)
    dfEHF2Totals.to_excel(writer, sheet_name='dfEHF2Totals', index=False)

### SET PARAMS FOR GENERATING CHARTS 

params = {}
params['pFundID'] = 'EHF2'
params['pQtrEnd'] = 'December 31, 2023'

# Now will load dfChart1Data that has all the project information
#   we will subset this to the correct fund and exclude Closed Projects

#       dfChart1Data has ALL Project Data
#       dfChart1DataA has only Active Project Data

dfChart1Data = dfChart1Data[dfChart1Data['fundid'] == params['pFundID']]
dfChart1Data.reset_index(drop=True, inplace=True)

dfChart1DataA = dfChart1Data[dfChart1Data['closed'] == "Active"]
dfChart1DataA.reset_index(drop=True, inplace=True)

# The Summary page will use the EHFX dataframe ...
if params['pFundID'] == "EHF2":
    dfEHFXTotals = dfEHF2Totals
else:
    dfEHFXTotals = dfEHF1Totals


columns_to_convert = ["Originations", "Closed", "Active"]
dfEHFXTotals[columns_to_convert] = dfEHFXTotals[columns_to_convert].astype(int)
selected_columns = ["Builder", "Originations", "Closed", "Active"]
df_selected = dfEHFXTotals[selected_columns]


# Function to get project names based on builder and closed status
def get_projects(builder, status):
    
    projects = dfChart1Data[(dfChart1Data['builder_3letter'] == builder) & (dfChart1Data['closed'] == status)]['proj_name'].tolist()
    return builder,"+".join(projects)

def generate_html_content(data):
    # Define CSS styles
    css_styles = """
    /* Define table style */
    table {
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
    }

    /* Define table header style */
    th {
        background-color: #f2f2f2;
        text-align: left;
        padding: 8px;
    }

    /* Define table row style */
    td {
        border-bottom: 1px solid #ddd;
        padding: 8px;
    }

    /* Define table row hover style */
    tr:hover {
        background-color: #f2f2f2;
    }

    /* Define table footer style */
    tfoot {
        font-weight: bold;
    }
    """

    # Define a function to generate the table using Matplotlib
    def generate_table(ax, data, column_labels, row_labels, cell_colors, cell_text_colors):
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=data, colLabels=column_labels, rowLabels=row_labels, loc='center',
                         cellColours=cell_colors, cellLoc='center', cellTextColors=cell_text_colors)  # Pass cell_text_colors
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)  # Adjust the table size as needed

    # Create a figure and axes for the table
    fig, ax = plt.subplots(figsize=(10, 5))

    # Prepare data for the table
    column_labels = data[0]
    row_labels = [row[0] for row in data[1:]]
    data_values = [row[1:] for row in data[1:]]

    # Specify colors for cells
    num_colors = len(data_values)
    num_cols = len(column_labels)  # Number of columns
    cell_colors = [['paleturquoise' if (i + j) % 2 == 0 else 'lavender' for j in range(num_cols)] for i in range(num_colors)]
    cell_text_colors = [[mcolors.to_rgba('black', alpha=1.0) for _ in range(num_cols)] for _ in range(num_colors)]

    table_title = "Origination Summary (Number of Investments)"

    # Generate the table
    generate_table(ax, data_values, column_labels, row_labels, cell_colors, cell_text_colors)

    # Save the figure as an image (e.g., PNG)
    table_image_path = "/Users/kavya/Dropbox/Board Package Material/HTMLOrigSumtable_image.png"
    plt.savefig(table_image_path)

    # Define the HTML content
    html_content = (
        "<html>"
        "<head>"
        "<style>" + css_styles + "</style>"
        "<script>"
        "// JavaScript function to toggle visibility of closed projects table\n"
        "function toggleClosedProjectsTable(rowIndex) {"
        "var table = document.getElementById('closed-projects-' + rowIndex);"
        "table.style.display = (table.style.display === 'none') ? 'block' : 'none';"
        "}"
        "// JavaScript function to toggle visibility of active projects table\n"
        "function toggleActiveProjectsTable(rowIndex) {"
        "var table = document.getElementById('active-projects-' + rowIndex);"
        "table.style.display = (table.style.display === 'none') ? 'block' : 'none';"
        "}"
        "</script>"
        "</head>"
        "<body>"
        "<h2>" + table_title + "</h2>"
        "<table>"
        "<!-- Your table content here -->"
        "<tr>"
        "<th>Builder</th>"
        "<th>Originations</th>"
        "<th>Closed</th>"
        "<th>Active</th>"
        "</tr>"
        "<!-- Iterate over your data and generate table rows -->"
        + ''.join([
            "<tr><td>{}</td><td>{}</td><td onclick=\"toggleClosedProjectsTable('{}')\" style=\"cursor: pointer;\">{}</td><td onclick=\"toggleActiveProjectsTable('{}')\" style=\"cursor: pointer;\">{}</td></tr>".format(row[0], row[1], index, row[2], index, row[3]) for index, row in enumerate(data[1:])
        ]) +
        "</table>"
        "<!-- Hidden tables for closed projects -->"
        + ''.join([
            "<table id=\"closed-projects-{}\" style=\"display: none;\"><tr><th colspan=\"3\">Closed Projects for {}</th></tr>{}</table>".format(index, row[0], ''.join(['<tr><td>{}</td></tr>'.format(project) for project in get_projects(row[0], 'Closed')[1].split('+')])) for index, row in enumerate(data[1:])
        ])
        + ''.join([
            "<table id=\"active-projects-{}\" style=\"display: none;\"><tr><th colspan=\"4\">Active Projects for {}</th></tr>{}</table>".format(index, row[0], ''.join(['<tr><td>{}</td></tr>'.format(project) for project in get_projects(row[0], 'Active')[1].split('+')])) for index, row in enumerate(data[1:])
        ])
        + "</body>"
        "</html>"
    )

    return html_content

html_content = generate_html_content(df_selected.values.tolist())
# Define the filename for saving the HTML content
html_filename = "styled_table.html"

# Define the full path to the output file
output_filepath = "/Users/kavya/Dropbox/Board Package Material/" + html_filename


# Write the HTML content to a file
with open(output_filepath, "w") as html_file:
    html_file.write(html_content)

# Open the saved HTML file in the default web browser
import webbrowser
webbrowser.open(output_filepath)


html_content = generate_html_content(df_selected.values.tolist())
# Define the filename for saving the HTML content
html_filename = "styled_table.html"

# Define the full path to the output file
output_filepath = "/Users/kavya/Dropbox/Board Package Material/" + html_filename


# Write the HTML content to a file
with open(output_filepath, "w") as html_file:
    html_file.write(html_content)

# Open the saved HTML file in the default web browser
import webbrowser
webbrowser.open(output_filepath)

