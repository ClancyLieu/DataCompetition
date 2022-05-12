#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:52:41 2022
@author: lieu
"""

import sys
sys.path.append('/Users/lieu/Documents/Code/MachineLearning')
import pandas as pd
import numpy as np
import mlMethod.preProcess as prep
from mlMethod import annClass as ann
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay

'''
import data
'''
trainset = pd.read_csv('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/train.csv',
                       header=0)
testset_a = pd.read_csv('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/test_A.csv',
                       header=0)
trainset['sign'] = 'train'
testset_a['sign'] = 'testa'
dataset = pd.concat([trainset,testset_a], axis=0, ignore_index=True)

'''
check and preprocess data
'''
preset = dataset.copy()
#unify the symbols of NaN
preset.replace('?', np.nan, inplace=True)
#subtract 2 from values and fill nans with 0
def minus2nfillna(data,column):
    for cl in column:
        data[cl] = pd.to_numeric(data[cl])
        data[cl] = data[cl].map(lambda x:(x-2) if x!=np.nan else x)
        #np.nan cant be filled by lambda function
        data[cl].fillna(0,inplace=True)
    return data
def fillna(data,column,a):
    for cl in column:
        data[cl].fillna(a,inplace=True)
    return data
m2f0 = ['AGN_CNT_RCT_12_MON','ICO_CUR_MON_ACM_TRX_TM','NB_RCT_3_MON_LGN_TMS_AGV',
        'AGN_CUR_YEAR_AMT','AGN_CUR_YEAR_WAG_AMT','AGN_AGR_LATEST_AGN_AMT','ICO_CUR_MON_ACM_TRX_AMT',
        'COUNTER_CUR_YEAR_CNT_AMT','PUB_TO_PRV_TRX_AMT_CUR_YEAR','CUR_YEAR_PUB_TO_PRV_TRX_PTY_CNT',
        'MON_12_EXT_SAM_TRSF_IN_AMT','MON_12_EXT_SAM_TRSF_OUT_AMT','MON_12_EXT_SAM_NM_TRSF_OUT_CNT','MON_12_EXT_SAM_AMT',
        'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT','CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT','MON_12_TRX_AMT_MAX_AMT_PCTT',
        'MON_12_ACM_ENTR_ACT_CNT','MON_12_ACM_LVE_ACT_CNT','MON_6_50_UP_ENTR_ACT_CNT','MON_6_50_UP_LVE_ACT_CNT','CUR_YEAR_COUNTER_ENCASH_CNT',
        'MON_12_ACT_OUT_50_UP_CNT_PTY_QTY','MON_12_ACT_IN_50_UP_CNT_PTY_QTY','LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL',
        'LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV','LAST_12_MON_MON_AVG_TRX_AMT_NAV','COR_KEY_PROD_HLD_NBR','CUR_YEAR_MID_BUS_INC',
        'EMP_NBR','REG_CPT','SHH_BCK','HLD_DMS_CCY_ACT_NBR','HLD_FGN_CCY_ACT_NBR']
fn = ['MON_12_CUST_CNT_PTY_ID']
fb = ['WTHR_OPN_ONL_ICO']
fg = ['LGP_HLD_CARD_LVL','NB_CTC_HLD_IDV_AIO_CARD_SITU']
f0 = ['CUR_YEAR_MON_AGV_TRX_CNT','MON_12_AGV_TRX_CNT','MON_12_AGV_ENTR_ACT_CNT','MON_12_AGV_LVE_ACT_CNT',
      'LAST_12_MON_COR_DPS_DAY_AVG_BAL','CUR_MON_COR_DPS_MON_DAY_AVG_BAL','CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL','CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR',
      'AI_STAR_SCO','REG_DT','OPN_TM']
preset = minus2nfillna(preset,m2f0)
preset = fillna(preset,fn,'N')
preset = fillna(preset,fb,'B')#I am not sure
preset = fillna(preset,fg,'G')
preset = fillna(preset,f0,0)
#dtype = object, need to be converted into np.number
for cl in f0:
    preset[cl] = pd.to_numeric(preset[cl])
#number code to char code
preset['AI_STAR_SCO'].map({0:'f',1:'a',2:'b',3:'c',4:'d',5:'e'})

'''
feature engineering
'''
dis_column = ['MON_12_CUST_CNT_PTY_ID','WTHR_OPN_ONL_ICO','LGP_HLD_CARD_LVL','NB_CTC_HLD_IDV_AIO_CARD_SITU','AI_STAR_SCO']
for cl in dis_column:
    counts = preset.value_counts(subset=cl)
    preset[cl] = preset[cl].map(lambda x:counts[x]/counts.sum())
#standardization
'''for cl in preset.columns:
    if (cl=='CUST_UID')|(cl=='LABEL'): continue#CUST_UID is the primary key(lots of values)
    elif np.issubdtype(preset[cl],np.number):
        plt.title(cl)
        plt.hist(preset[cl],bins=np.linspace(preset[cl].min(),preset[cl].max(),num=10))
        plt.show()
    else:
        count = preset.value_counts(subset=cl)
        plt.title(cl)
        plt.bar(count.index,height=count)
        plt.show()'''
numberFeature = m2f0+f0
numberFeature.remove('AI_STAR_SCO')   
delist = ['AGN_CUR_YEAR_WAG_AMT','MON_12_EXT_SAM_TRSF_IN_AMT','MON_12_EXT_SAM_TRSF_OUT_AMT','CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT',
          'MON_12_ACM_ENTR_ACT_CNT','MON_6_50_UP_ENTR_ACT_CNT','LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV',
          'LAST_12_MON_MON_AVG_TRX_AMT_NAV','CUR_YEAR_MID_BUS_INC','EMP_NBR','MON_12_AGV_TRX_CNT','MON_12_AGV_ENTR_ACT_CNT','MON_12_ACM_LVE_ACT_CNT',
       'MON_12_AGV_LVE_ACT_CNT','LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL','CUR_MON_COR_DPS_MON_DAY_AVG_BAL']  
log_pre_list = ['AGN_CNT_RCT_12_MON','ICO_CUR_MON_ACM_TRX_TM','NB_RCT_3_MON_LGN_TMS_AGV',
                'AGN_CUR_YEAR_AMT','AGN_AGR_LATEST_AGN_AMT','ICO_CUR_MON_ACM_TRX_AMT','COUNTER_CUR_YEAR_CNT_AMT',
                'PUB_TO_PRV_TRX_AMT_CUR_YEAR','MON_12_EXT_SAM_NM_TRSF_OUT_CNT','CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT',
                'MON_12_TRX_AMT_MAX_AMT_PCTT','CUR_YEAR_MON_AGV_TRX_CNT','CUR_YEAR_PUB_TO_PRV_TRX_PTY_CNT', 
                'MON_6_50_UP_LVE_ACT_CNT','CUR_YEAR_COUNTER_ENCASH_CNT','MON_12_ACT_OUT_50_UP_CNT_PTY_QTY','MON_12_ACT_IN_50_UP_CNT_PTY_QTY',
                'LAST_12_MON_COR_DPS_DAY_AVG_BAL','CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL','COR_KEY_PROD_HLD_NBR',
                'REG_CPT','HLD_DMS_CCY_ACT_NBR','REG_DT','HLD_FGN_CCY_ACT_NBR','OPN_TM']+delist
log_pre_list.remove('CUR_YEAR_MID_BUS_INC')
for cl in log_pre_list:
    preset[cl] = np.log(preset[cl]+1)
for cl in numberFeature:
    preset[cl] = prep.dataStandardizing(preset[cl])

'''
feature control
'''
high_var = ['LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL','LAST_12_MON_COR_DPS_DAY_AVG_BAL','CUR_MON_COR_DPS_MON_DAY_AVG_BAL','CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL']
mid_var = ['MON_12_EXT_SAM_TRSF_IN_AMT','CUR_YEAR_MON_AGV_TRX_CNT','MON_12_AGV_TRX_CNT','MON_12_ACM_ENTR_ACT_CNT','MON_12_AGV_ENTR_ACT_CNT',
           'MON_6_50_UP_ENTR_ACT_CNT','MON_6_50_UP_LVE_ACT_CNT','MON_12_ACT_OUT_50_UP_CNT_PTY_QTY','LAST_12_MON_MON_AVG_TRX_AMT_NAV',
           'EMP_NBR','HLD_DMS_CCY_ACT_NBR']
low_var = ['AGN_CNT_RCT_12_MON','ICO_CUR_MON_ACM_TRX_TM','NB_RCT_3_MON_LGN_TMS_AGV','AGN_CUR_YEAR_AMT',
           'AGN_CUR_YEAR_WAG_AMT','AGN_AGR_LATEST_AGN_AMT','ICO_CUR_MON_ACM_TRX_AMT','MON_12_EXT_SAM_TRSF_OUT_AMT','MON_12_EXT_SAM_NM_TRSF_OUT_CNT',
           'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT','CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT','MON_12_ACM_LVE_ACT_CNT','MON_12_AGV_LVE_ACT_CNT',
           'CUR_YEAR_PUB_TO_PRV_TRX_PTY_CNT','MON_12_ACT_IN_50_UP_CNT_PTY_QTY','LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV','COR_KEY_PROD_HLD_NBR',
           'SHH_BCK','OPN_TM','HLD_FGN_CCY_ACT_NBR']

'''接下来这部分特征构建与选择，我知道这里写的很蠢，因为第一次参加竞赛写代码
A榜特征的构造和选取是自己一种方法一种方法试然后通过AUC变化进行筛选和总结的
虽然A榜排名比不上使用深度学习模型的大佬，但也不是无用功，总结了一套特征构造和评分的方法
多亏了A榜的特征工程，B榜第一次提交就有0.73，也是总分最终能位列腰部的重要原因
（不然就得位列脚踝了）'''

#feature constructure
#zt0,zt1,zt2,zt3,bt10 特征分布不同
#zt0 = 1/preset['CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR']
#zt1 = preset['LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL']/preset['CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR']
#zt2 = preset['CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL']/preset['CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR']#higher slightly
#zt3 = preset['LAST_12_MON_MON_AVG_TRX_AMT_NAV']/preset['CUR_YEAR_MID_BUS_INC']
#zt6,bt7,bt9,bt10,bt12 分离度不高
zt4 = preset['LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL']+preset['CUR_MON_COR_DPS_MON_DAY_AVG_BAL']#higher sightly
#zt6 = preset['CUR_YEAR_MON_AGV_TRX_CNT']+preset['MON_6_50_UP_ENTR_ACT_CNT']#higher slightly
zt7 = preset['LAST_12_MON_COR_DPS_DAY_AVG_BAL']+preset['CUR_MON_COR_DPS_MON_DAY_AVG_BAL']
zt8 = prep.tanh(preset['CUR_MON_COR_DPS_MON_DAY_AVG_BAL'])
bt1 = preset['MON_6_50_UP_ENTR_ACT_CNT']**3
bt2 = bt1 + preset['CUR_MON_COR_DPS_MON_DAY_AVG_BAL']
#bt3 = preset['CUR_MON_COR_DPS_MON_DAY_AVG_BAL']**3#信息量不足，信息比只有0.16
bt4 = preset['MON_6_50_UP_LVE_ACT_CNT']**3
#bt5 = preset['HLD_DMS_CCY_ACT_NBR']**3#信息量不足，信息比只有0.13
bt6 = preset['EMP_NBR']**3
#bt7 = preset['LAST_12_MON_MON_AVG_TRX_AMT_NAV']**3
#bt8 = preset['MON_12_AGV_ENTR_ACT_CNT']**3#信息量不足，信息比只有0.19
#bt9 = preset['CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL']**3
#bt10 = preset['LAST_12_MON_COR_DPS_DAY_AVG_BAL']**3
#bt11 = preset['LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL'].map(lambda x:x**2 if x>=0 else -((-x)**2))#信息量不足，信息比只有0.12
#bt12 = bt3+bt9+bt10+bt11
bt13 = preset['CUR_MON_COR_DPS_MON_DAY_AVG_BAL'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))#信息量贼高，信息比1.15
bt14 = preset['CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))#信息量贼高
bt15 = preset['LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt16 = preset['LAST_12_MON_COR_DPS_DAY_AVG_BAL'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))#信息量贼高
bt17 = preset['MON_6_50_UP_ENTR_ACT_CNT'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt18 = preset['MON_6_50_UP_LVE_ACT_CNT'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt19 = preset['HLD_DMS_CCY_ACT_NBR'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt20 = preset['EMP_NBR'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt21 = preset['LAST_12_MON_MON_AVG_TRX_AMT_NAV'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt22 = preset['MON_12_ACT_OUT_50_UP_CNT_PTY_QTY'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt23 = preset['MON_12_AGV_ENTR_ACT_CNT'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt24 = preset['MON_12_AGV_TRX_CNT'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt25 = preset['CUR_YEAR_MON_AGV_TRX_CNT'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt26 = preset['MON_12_ACM_ENTR_ACT_CNT'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt27 = preset['MON_12_EXT_SAM_TRSF_IN_AMT'].map(lambda x:x**(1/3) if x>=0 else -((-x)**(1/3)))
bt28 = bt13+bt14+bt15+bt16#信息量贼高，信息比1.07
bt29 = bt17+bt18+bt19+bt20+bt21
bt30 = bt22+bt23+bt24+bt25+bt26+bt27
#b31,b32,b33,b34,b35,b36,b37,bt38,b39,b40,bt42,bt45,bt46,bt47,bt48,bt50 信息量不足，分离度不够
#b33,b34,b37,bt38,b39,bt46,bt47,bt48,bt50 特征分布不同
#bt31 = bt1*bt2
#bt32 = bt1*bt3
#bt33 = bt1*bt2*bt3
#bt34 = bt1*bt4
#bt35 = bt1*bt2*bt3*bt4
#bt36 = bt2*bt3
#bt37 = bt2*bt4
#bt38 = bt2*bt3*bt4
#bt39 = bt1*bt5
#bt40 = bt1*bt6
#bt41 = bt1*bt7
#bt42 = bt1*bt8
#bt43 = bt1*bt9
#bt44 = bt1*bt10
#bt45 = bt1*bt11
#bt46 = bt1*bt2*bt4
#bt47 = bt1*bt2*bt5
#bt48 = bt1*bt2*bt6
#bt49 = bt1*bt2*bt7
#bt50 = bt1*bt2*bt8
#bt51 = bt1*bt2*bt9
#bt52 = bt1*bt2*bt10
#bt53,bt54,bt55,bt56,bt58,bt61,bt62,bt63,bt65,bt68,bt69,bt70,bt72,bt75,bt76,bt77,bt78,bt79,bt81,bt84,bt85,bt87,bt91,bt93,bt96,bt99,bt102 信息量不够
#bt53,bt54,bt55,bt56,bt58,bt62,bt63,bt65,bt68,bt69,bt70,bt72,bt75,bt76,bt77,bt79,bt85,bt87,bt91,bt93,bt96,bt99,bt102   特征分布不同
#bt53 = bt1*bt2*bt11
#bt54 = bt1*bt3*bt4
#bt55 = bt1*bt3*bt5
#bt56 = bt1*bt3*bt6
#bt57 = bt1*bt3*bt7
#bt58 = bt1*bt3*bt8
#bt59 = bt1*bt3*bt9
#bt60 = bt1*bt3*bt10
#bt61 = bt1*bt3*bt11
#bt62 = bt1*bt4*bt5
#bt63 = bt1*bt4*bt6
#bt64 = bt1*bt4*bt7
#bt65 = bt1*bt4*bt8
#bt66 = bt1*bt4*bt9
#bt67 = bt1*bt4*bt10
#bt68 = bt1*bt4*bt11
#bt69 = bt1*bt2*bt3*bt4*bt5
#bt70 = bt1*bt5*bt6
#bt71 = bt1*bt5*bt7
#bt72 = bt1*bt5*bt8
#bt73 = bt1*bt5*bt9
#bt74 = bt1*bt5*bt10
#bt75 = bt1*bt5*bt11
#bt76 = bt3*bt4*bt5
#bt77 = bt2*bt3*bt4*bt5
#bt78 = bt2*bt3*bt5
#bt79 = bt2*bt3*bt6
#bt80 = bt2*bt3*bt7
#bt81 = bt2*bt3*bt8
#bt82 = bt2*bt3*bt9
#bt83 = bt2*bt3*bt10
#bt84 = bt2*bt3*bt11
#bt85 = bt3*bt4*bt6
#bt86 = bt3*bt4*bt7
#bt87 = bt3*bt4*bt8
#bt88 = bt3*bt4*bt9
#bt89 = bt3*bt4*bt10
#bt90 = bt3*bt4*bt12
#bt91 = bt4*bt5*bt6
#bt92 = bt4*bt5*bt7
#bt93 = bt4*bt5*bt8
#bt94 = bt4*bt5*bt9
#bt95 = bt4*bt5*bt10
#bt96 = bt4*bt5*bt11
#bt97 = bt4*bt5*bt12
#bt98 = bt5*bt6*bt7
#bt99 = bt5*bt6*bt8
#bt100 = bt5*bt6*bt9
#bt101 = bt5*bt6*bt10
#bt102 = bt5*bt6*bt11
#bt103 = bt5*bt6*bt12
#bt104 = bt6*bt7*bt8
#bt105 = bt6*bt7*bt9
#bt106 = bt6*bt7*bt10
#bt107 = bt6*bt7*bt11
#bt108 = bt6*bt7*bt12
#bt109 = bt7*bt8*bt9
#bt110 = bt7*bt8*bt10
#bt111 = bt7*bt8*bt11
#bt112 = bt7*bt8*bt12
#bt113 = bt8*bt9*bt10
#bt114 = bt9*bt10*bt11
#bt115 = bt10*bt11*bt12
#bt116 = bt8*bt10*bt12
#bt120,bt119,bt145,bt142,bt143,bt144 信息量低
bt117 = bt13*bt14*bt15
bt118 = bt13*bt14*bt16#信息量高，0.7
#bt119 = bt17*bt18*bt19*bt20
#bt120 = bt21*bt22*bt23*bt24*bt25*bt26*bt27
bt121 = bt13*bt14*bt17
bt122 = bt13*bt14*bt18
bt123 = bt13*bt14*bt19
bt124 = bt13*bt14*bt20
bt125 = bt13*bt14*bt21
bt126 = bt13*bt14*bt22
bt127 = bt13*bt14*bt23
bt128 = bt13*bt14*bt24
bt129 = bt13*bt14*bt25
bt130 = bt13*bt14*bt26
bt131 = bt13*bt14*bt27
bt132 = bt13*bt14*bt28#信息量高，0.8
bt133 = bt13*bt14*bt29
bt134 = bt13*bt14*bt30
bt135 = bt14*bt15*bt16
bt136 = bt14*bt15*bt17
bt137 = bt14*bt15*bt18
bt138 = bt14*bt15*bt19
bt139 = bt14*bt15*bt20
bt140 = bt14*bt15*bt21#信息量高，0.62
bt141 = bt14*bt15*bt22
#bt142 = bt14*bt15*bt23
#bt143 = bt14*bt15*bt24
#bt144 = bt14*bt15*bt25
#bt145 = bt14*bt15*bt26
bt146 = bt14*bt15*bt27
bt147 = bt14*bt15*bt28#信息量高，0.64
bt148 = bt14*bt15*bt29
bt149 = bt14*bt15*bt30
bt150 = bt15*bt16*bt17
bt151 = bt15*bt16*bt18
bt152 = bt15*bt16*bt19
bt153 = bt15*bt16*bt20
bt154 = bt15*bt16*bt21#信息量高，0.61
bt155 = bt15*bt16*bt22
#bt159,bt157,bt158 信息量不足
bt156 = bt15*bt16*bt23
#bt157 = bt15*bt16*bt24
#bt158 = bt15*bt16*bt25
#bt159 = bt15*bt16*bt26
bt160 = bt15*bt16*bt27
bt161 = bt15*bt16*bt28#信息量高，0.66
bt162 = bt15*bt16*bt29
bt163 = bt15*bt16*bt30
bt164 = bt16*bt17*bt18#信息量高，0.7
bt165 = bt16*bt17*bt19
bt166 = bt16*bt17*bt20
bt167 = bt16*bt17*bt21#信息量高，0.62
bt168 = bt16*bt17*bt22
bt169 = bt16*bt17*bt23
bt170 = bt16*bt17*bt24
bt171 = bt16*bt17*bt25
bt172 = bt16*bt17*bt26
bt173 = bt16*bt17*bt27
bt174 = bt16*bt17*bt28
bt175 = bt16*bt17*bt30
bt176 = bt17*bt18*bt19
bt177 = bt17*bt18*bt20
bt178 = bt17*bt18*bt21
bt179 = bt17*bt18*bt22
bt180 = bt17*bt18*bt23
bt181 = bt17*bt18*bt24
bt182 = bt17*bt18*bt25
bt183 = bt17*bt18*bt26
bt184 = bt17*bt18*bt27
bt185 = bt17*bt18*bt28#信息量高，0.77
bt186 = bt17*bt18*bt29
bt187 = bt17*bt18*bt30
bt188 = bt18*bt19*bt20
bt189 = bt18*bt19*bt21
bt190 = bt18*bt19*bt22
#bt193,bt192 信息量不足
bt191 = bt18*bt19*bt23
#bt192 = bt18*bt19*bt24
#bt193 = bt18*bt19*bt25
bt194 = bt18*bt19*bt26
bt195 = bt18*bt19*bt27
bt196 = bt18*bt19*bt28
bt197 = bt18*bt19*bt29
bt198 = bt18*bt19*bt30
bt199 = bt19*bt20*bt21
#bt200 = bt19*bt20*bt22
#bt201 = bt19*bt20*bt23
#bt202 = bt19*bt20*bt24
#bt203 = bt19*bt20*bt25
#bt204 = bt19*bt20*bt26
bt205 = bt19*bt20*bt27
bt206 = bt19*bt20*bt28
bt207 = bt19*bt20*bt29
bt208 = bt19*bt20*bt30
bt209 = bt20*bt21*bt22
bt210 = bt20*bt21*bt23
bt211 = bt20*bt21*bt24
bt212 = bt20*bt21*bt25
bt213 = bt20*bt21*bt26
bt214 = bt20*bt21*bt27
bt215 = bt20*bt21*bt28
bt216 = bt20*bt21*bt29
bt217 = bt20*bt21*bt30
bt218 = bt21*bt22*bt23
bt219 = bt21*bt22*bt24
bt220 = bt21*bt22*bt25
bt221 = bt21*bt22*bt26
bt222 = bt21*bt22*bt27
bt223 = bt21*bt22*bt28#信息量较高。0.62
bt224 = bt21*bt22*bt29
bt225 = bt21*bt22*bt30
bt226 = bt22*bt23*bt24
bt227 = bt22*bt23*bt25
bt228 = bt22*bt23*bt26
#bt229 = bt22*bt23*bt27
bt230 = bt22*bt23*bt28
bt231 = bt22*bt23*bt29
bt232 = bt22*bt23*bt30
bt233 = bt23*bt24*bt25
bt234 = bt23*bt24*bt26
bt235 = bt23*bt24*bt27
bt236 = bt23*bt24*bt28#信息量较高。0.78
bt237 = bt23*bt24*bt29
bt238 = bt23*bt24*bt30
bt239 = bt24*bt25*bt26
bt240 = bt24*bt25*bt27
bt241 = bt24*bt25*bt28#信息量较高。0.82
bt242 = bt24*bt25*bt29
bt243 = bt24*bt25*bt30
bt244 = bt25*bt26*bt27
bt245 = bt25*bt26*bt28#信息量较高。0.73
bt246 = bt25*bt26*bt29
bt247 = bt25*bt26*bt30
bt248 = bt26*bt27*bt28
bt249 = bt26*bt27*bt29
bt250 = bt26*bt27*bt30
bt251 = bt27*bt28*bt29
bt252 = bt27*bt28*bt30
bt253 = bt28*bt29*bt30
control_set = preset[high_var+['LABEL','CUST_UID','sign']]#only high mean-difference features
control_set2 = pd.concat([zt4,zt7,zt8,bt1,bt2,bt4,bt6,bt13,bt14,bt15,bt16,bt17,bt18,bt19,bt20,bt21,bt22,bt23,bt24,bt25,bt26,bt27,bt28,bt29,bt30,bt117,bt118,bt121,bt122,bt123,bt124,bt125,bt126,bt127,bt128,bt129,bt130,bt131,bt132,bt133,bt134,bt135,bt136,bt137,bt138,bt139,bt140,
bt141,bt146,bt147,bt148,bt149,bt150,bt151,bt152,bt153,bt154,bt155,bt156,bt160,bt161,bt162,bt163,bt164,bt165,bt166,bt167,bt168,bt169,bt170,
bt171,bt172,bt173,bt174,bt175,bt176,bt177,bt178,bt179,bt180,bt181,bt182,bt183,bt184,bt185,bt186,bt187,bt188,bt189,bt190,
bt191,bt194,bt195,bt196,bt197,bt198,bt199,bt205,bt206,bt207,bt208,bt209,bt210,
bt211,bt212,bt213,bt214,bt215,bt216,bt217,bt218,bt219,bt220,bt221,bt222,bt223,bt224,bt225,bt226,bt227,bt228,bt230,
bt231,bt232,bt233,bt234,bt235,bt236,bt237,bt238,bt239,bt240,bt241,bt242,bt243,bt244,bt245,bt246,bt247,bt248,bt249,bt250,bt251,bt252,bt253],axis=1,ignore_index=True)
control_set = pd.concat([control_set,control_set2],axis=1)
control_set.info()
'''这块是特征构建与新特征分离度分析的代码'''
for cl in control_set.columns:
    if (cl!='CUST_UID')&(cl!='LABEL')&(cl!='sign'):
        amean = control_set[cl].mean()
        astd = control_set[cl].mean()
        a = prep.cofVariation(control_set[cl])
        bmean = control_set[cl][control_set['LABEL']==0].mean()
        bstd = control_set[cl][control_set['LABEL']==0].std()
        b = prep.cofVariation(control_set[cl][control_set['LABEL']==0])
        cmean = control_set[cl][control_set['LABEL']==1].mean()
        cstd = control_set[cl][control_set['LABEL']==1].std()
        c = prep.cofVariation(control_set[cl][control_set['LABEL']==1])
        dmean = control_set[cl][control_set['sign']=='testa'].mean()
        emean = control_set[cl][control_set['sign']=='train'].mean()
        dstd = control_set[cl][control_set['sign']=='testa'].std()
        estd = control_set[cl][control_set['sign']=='train'].std()
        d = prep.cofVariation(control_set[cl][control_set['sign']=='testa'])
        e = prep.cofVariation(control_set[cl][control_set['sign']=='train'])
        #print('%s列总均值%.3f,标准差%.4f,总变异系数%.4f,' %(cl,amean,astd,a))#mean=0 we have standradize the feature
        print('%s列label=0均值%.3f,标准差%.4f,总变异系数%.4f,' %(cl,bmean,bstd,b))
        print('%s列label=1均值%.3f,标准差%.4f,总变异系数%.4f,' %(cl,cmean,cstd,c))
        print('%s列label=test均值%.3f,标准差%.4f,总变异系数%.4f,' %(cl,dmean,dstd,d))
        print('%s列label=train均值%.3f,标准差%.4f,总变异系数%.4f,' %(cl,emean,estd,e))
        #print(' mean difference %.4f'%(bmean-cmean))
        print(abs(bmean-cmean)/estd)

'''
sample spilit
'''
finaldata = control_set.copy()
traindata = finaldata[finaldata['sign']=='train']
testadata = finaldata[finaldata['sign']=='testa']
#这句validation的抽样，不应该重置索引，写错了。
validation = traindata.sample(frac = 0.2, random_state=10).reset_index(drop=True,inplace=False)
train = traindata.drop(validation.index,axis=0,inplace=False).reset_index(drop=True,inplace=False)
#trainset and validation set
y_train = train['LABEL']
x_train = train.drop(['LABEL','CUST_UID','sign'],axis=1,inplace=False)#UID cant be one of features
y_valid = validation['LABEL']
x_valid = validation.drop(['LABEL','CUST_UID','sign'],axis=1,inplace=False)#UID cant be one of features
#test set
x_test = testadata.drop(['LABEL','CUST_UID','sign'],axis=1,inplace=False)

'''construt SVM'''
from sklearn import svm
estimator = svm.SVC(C=100,kernel='rbf',gamma=2**(-15),probability=True)
estimator.fit(x_train,y_train)
y_threshold = estimator.predict_proba(x_valid)[:,1]
print(roc_auc_score(y_valid,y_threshold))
RocCurveDisplay.from_predictions(y_valid,y_threshold)
plt.show()
#if AUC meets our demand, we should train the model again
traindata_y = traindata['LABEL']
traindata_x = traindata.drop(['LABEL','CUST_UID','sign'],axis=1,inplace=False)
estimator.fit(traindata_x,traindata_y)
estimation = estimator.predict_proba(x_test)[:,1]
final = pd.DataFrame({'CUST_UID':testadata['CUST_UID'],'Probablity':estimation})
final.to_csv('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/prediction_2.txt',
             index=False,float_format='%.10f',encoding='utf-8',sep=' ')

'''
construct Artificial Neuron Network
A榜没能想明白为什么人工神经网络性能这么低
因此虽然写了ANN的代码，但性能达不到竞赛要求
'''
'''
#label onehot
label_enc = OneHotEncoder()
label_enc.fit(train[['LABEL']])
label_train = pd.DataFrame(label_enc.transform(train[['LABEL']]).toarray(),columns = ['y_0','y_1'])
label_valid = pd.DataFrame(label_enc.transform(validation[['LABEL']]).toarray(),columns = ['y_0','y_1'])
#ann
layers = [10,10,2]
alpha = [0.05,0.0001]
epoch = 20
num_batch = 5000
estimator2 = ann.annClass(x_train,label_train,layers)
estimator2.weights[1][:-1,:].shape
for a in alpha:   
    optimal_weights = estimator2.train(a,epoch,num_batch)
probability = estimator2.predict(x_valid)[:,1]
print(roc_auc_score(y_valid,probability))
RocCurveDisplay.from_predictions(y_valid,probability)
plt.show()
#prediciton
estimation = estimator2.predict(x_test)[:,1]
final = pd.DataFrame({'CUST_UID':testadata['CUST_UID'],'Probablity':estimation})
final.to_csv('/Users/lieu/Documents/Code/MachineLearning/Kaggle/cmbFintech/2022cmbData/prediction_2.txt',
             index=False,float_format='%.10f',encoding='utf-8',sep=' ')
'''
