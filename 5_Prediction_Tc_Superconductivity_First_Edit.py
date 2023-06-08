# بسم الله الرحمن الرحیم



import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
from collections import Counter
import chemparse
from catboost import CatBoostRegressor
import catboost
import mendeleev
from mendeleev import element
from mendeleev import *
from mendeleev.fetch import fetch_ionization_energies
from matminer.featurizers.composition.element import ElementFraction
import pymatgen
from pymatgen.core.composition import Composition
import joblib

def load_data():

    global Supercon_Main
    global List_Column_Names_DataG
    global x_DataG_13022_30Features_103columns
    global y_DataG_13022

    Supercon_Main=pd.read_csv('/home/gashmard/A_Machine Learning/A Main Programming/Machine learning/Based_Project/Project_9_GashmardPackage_and_Package_for_Prediction_Tc/Packge_for_Prediction_Tc_Superconducting_Materials/A_Main_Final_Code_for_DataG_DataH_Prediction_Tc/Prediction_Tc_Based_DataG_13022sample/Required_datasets/SuperCon_Main_Dataset_20562sample.csv')
    # Supercon_Main

    DataG_13022=pd.read_csv('/home/gashmard/A_Machine Learning/A Main Programming/Machine learning/Based_Project/Project_9_GashmardPackage_and_Package_for_Prediction_Tc/Packge_for_Prediction_Tc_Superconducting_Materials/A_Main_Final_Code_for_DataG_DataH_Prediction_Tc/Prediction_Tc_Based_DataG_13022sample/Required_datasets/DataG_13022sample_30features_103fraction.csv')
    # DataG_13022
    DataG_13022.drop(['Unnamed: 0'], axis=1, inplace=True)

    x_DataG_13022_30Features_103columns=DataG_13022[DataG_13022.columns[6:140]]
    # x_DataG_13022_30Features_103columns

    List_Column_Names_DataG=list(x_DataG_13022_30Features_103columns)
    # List_Column_Names_DataG

    y_DataG_13022=DataG_13022.Tc_mean
    # y_DataG_13022


def separation_compounds(New_Sample):

    global Tc_pred
    global sample_for_prediction_Tc

    list_element_Main_dataset_Normalaize=list(Supercon_Main.Element_Normalize)
    tekrar_elements= Counter(list_element_Main_dataset_Normalaize)
    Main_List_Compound=[]
    for index, row in New_Sample.iterrows():
        Element= row["element"]
        # print(Element)
        comp=Composition(Element)
        g=comp.reduced_formula
        Element_Normalize=comp.get_integer_formula_and_factor()
        Main_List_Compound.append(Element_Normalize[0])
    New_Sample['Element_Normalize']=Main_List_Compound

    for i in range(len(New_Sample)):
        new_compound=(New_Sample['element'][i])
        new_compound_Normalize=New_Sample['Element_Normalize'][i]
        # print(new_compound)
        index_2=New_Sample.index[i]
        list_Tc=[]
        Main_index=[] 

        if new_compound_Normalize not in list_element_Main_dataset_Normalaize:
            data=[new_compound]
            sample_for_prediction_Tc=pd.DataFrame()
            sample_for_prediction_Tc=pd.DataFrame(data,columns=['element'])
            st.write(new_compound,'is not exist in the main dataset.')
            # print(new_compound,'is not exist in the main dataset.')
            Tc_pred= Prediction_Tc(load_CatBoost_Model_DataG, sample_for_prediction_Tc)
            st.write(f"Prediction of the transition temperature for {new_compound} Using the model based on machine learning algorithm is= {Tc_pred} K.")
            # st.write('Prediction of the transition temperature for',new_compound,'Using the model based on machine learning algorithm is=',end=" "),Prediction_Tc(sample_for_prediction_Tc), st.write("K")
            # print('Prediction of the transition temperature for',new_compound,'Using the model based on machine learning algorithm is=',end=" "),Prediction_Tc(sample_for_prediction_Tc), print("K"),print("\n")
            st.success(f"Tc = {Tc_pred} K")
            st.success(f"Prediction Tc = {Tc_pred} K")
            st.success(f"Tc for {new_compound}  = {Tc_pred} K")
        elif new_compound_Normalize in list_element_Main_dataset_Normalaize:
            for j in range(len(Supercon_Main)):
                element=Supercon_Main['element'][j]
                Tc_1=Supercon_Main['Tc'][j]
                element_Normalize=Supercon_Main['Element_Normalize'][j]    

                if  element_Normalize == new_compound_Normalize:

                    if tekrar_elements[new_compound_Normalize] == 1:
                        st.write(f"{new_compound} exists in the main dataset and it is repeated for {tekrar_elements[new_compound_Normalize]} times.")
                        # print(new_compound,'exists in the main dataset and it is repeated for',tekrar_elements[new_compound_Normalize], 'times')
                        st.write(f"The transition temperature reported in the main dataset for{new_compound} is = {Tc_1} K.")
                        # print('The transition temperature reported in the main dataset for',new_compound,'is =',Tc_1,'K'), print("\n")
                    
                    if tekrar_elements[new_compound_Normalize] > 1:
                        index_1=Supercon_Main.index[j]
                        Main_index.append(index_1)
                        if len(Main_index) == 1:
                            st.write(f"{new_compound} exists in the main dataset and it is repeated for {tekrar_elements[new_compound_Normalize]} times.")
                            # print(new_compound,'exists in the main dataset and it is repeated for',tekrar_elements[new_compound_Normalize], 'times')
                            st.write(f"The transition temperatures reported for {new_compound} in the dataset are:")
                            # print("The transition temperatures reported for",new_compound,"in the dataset are:"), print("\n")
                        
                        list_Tc.append(float(Tc_1))
                    if tekrar_elements[new_compound_Normalize] == len(list_Tc):
                        for k in range(tekrar_elements[new_compound_Normalize]):
                            # print(list_Tc[k],'K',",",end=" "), print("\n")
                            st.write(f" {list_Tc[k]} K , ")
                        # st.line_chart(list_Tc)
                        # chart_data =pd.DataFrame(list_Tc,columns=['a'])
                        # st.line_chart(chart_data)
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        ax.hist(list_Tc, bins=20)
                        ax.set_xlabel('Tc')
                        ax.set_ylabel('Counts')
                        ax.set_title('Scatter diagram of transition temperature')
                        st.pyplot(fig)


def train_model(x_DataG_13022_30Features_103columns, y_DataG_13022):

    global CatBoost_Model_DataG

    CatBoost_Model_DataG = CatBoostRegressor(iterations=30, learning_rate=0.07, depth=3, logging_level='Silent', random_state=42, 
                        early_stopping_rounds=200 ,l2_leaf_reg=3,thread_count=4,snapshot_interval=600, one_hot_max_size=30
                        , eval_metric='AUC', bagging_temperature=3, subsample=0.35,random_strength=0.35,od_pval=0.001,grow_policy='SymmetricTree',
                        min_child_samples=1, sampling_frequency='PerTree' )
    CatBoost_Model_DataG= CatBoost_Model_DataG.fit(x_DataG_13022_30Features_103columns, y_DataG_13022)
    return  CatBoost_Model_DataG

# Save the trained model
def save_model(CatBoost_Model_DataG):
    joblib.dump(CatBoost_Model_DataG, 'CatBoost_Model_DataG.joblib')  # Save the model to 'model.joblib'

def load_train_model():

    global load_CatBoost_Model_DataG 

    load_CatBoost_Model_DataG = joblib.load('CatBoost_Model_DataG.joblib')  # Load the model from 'model.joblib'
    return load_CatBoost_Model_DataG

# CatBoost_Model_DataG.save_model('./CatBoost_Model_DataG_1')

# # Save the model
# joblib.dump(CatBoost_Model_DataG, 'CatBoost_Model_DataG.joblib')  # Save the model to 'model.joblib'
# CatBoost_Model_DataG = joblib.load("/home/gashmard/A_Python/Streamlit Package/Codes/CatBoost_Model_DataG.joblib")
# file_path = st.text_input("Enter the file path of the trained model:", "")
# CatBoost_Model_DataG = joblib.load(file_path)

# برای مثال
# data=[['0','Ba0.6Sm0.4As2Fe2'],['1','Fe2As2Sr4V2O5'],['2','Na1Cl1'],['3','Mg1B2'],['4','Mg0.5B3']]
# New_Sample=pd.DataFrame(data,columns=['index','element'])
# # New_Sample


# Main
# این سلول بسیار مهم از دوتا تابع مهم تشکیل شده است. تابع اول که اسم آن سپریشن کمپوندس هست میاد و کار جداسازی رو انجام میده در واقع در 
# این تابع اگر ترکیبی در دیتاست اصلی سوپرکان بود دمای گذارش ارایه میشود اما اگر ترکیبی در دیتاست اصلی سوپرکان نبود آن ترکیب به تابع دوم
# که در پایین است پاس داده میشود تا دمای گذار آن ترکیب را پیش بینی کند
# بنابراین تابع دوم با نام پریدیکشن تی سی در پایین تر همین سلول قرار دارد که برای پیش بینی تی سی هست 
#  نکته بسیار مهم اینکه ترکیب یا ترکیب هایی که برای پیش بینی دمای گذار آنها قرار است از این توابع استفاده کنیم حتما باید در قالب دیتافریم و با
#  اسم نیوسمپل به این سلول فرستاده شوند 
#  برای نمایش و چاپ دمای گذار ترکیب هایی که در دیتاست نیوسمپل قرار دارند آخرین خط همین سلول را اجرا کن


                   
def Prediction_Tc(load_CatBoost_Model_DataG, sample_for_prediction_Tc):
    from mendeleev import element

    global Tc_pred

# Number_of_Elements
    Number_of_Elements=[]
    for row in sample_for_prediction_Tc.iterrows():
        comp = Composition(row[1]['element'])
        Number=len(comp) 
        Number_of_Elements.append(Number)   
    sample_for_prediction_Tc['Number_Elements']= Number_of_Elements 

    #  Sum_of_Subscript
    Sum_Subscript_List=[]
    for row in sample_for_prediction_Tc.iterrows():   
        comp = Composition(row[1]['element'])
        Sum_Subscript_atoms =comp.num_atoms
        Sum_Subscript_List.append(Sum_Subscript_atoms)
    sample_for_prediction_Tc['Sum_Subscript']= Sum_Subscript_List

    # thermal_conductivity= Thermal_conduct
    Create_Features_dict_Thermal_Comd = {'element':['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
    'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
    'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
    'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr',
    'Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
    'Cn','Nh','Fl','Mc','Lv','Ts','Og'],
    'Thermal_Conductivity_W_mk':['0.1805','0.1513','85','190','27','140','0.02583','0.02658','0.0277',
    '0.0491','140','160','235','150','0.236','0.205','0.0089','0.01772','100','200','16','22','31','94','7.8',
    '80','100','91','400','120','29','60','50','0.52','0.12','0.00943','58','35','17','23',
    '54','139','51','120','150','72','430','97','82','67','24','3','0.449','0.00565',
    '36','18','13','11','13','17','15','13','14','11','11','11','16','15','17','39',
    '16','23','57','170','48','88','150','72','320','8.3','46','35','8','20','2'
    ,'0.00361','15','19','12','54','47','27','6','6','10','10','10','10','10',
    '10','10','10','10','23','58','19','0.01','0.01','0.01','0.01','0.01','0.01','0.01','0.01','0.01','0.01','0.01','0.0023']}                                 
    Thermal_Conductivity = pd.DataFrame(Create_Features_dict_Thermal_Comd)
    total_Thermal_conduct_Based_Elemental_range=[]
    total_Thermal_conduct_Based_fraction_median=[]

    #ElecAffinity
    Create_Features_dict_wiki = {'element':['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
    'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
    'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
    'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr',
    'Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
    'Cn','Nh','Fl','Mc','Lv','Ts','Og'],
    'Electron_affinity_Wiki_Kj_mol':['72.76','-48','59.63','-48','26.98','121.77','-6.8','140.97','328.16',
    '-116','52.86','-40','41.76','134.06','72.03','200.41','348.57','-96','48.38','2.37','18','7.28','50.91','65.21','-50',
    '14.78','63.89','111.65','119.23','-58','29.06','118.93','77.65','194.95','324.53','-96','46.88','5.02','29.6','41.80',
    '88.51','72.1','53','100.96','110.27','54.24','125.86','-68','37.04','107.29','101.05','190.16','295.15','-77',
    '45.5','13.95','53.79','55','10.53','9.4','12.45','15.63','11.2','13.22','12.67','33.96','32.61','30.1','99','-1.93',
    '23.04','17.18','31','78.76','5.82','103.99','150.90','205.04','222.74','-48','30.88','34.41','90.92','136','233.08'
    ,'-68','46.89','9.64','33.77','58.63','53.03','30.39','45.85','-48.33','9.93','27.17','-165.24','-97.31','-28.60',
    '33.96','93.91','-223.22','-30.04','0','0','0','0','0','0','0','151','0','66.6','0','35.3','74.9','165.9','5.4']}                             
    ElecAffinity_Wiki = pd.DataFrame(Create_Features_dict_wiki) 
    total_ElecAffinity_Based_fraction_max=[]
    total_ElecAffinity_Based_fraction_min=[]
    total_ElecAffinity_Based_fraction_median=[]
    total_ElecAffinity_Based_Elemental_min=[]
    total_ElecAffinity_Based_Elemental_range=[] 

    # Ionic_Radius
    Ionic_Radius_Element_periodic_tabel = {'element':['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
    'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
    'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
    'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr',
    'Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
    'Cn','Nh','Fl','Mc','Lv','Ts','Og'],
    'Ionic_Radius_Based_Angestrom':['1.48','0.4','0.76','0.27','0.11','0.15','1.46','1.38','1.3',
    '0.4','1.02','0.72','0.39','0.26','0.17','1.84','1.81','0.4','1.38','1.12','0.745','0.67','0.46','0.615','0.83',
    '0.78','0.745','0.69','0.65','0.6','0.47','0.39','0.335','1.98','1.96','0.4','1.56','1.21','0.9','0.72',
    '0.72','0.65','0.645','0.68','0.665','0.86','0.94','0.87','0.8','0.69','0.76','2.21','2.2','0.4',
    '1.74','1.35','1.1','1.01','0.99','1.109','0.97','1.079','1.01','1','0.923','0.912','0.901','0.89','0.88','0.868'
    ,'0.861','0.83','0.69','0.62','0.63','0.49','0.625','0.625','0.85','0.69','1.5','1.19','1.03','0.94',
    '0.62','0.4','1.8','1.7','1.12','1.09','1.01','1','0.98','0.96','1.09','0.95','0.93','0.92','0.4','0.4','0.4','1.1','0.4',
    '0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4','0.4']}                         
    df_Ionic_Radius_Element_periodic_tabel = pd.DataFrame(Ionic_Radius_Element_periodic_tabel)
    total_Ionic_Radius_Based_Elemental_range=[] 
    total_Ionic_Radius_Based_Elemental_min=[]
    total_Ionic_Radius_Based_Elemental_max=[]

    # Electrical_conductivity= Electric_Conduct 
    Create_Features_dict_Electric_Cond = {'element':['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
    'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
    'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
    'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr',
    'Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
    'Cn','Nh','Fl','Mc','Lv','Ts','Og'],
    'Elec_Conductivity_MS_m':['0.00012','0.00012','11','25','0.000000001','0.1','0.00012','0.00012','0.00012',
    '0.0001','21','23','38','0.001','10','0.000000000000000000001','0.00000001','0.00012','14','29','1.8','2.5','5','7.9','0.62',
    '10','17','14','59','17','7.1','0.002','3.3','0.00012','0.00000000000000001','0.00012','8.3','7.7','1.8','2.4',
    '6.7','20','5','14','23','10','62','14','12','9.1','2.5','0.01','0.0000000000001','0.00012',
    '5','2.9','1.6','1.4','1.4','1.6','1.3','1.1','1.1','0.77','0.83','1.1','1.1','1.2','1.4','3.6',
    '1.8','3.3','7.7','20','5.6','12','21','9.4','45','1.0','6.7','4.8','0.77','2.3','0.00012'
    ,'0.00012','0.00012','1.0','0.00012','6.7','5.6','3.6','0.83','0.67','0.00012','0.00012','0.00012','0.00012','0.00012',
    '0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012',
    '0.00012','0.00012','0.00012','0.00012','0.00012','0.00012','0.00012']}                              
    Electrical_Conductivity = pd.DataFrame(Create_Features_dict_Electric_Cond) 
    total_Electric_Conduct_Based_Elemental_min=[]

    # pettifor_number=Pettifor_numb 
    total_pettifor_number_Based_Elemental_range=[] 
    total_pettifor_number_Based_fraction_min=[]
    total_pettifor_number_Based_fraction_mean=[]

    # heat_of_formation=heat_formation
    total_heat_formation_Based_fraction_mean=[]
    total_heat_formation_Based_fraction_max=[]
    total_heat_formation_Based_Elemental_max=[]
    total_heat_formation_Based_Elemental_range=[]

    # First Ionisation Energy= First_Ionis_Energy
    total_First_Ionis_Energy_Based_Elemental_mean=[]

    # dipole_polarizability
    total_dipole_polarizability_Based_fraction_range=[]
    total_dipole_polarizability_Based_Elemental_median=[]
    total_dipole_polarizability_Based_Elemental_range=[]

    # number electron valence= Number_Elec_Valence
    total_Number_Elec_Valence_Based_Elemental_median=[]

    # period number
    total_period_number_Based_Elemental_max=[]
    total_period_number_Based_Elemental_range=[]

    # Electrongativity = ElecGativ
    total_electronegativ_Based_Elemental_median=[]
    total_electronegativ_Based_fraction_max=[]

    # unpaired_electrons = UnpairedElec
    total_unpaired_electron_Based_Elemental_max=[]

    for index, row in sample_for_prediction_Tc.iterrows():
        Element_dict=chemparse.parse_formula(row["element"]) 
        Elem = row["element"]
        comp = Composition(Elem)

        Thermal_conduct_List=[]  
        Thermal_conduct_List_Based_on_fraction=[]
        pettifor_number_List=[]  
        pettifor_number_List_Based_on_fraction=[]
        ElecAffinity_List=[]
        ElecAffinity_List_Based_on_fraction=[]
        heat_formation_List=[]
        heat_formation_List_Based_on_fraction=[]
        First_Ionis_Energy_List=[]
        dipole_polarizability_List=[]
        dipole_polarizability_List_Based_on_fraction=[]
        Ionic_Radius_List=[]
        Number_Elec_Valence_List=[] 
        period_number_List=[]
        Electric_Conduct_List=[] 
        electronegativ_List=[]
        electronegativ_List_Based_on_fraction=[] 
        unpaired_electron_List=[]  

        for k, v in Element_dict.items(): # k=key and v= value
            elem_str=element(str(k))
            fraction_Elements = comp.get_atomic_fraction(k)

            Thermal_Conductiv=Thermal_Conductivity[Thermal_Conductivity['element']==str(k)]['Thermal_Conductivity_W_mk'].values[0]
            Thermal_Conductiv=float(Thermal_Conductiv)
            Thermal_conduct_List.append(Thermal_Conductiv)       
            Thermal_conduct_Based_on_fraction=Thermal_Conductiv*fraction_Elements 
            Thermal_conduct_List_Based_on_fraction.append(Thermal_conduct_Based_on_fraction)

            elem_pettifor_number=elem_str.pettifor_number
            if elem_pettifor_number==None:
                elem_pettifor_number=120
            pettifor_number_List.append(elem_pettifor_number)       
            pettifor_number_Based_on_fraction=elem_pettifor_number*fraction_Elements 
            pettifor_number_List_Based_on_fraction.append(pettifor_number_Based_on_fraction)

            ElecAffinity=ElecAffinity_Wiki[ElecAffinity_Wiki['element']==str(k)]['Electron_affinity_Wiki_Kj_mol'].values[0]
            ElecAffinity=float(ElecAffinity)
            ElecAffinity_List.append(ElecAffinity)
            ElecAffinity_Based_on_fraction=ElecAffinity*fraction_Elements 
            ElecAffinity_List_Based_on_fraction.append(ElecAffinity_Based_on_fraction)

            elem_heat_formation=elem_str.heat_of_formation 
            if elem_heat_formation==None:
                elem_heat_formation=170
            heat_formation_List.append(elem_heat_formation)       
            heat_formation_Based_on_fraction=elem_heat_formation*fraction_Elements 
            heat_formation_List_Based_on_fraction.append(heat_formation_Based_on_fraction)

            Atomic_number_element=elem_str.atomic_number
            elem_First_Ionis_Energy =fetch_ionization_energies(degree=1)['IE1'][Atomic_number_element]
            if elem_First_Ionis_Energy==None:
                elem_First_Ionis_Energy=1
            First_Ionis_Energy_List.append(elem_First_Ionis_Energy)

            elem_dipole_polarizability=elem_str.dipole_polarizability
            if elem_dipole_polarizability==None:
                elem_dipole_polarizability=75
            dipole_polarizability_List.append(elem_dipole_polarizability)       
            dipole_polarizability_Based_on_fraction=elem_dipole_polarizability*fraction_Elements 
            dipole_polarizability_List_Based_on_fraction.append(dipole_polarizability_Based_on_fraction)

            Ionic_Radius=df_Ionic_Radius_Element_periodic_tabel[df_Ionic_Radius_Element_periodic_tabel['element']==str(k)]['Ionic_Radius_Based_Angestrom'].values[0]
            Ionic_Radius=float(Ionic_Radius)
            if Ionic_Radius==None:
                Ionic_Radius=1
            Ionic_Radius_List.append(Ionic_Radius) 

            elem_Number_Elec_Valence=elem_str.nvalence()
            if elem_Number_Elec_Valence==None:
                elem_Number_Elec_Valence=2
            Number_Elec_Valence_List.append(elem_Number_Elec_Valence) 

            elem_period_number=elem_str.period
            if elem_period_number==None:
                elem_period_number=3
            period_number_List.append(elem_period_number) 

            Electrical_Conductiv=Electrical_Conductivity[Electrical_Conductivity['element']==str(k)]['Elec_Conductivity_MS_m'].values[0]
            Electrical_Conductiv=float(Electrical_Conductiv)
            if Electrical_Conductiv==None:
                Electrical_Conductiv=1
            Electric_Conduct_List.append(Electrical_Conductiv) 

            elem_electronegativ=elem_str.electronegativity('pauling')
            if elem_electronegativ==None:
                elem_electronegativ=0.1
            if k == 'Kr':
                elem_electronegativ=3
            electronegativ_List.append(elem_electronegativ)       
            fraction_Elements = comp.get_atomic_fraction(k)
            electronegativ_Based_on_fraction=elem_electronegativ*fraction_Elements 
            electronegativ_List_Based_on_fraction.append(electronegativ_Based_on_fraction)

            elem_unpaired_electron=elem_str.ec.unpaired_electrons() 
            if elem_unpaired_electron==None:
                elem_unpaired_electron=1
            unpaired_electron_List.append(elem_unpaired_electron) 

        total_Thermal_conduct_Based_Elemental_range.append(np.ptp(Thermal_conduct_List))
        total_Thermal_conduct_Based_fraction_median.append(np.median(Thermal_conduct_List_Based_on_fraction))
        total_pettifor_number_Based_Elemental_range.append(np.ptp(pettifor_number_List))
        total_pettifor_number_Based_fraction_min.append(np.min(pettifor_number_List_Based_on_fraction))
        total_pettifor_number_Based_fraction_mean.append(np.mean(pettifor_number_List_Based_on_fraction))
        total_ElecAffinity_Based_fraction_max.append(np.max(ElecAffinity_List_Based_on_fraction))
        total_ElecAffinity_Based_fraction_min.append(np.min(ElecAffinity_List_Based_on_fraction))
        total_ElecAffinity_Based_fraction_median.append(np.median(ElecAffinity_List_Based_on_fraction))
        total_ElecAffinity_Based_Elemental_min.append(np.min(ElecAffinity_List))
        total_ElecAffinity_Based_Elemental_range.append(np.ptp(ElecAffinity_List))
        total_heat_formation_Based_fraction_mean.append(np.mean(heat_formation_List_Based_on_fraction))
        total_heat_formation_Based_fraction_max.append(np.max(heat_formation_List_Based_on_fraction))
        total_heat_formation_Based_Elemental_max.append(np.max(heat_formation_List))
        total_heat_formation_Based_Elemental_range.append(np.ptp(heat_formation_List))
        total_First_Ionis_Energy_Based_Elemental_mean.append(np.mean(First_Ionis_Energy_List))
        total_dipole_polarizability_Based_fraction_range.append(np.ptp(dipole_polarizability_List_Based_on_fraction))
        total_dipole_polarizability_Based_Elemental_median.append(np.median(dipole_polarizability_List))
        total_dipole_polarizability_Based_Elemental_range.append(np.ptp(dipole_polarizability_List))
        total_Ionic_Radius_Based_Elemental_range.append(np.ptp(Ionic_Radius_List))
        total_Ionic_Radius_Based_Elemental_min.append(np.min(Ionic_Radius_List))
        total_Ionic_Radius_Based_Elemental_max.append(np.max(Ionic_Radius_List))
        total_Number_Elec_Valence_Based_Elemental_median.append(np.median(Number_Elec_Valence_List))
        total_period_number_Based_Elemental_max.append(np.max(period_number_List))
        total_period_number_Based_Elemental_range.append(np.ptp(period_number_List))
        total_Electric_Conduct_Based_Elemental_min.append(np.min(Electric_Conduct_List))
        total_electronegativ_Based_Elemental_median.append(np.median(electronegativ_List))
        total_electronegativ_Based_fraction_max.append(np.max(electronegativ_List_Based_on_fraction))
        total_unpaired_electron_Based_Elemental_max.append(np.max(unpaired_electron_List))

    sample_for_prediction_Tc['range_Thermal_conduct_Elemental']=total_Thermal_conduct_Based_Elemental_range  
    sample_for_prediction_Tc['median_Thermal_conduct_fraction']=total_Thermal_conduct_Based_fraction_median
    sample_for_prediction_Tc['range_Pettifor_Elemental']=total_pettifor_number_Based_Elemental_range
    sample_for_prediction_Tc['min_Pettifor_fraction']=total_pettifor_number_Based_fraction_min
    sample_for_prediction_Tc['mean_Pettifor_fraction']=total_pettifor_number_Based_fraction_mean
    sample_for_prediction_Tc['max_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_max
    sample_for_prediction_Tc['min_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_min
    sample_for_prediction_Tc['median_ElecAffinity_fraction']=total_ElecAffinity_Based_fraction_median
    sample_for_prediction_Tc['range_ElecAffinity_Elemental']=total_ElecAffinity_Based_Elemental_range
    sample_for_prediction_Tc['min_ElecAffinity_Elemental']=total_ElecAffinity_Based_Elemental_min
    sample_for_prediction_Tc['mean_heat_formation_fraction']=total_heat_formation_Based_fraction_mean
    sample_for_prediction_Tc['max_heat_formation_fraction']=total_heat_formation_Based_fraction_max
    sample_for_prediction_Tc['max_heat_formation_Elemental']=total_heat_formation_Based_Elemental_max
    sample_for_prediction_Tc['range_heat_formation_Elemental']=total_heat_formation_Based_Elemental_range
    sample_for_prediction_Tc['mean_First_Ionis_Energy_Elemental']=total_First_Ionis_Energy_Based_Elemental_mean
    sample_for_prediction_Tc['range_dipole_polarizability_fraction']=total_dipole_polarizability_Based_fraction_range
    sample_for_prediction_Tc['median_dipole_polarizability_Elemental']=total_dipole_polarizability_Based_Elemental_median
    sample_for_prediction_Tc['range_dipole_polarizability_Elemental']=total_dipole_polarizability_Based_Elemental_range
    sample_for_prediction_Tc['range_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_range
    sample_for_prediction_Tc['min_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_min
    sample_for_prediction_Tc['max_Ionic_Radius_Elemental']=total_Ionic_Radius_Based_Elemental_max
    sample_for_prediction_Tc['median_NumElecValence_Elemental']=total_Number_Elec_Valence_Based_Elemental_median
    sample_for_prediction_Tc['max_period_Elemental']=total_period_number_Based_Elemental_max
    sample_for_prediction_Tc['range_period_Elemental']=total_period_number_Based_Elemental_range
    sample_for_prediction_Tc['min_Electric_Conduct_Elemental']=total_Electric_Conduct_Based_Elemental_min
    sample_for_prediction_Tc['median_ElecGativ_Elemental']=total_electronegativ_Based_Elemental_median
    sample_for_prediction_Tc['max_ElecGativ_Fraction']=total_electronegativ_Based_fraction_max
    sample_for_prediction_Tc['max_UnpairedElec_Elemental']=total_unpaired_electron_Based_Elemental_max

    # Finding Fractions
    sample_for_prediction_Tc.index=range(0,len(sample_for_prediction_Tc))
    # ele_fra_mat=element_fraction_material
    ele_fra_mat=np.zeros([len(sample_for_prediction_Tc), 103])
    ef = ElementFraction()
    for index in sample_for_prediction_Tc.index:
        com=Composition(sample_for_prediction_Tc.loc[index]["element"])
        ele_fra_mat[index] = ef.featurize(com)
    df_Fraction_Element=pd.DataFrame(ele_fra_mat,columns=['H_Frac','He_Frac','Li_Frac','Be_Frac','B_Frac','C_Frac','N_Frac','O_Frac','F_Frac',
    'Ne_Frac','Na_Frac','Mg_Frac','Al_Frac','Si_Frac','P_Frac',
    'S_Frac','Cl_Frac','Ar_Frac','K_Frac','Ca_Frac','Sc_Frac','Ti_Frac','V_Frac','Cr_Frac','Mn_Frac','Fe_Frac','Co_Frac','Ni_Frac',
    'Cu_Frac','Zn_Frac','Ga_Frac','Ge_Frac','As_Frac','Se_Frac','Br_Frac','Kr_Frac','Rb_Frac','Sr_Frac','Y_Frac',
    'Zr_Frac','Nb_Frac','Mo_Frac','Tc_Frac','Ru_Frac','Rh_Frac','Pd_Frac','Ag_Frac','Cd_Frac','In_Frac','Sn_Frac','Sb_Frac',
    'Te_Frac','I_Frac','Xe_Frac','Cs_Frac','Ba_Frac','La_Frac','Ce_Frac','Pr_Frac','Nd_Frac','Pm_Frac','Sm_Frac','Eu_Frac',
    'Gd_Frac','Tb_Frac','Dy_Frac','Ho_Frac','Er_Frac','Tm_Frac','Yb_Frac','Lu_Frac','Hf_Frac','Ta_Frac','W_Frac','Re_Frac',
    'Os_Frac','Ir_Frac','Pt_Frac','Au_Frac','Hg_Frac','Tl_Frac','Pb_Frac','Bi_Frac','Po_Frac','At_Frac','Rn_Frac','Fr_Frac',
    'Ra_Frac','Ac_Frac','Th_Frac','Pa_Frac','U_Frac','Np_Frac','Pu_Frac','Am_Frac','Cm_Frac','Bk_Frac','Cf_Frac',
    'Es_Frac','Fm_Frac','Md_Frac','No_Frac','Lr_Frac'])
    sample_for_prediction_Tc_30Features_103columns= pd.concat([sample_for_prediction_Tc, df_Fraction_Element], axis=1, join='inner')
    sample_for_prediction_Tc_New=sample_for_prediction_Tc_30Features_103columns[List_Column_Names_DataG]
    predictions = load_CatBoost_Model_DataG.predict(sample_for_prediction_Tc_New)
    # print(predictions[0],end=" ")
    # loaded_model = joblib.load(CatBoost_Model_DataG)  # Load the model from 'model.joblib'
    # predictions= loaded_model.predict(sample_for_prediction_Tc_New)
    Tc_pred=predictions[0]
    st.write('Tc =',Tc_pred, 'K')
    return Tc_pred

def Main_Function():
    st.header("Prediction Tc for superconducting materials using machine learning algorithms")
    # st.success("Prediction Tc for superconducting materials using machine learning algorithms")

    # Load the data
    load_data()

    New_Sample= st.text_input(label='***Please enter a compound***',placeholder='Example: MgB2 or B2Mg or ...')

    st.write('Your desired compound: ', New_Sample)
    st.write(pd.DataFrame({'element': [New_Sample],}))

    # در زیر میخواهیم یک دیتافریم بسازیم و سپس آن را در لوکال هاست نمایش دهیم با استریملیت
    data = {'element': [New_Sample]}
    New_Sample = pd.DataFrame(data)
    st.dataframe(New_Sample)
    # st.write(type(New_Sample))

    

    train_model(x_DataG_13022_30Features_103columns, y_DataG_13022)

    save_model(CatBoost_Model_DataG)

    load_train_model()

    separation_compounds(New_Sample)

    # Make prediction
    predict_button = st.button("Predict ") # یک دکمه با پریدیکشن ساخته میشه که کاربر همینکه اینو وارد کنه فرآیند محاسبه انجام میشه و جواب نهایی رو بهش نمایش میده
    if predict_button:
        Tc_pred = Prediction_Tc(load_CatBoost_Model_DataG, sample_for_prediction_Tc)
        # prediction = predict(loaded_model, [input_data])
        st.success(f"Prediction = {Tc_pred}")
    






if __name__ == '__main__':
    Main_Function()
    # Main_Function(New_Sample)
    




# title = st.text_input(label='**Please enter a compound**',
#                       value='Example: MgB2 or B2Mg or ...')

# st.write('Your desired compound: ', title)

# "------------------------------------------------------------"

# name_compound= st.text_input(label='***Please enter a compound***',
#                       placeholder='Example: MgB2 or B2Mg or ...')

# st.write('Your desired compound: ', name_compound)

# "----------------------------------------------------------------"

# #### radio button
# # st.header('Please select the dataset to use for predicting the transition temperature of Sc.')
# input_data = st.radio(

#     label="**Please select the dataset to use for predicting the transition temperature of Sc.**",
#     options=('DataG (recommended)', 'DataH'))
# if input_data == 'DataG (recommended)':

#     st.write('You selected DataG.')
# else:
#     st.write("You selected DataH.")

# "------------------------------------------------------------"

# ##### selectbox
# input_data_1 = st.selectbox(

#     '***Please select the dataset to use for predicting the transition temperature of Sc.***',
#     ('DataG (recommended)', 'DataH'))
# st.write('You selected :', input_data_1)





# "-------------------------------------------------------------"
# st.markdown("## for soraya package ")
# uploaded_file = st.file_uploader("***Choose a file***")

# if uploaded_file is not None:
#     # Can be used wherever a "file-like" object is accepted:
#     dataframe = pd.read_csv(uploaded_file)
#     st.write(dataframe)

# "--------------------------------------------------------------"

# text_content="""salam bar shoma"""
# st.download_button(label="Soraya output file", data=text_content)

# "--------------------------------------------------------------"
