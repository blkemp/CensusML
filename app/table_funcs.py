import numpy as np
import pandas as pd
import os
import operator

# Set a variable for current notebook's path for various loading/saving mechanisms
td_path = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.dirname(td_path)

full_measures_list = ["Age", "Ancestry", "Birthplace", "Building occupation status", "Building type", 
"Country of Birth of Parents", "Dependent children in couple families", "Disability status",
"Education", "Educational status", "Family Composition", "Family number parents + children",
"Family type + Child age", "Family with children type", "Field of Study", "Heritage",
"Hours worked", "Household Composition", "Household type", "Income", "Indigenous", "Industry",
"Internet status", "Labour force status", "Labour Force Status of Female Parent",
"Labour Force Status of Male Parent", "Language", "Language Dialect", "Language of father", 
"Language of Parents", "Language of mother", "Language Spoken at Home", "Location", 
"Marital Status", "Mortgage", "Number", "Number of Commuting Methods", 
"Migration", "Number of Bedrooms", "Number of children", "Number of Hours", "Occupation", 
"Ownership", "Place of Birth", "Place of Residence class", "Place of Usual Residence", 
"Proficiency in Spoken English", "Relationship in Household", "Religion", "Religious Sect", 
"Rent", "Rental authority", "Rental cost", "School type", "Sex", "State", "Travel method type", 
"Unemployed Status", "Unpaid status", "Volunteering status", "Year of Arrival"]


def return_measures(measures_list = full_measures_list):
    return measures_list


def return_relevant_tables(measures_list = full_measures_list):
    '''
    Reads through the Metadata csv to filter for tables which included the features included in measure_list

    INPUTS:
    measures_list: LIST of STRING objects - the list of measures you are interested in, e.g. Age, Sex, Place of Birth

    OUTPUTS:
    Pandas dataframe object with two columns, one for the Datapack table file reference and on of the table name itself
    '''
    df_meta = pd.read_csv('{}\\Data\\Metadata\\Metadata_2016_refined.csv'.format(env_path))
    
    df_refined = pd.DataFrame()
    
    for measure_item in measures_list:
        if(df_refined.shape[0])==0:
            df_refined = df_meta[df_meta[measure_item]>0][['DataPack file','Table name']]
        else:
            df_refined = df_refined.append(df_meta[df_meta[measure_item]>0][['DataPack file','Table name']])
        
    return df_refined.drop_duplicates(subset='DataPack file')


def load_census_csv(table_list, statistical_area_code='SA3'):
    '''
    Navigates the file structure to import the relevant files for specified data tables at a defined statistical area level
    
    INPUTS
    table_list: LIST of STRING objects - the ABS Census Datapack table to draw information from (G01-G59)
    statistical_area_code: STRING - the ABS statistical area level of detail required (SA1-SA3)
    
    OUTPUTS
    A pandas dataframe
    '''
    statistical_area_code = statistical_area_code.upper()
    
    df_csv_load = pd.DataFrame()
    for index, table in enumerate(table_list):
        
        if index==0:
            df_csv_load = pd.read_csv('{}\\Data\\{}\\AUST\\2016Census_{}_AUS_{}.csv'.format(env_path,
                                                                                statistical_area_code,
                                                                                table,
                                                                                statistical_area_code
                                                                               ),
                                       engine='python')
        else:
            temp_df = pd.read_csv('{}\\Data\\{}\\AUST\\2016Census_{}_AUS_{}.csv'.format(env_path,
                                                                                statistical_area_code,
                                                                                table,
                                                                                statistical_area_code
                                                                               ),
                                       engine='python')
            merge_col = df_csv_load.columns[0]
            df_csv_load = pd.merge(df_csv_load, temp_df, on=merge_col)
    
    return df_csv_load


def refine_measure_name(table_namer, string_item, category_item, category_list):
    '''Simple function for generating measure names based on custom metadata information on ABS measures'''
    position_list = []
    for i, j in enumerate(category_item.split("|")):
        if j in category_list:
            position_list.append(i)
    return table_namer + '|' + '_'.join([string_item.split("|")[i] for i in position_list])


def load_table_refined(table_ref, category_list, statistical_area_code='SA3', drop_zero_area=True):
    '''
    Function for loading ABS census data tables, and refining/aggregating by a set of defined categories
    (e.g. age, sex, occupation, English proficiency, etc.) where available.
    
    INPUTS
    table_ref: STRING - the ABS Census Datapack table to draw information from (G01-G59)
    category_list: LIST of STRING objects - Cetegorical informatio to slice/aggregate information from (e.g. Age)
    statistical_area_code: STRING - the ABS statistical area level of detail required (SA1-SA3)
    drop_zero_area: BOOLEAN - an option to remove "non-geographical" area data points such as "no fixed address" or "migratory"
    '''
    df_meta = pd.read_csv('{}\\Data\\Metadata\\Metadata_2016_refined.csv'.format(os.getcwd()))
    index_reference = 'Area_index'
    
    # slice meta based on table
    meta_df_select = df_meta[df_meta['Profile table'].str.contains(table_ref)].copy()
    
    # for category in filter_cats, slice based on category >0
    for cat in category_list:
        # First, check if there *are* any instances of the given category
        try:
            if meta_df_select[cat].sum() > 0:
                # If so, apply the filter
                meta_df_select = meta_df_select[meta_df_select[cat]>0]
            else:
                pass # If not, don't apply (otherwise you will end up with no selections)
        except:
            pass
        
    # select rows with lowest value in "Number of Classes Excl Total" field
    min_fields = meta_df_select['Number of Classes Excl Total'].min()
    meta_df_select = meta_df_select[meta_df_select['Number of Classes Excl Total'] == min_fields]
    
    # Select the table file(s) to import
    import_table_list = meta_df_select['DataPack file'].unique()
    
    # Import the SA data tables
    df_data = load_census_csv(import_table_list, statistical_area_code.upper())
    
    # Select only columns included in the meta-sliced table above
    df_data.set_index(df_data.columns[0], inplace=True)
    refined_columns = meta_df_select.Short.tolist()
    df_data = df_data[refined_columns]
    
    # aggregate data by:
    # transposing the dataframe
    df_data_t = df_data.T.reset_index()
    df_data_t.rename(columns={ df_data_t.columns[0]: 'Short' }, inplace = True)
    # merging with the refined meta_df to give table name, "Measures" and "Categories" fields
    meta_merge_ref = meta_df_select[['Short','Table name','Measures','Categories']]
    df_data_t = df_data_t.merge(meta_merge_ref, on='Short')
    
    # from the "Categories" field, you should be able to split an individual entry by the "|" character
    # to give the index of the measure you are interested in grouping by
    # create a new column based on splitting the "Measure" field and selecting the value of this index/indices
    # Merge above with the table name to form "[Table_Name]|[groupby_value]" to have a good naming convention
    # eg "Method_of_Travel_to_Work_by_Sex|Three_methods_Females"
    df_data_t[index_reference] = df_data_t.apply(lambda x: refine_measure_name(x['Table name'], 
                                                                               x['Measures'], 
                                                                               x['Categories'], 
                                                                               category_list), axis=1)
    
    # then groupby this new column 
    # then transpose again and either create the base data_df for future merges or merge with the already existing data_df
    df_data_t = df_data_t.drop(['Short','Table name','Measures','Categories'], axis=1)
    df_data_t = df_data_t.groupby([index_reference]).sum()
    df_data_t = df_data_t.T
    
    if drop_zero_area:
        df_zero_area = pd.read_csv('{}\\Data\\Metadata\\Zero_Area_Territories.csv'.format(os.getcwd()))
        zero_indicies = set(df_zero_area['AGSS_Code_2016'].tolist())
        zero_indicies_drop = set(df_data_t.index.values).intersection(zero_indicies)
        df_data_t = df_data_t.drop(zero_indicies_drop, axis=0)
    
    return df_data_t


def load_tables_specify_cats(table_list, category_list, statistical_area_code='SA3'):
    '''
    Function for loading ABS census data tables, and refining/aggregating by a set of defined categories
    (e.g. age, sex, occupation, English proficiency, etc.) where available.
    
    INPUTS
    table_list: LIST of STRING objects - list of the ABS Census Datapack tables to draw information from (G01-G59)
    category_list: LIST of STRING objects - Cetegorical information to slice/aggregate information from (e.g. Age)
    statistical_area_code: STRING - the ABS statistical area level of detail required (SA1-SA3)
    
    OUTPUTS
    A pandas dataframe
    '''
    for index, table in enumerate(table_list):
        if index==0:
            df = load_table_refined(table, category_list, statistical_area_code)
            df.reset_index(inplace=True)
        else:
            temp_df = load_table_refined(table, category_list, statistical_area_code)
            temp_df.reset_index(inplace=True)
            merge_col = df.columns[0]
            df = pd.merge(df, temp_df, on=merge_col)
    
    df.set_index(df.columns[0], inplace=True)
    
    return df


def sort_series_abs(S):
    '''Takes a pandas Series object and returns the series sorted by absolute value'''
    temp_df = pd.DataFrame(S)
    temp_df['abs'] = temp_df.iloc[:,0].abs()
    temp_df.sort_values('abs', ascending = False, inplace = True)
    return temp_df.iloc[:,0]


def WFH_create_Xy(stat_a_level, load_tables, load_features):
    '''
    A function which compiles a set of background information from defined ABS census tables and 
    creates input and output vectors to allow model training for the  "Work from home participation 
    rate" in a given region. Cleans the data for outliers (defined as >3 standard deviations from the mean) in the
    WFH participation rate, and scales all data by dividing by the "Total Population" feature for each region.
    
    INPUTS
    stat_a_level - String. The statistical area level of information the data should be drawn from (SA1-3)
    load_tables - List of Strings. A list of ABS census datapack tables to draw data from (G01-59)
    load_features - List of Strings. A list of population characteristics to use in analysis (Age, Sex, labor force status, etc.)
    
    OUTPUTS
    X - pandas DataFrame - a dataframe of features from the census datapacks tables, normalised by dividing each feature by 
                            the population attributable to the region
    y - pandas series - the Work from Home Participation Rate by region
    
    '''
    # Load table 59 (the one with commute mechanism) and have a quick look at the distribution of WFH by sex
    response_vector = 'WFH_Participation'
    df_travel = load_table_refined('G59', ['Number of Commuting Methods'], statistical_area_code=stat_a_level)
    cols_to_delete = [x for x in df_travel.columns if 'Worked_at_home' not in x]
    df_travel.drop(cols_to_delete,axis=1, inplace=True)

    df_pop = load_census_csv(['G01'], statistical_area_code=stat_a_level)
    df_pop.set_index(df_pop.columns[0], inplace=True)
    df_pop = df_pop.drop([x for x in df_pop.columns if 'Tot_P_P' not in x], axis=1)
    df_travel = df_travel.merge(df_pop, left_index=True, right_index=True)
    
    # Create new "Work From Home Participation Rate" vector to ensure consistency across regions
    # Base this off population who worked from home divided by total population in the region
    df_travel[response_vector] = (df_travel['Method of Travel to Work by Sex|Worked_at_home']/
                                  df_travel['Tot_P_P'])
    # Drop the original absolute values column
    df_travel = df_travel.drop(['Method of Travel to Work by Sex|Worked_at_home'], axis=1)
    
    # load input vectors
    input_vectors = load_tables_specify_cats(load_tables, load_features, statistical_area_code=stat_a_level)
    
    # Remove duplicate column values
    input_vectors = input_vectors.T.drop_duplicates().T

    # Bring in total population field and scale all the values by this item
    input_vectors = input_vectors.merge(df_pop, left_index=True, right_index=True)

    # convert input features to numeric
    cols = input_vectors.columns
    input_vectors[cols] = input_vectors[cols].apply(pd.to_numeric, errors='coerce')

    # Drop rows with zero population
    input_vectors = input_vectors.dropna(subset=['Tot_P_P'])
    input_vectors = input_vectors[input_vectors['Tot_P_P'] > 0]

    # Scale all factors by total region population
    for cols in input_vectors.columns:
        if 'Tot_P_P' not in cols:
            input_vectors[cols] = input_vectors[cols]/input_vectors['Tot_P_P']

    # merge and drop na values from the response vector
    df_travel = df_travel.merge(input_vectors, how='left', left_index=True, right_index=True)
    df_travel = df_travel.dropna(subset=[response_vector])

    df_travel = df_travel.drop([x for x in df_travel.columns if 'Tot_P_P' in x], axis=1)

    # drop outliers based on the WFHPR column
    # only use an upper bound for outlier detection in this case, based on 3-sigma variation 
    # had previously chosen to remove columns based on IQR formula, but given the skew in the data this was not effective
    #drop_cutoff = (((df_travel[response_vector].quantile(0.75)-df_travel[response_vector].quantile(0.25))*1.5)
    #               +df_travel[response_vector].quantile(0.75))
    drop_cutoff = df_travel[response_vector].mean() + (3* df_travel[response_vector].std())
    df_travel = df_travel[df_travel[response_vector] <= drop_cutoff]
    
    # Remove duplicate column values
    df_travel = df_travel.T.drop_duplicates().T

    # Create X & y
    X = df_travel.drop(response_vector, axis=1)
    y = df_travel[response_vector]

    # Get the estimator
    return X, y

