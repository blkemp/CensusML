import numpy as np
import pandas as pd
import os
import operator

# Set a variable for current notebook's path for various loading/saving mechanisms
td_path = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.dirname(td_path)

full_category_list = ["Age", "Ancestry", "Birthplace", "Building occupation status", "Building type", 
"Country of Birth of Parents", "Dependent children in couple families", "Disability status", "Education", 
"Educational status", "Family Composition", "Family number parents + children", "Family type + Child age", 
"Family with children type", "Field of Study", "Heritage", "Hours worked", "Household Composition", 
"Household type", "Income", "Indigenous", "Industry", "Internet status", "Labour force status", 
"Labour Force Status of Female Parent", "Labour Force Status of Male Parent", "Language", "Language Dialect", 
"Language of father", "Language of Parents", "Language of mother", "Language Spoken at Home", "Location", 
"Marital Status", "Measure", "Mortgage", "Number", "Number of Commuting Methods", "Migration", 
"Number of Bedrooms", "Number of children", "Number of Hours", "Occupation", "Ownership", "Place of Birth", 
"Place of Residence class", "Place of Usual Residence", "Proficiency in Spoken English", 
"Relationship in Household", "Religion", "Religious Sect", "Rent", "Rental authority", "Rental cost", 
"School type", "Sex", "State", "Travel method type", "Unemployed Status", "Unpaid status", "Visitor Origination", 
"Volunteering status", "Year of Arrival", "Measure", "Total"
]

full_tables_list = ["G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10", 
"G11", "G12", "G13", "G14", "G15", "G16", "G17", "G18", "G19", "G20", "G21", "G22", "G23", 
"G24", "G25", "G26", "G27", "G28", "G29", "G30", "G31", "G32", "G33", "G34", "G35", "G36", 
"G37", "G38", "G39", "G40", "G41", "G42", "G43", "G44", "G45", "G46", "G47", "G48", "G49", 
"G50", "G51", "G52", "G53", "G54", "G55", "G56", "G57", "G58", "G59"]

cat_import_dtype = {x: str for x in full_category_list}

def return_categories(category_list = full_category_list):
    return category_list


def return_tables(table_list = full_tables_list):
    return table_list


def return_relevant_categories(tables_list = full_tables_list):
    '''
    Reads through the Metadata csv to filter for categories which have data in the input tables list

    INPUTS:
    tables_list: LIST of STRING objects - the list of DataPack file references containing the categories
        you are interested in

    OUTPUTS:
    LIST of categories available in in the selected tables
    '''
    df_meta = pd.read_csv('{}\\Data\\Metadata\\Metadata_2016_w_Category_Values.csv'.format(env_path), dtype=cat_import_dtype)

    # filter the dataframe to only include the input tables, using only the first 3 digits to avoid things like G57B 
    # being distinct from G57A, etc.
    table_rest_df = df_meta[df_meta['DataPack file'].str[:3].isin(tables_list)]
    
    # sum the measure columns to determine those with measure counts > 0
    cats_in_tbls = table_rest_df[full_category_list].count()
    cats_in_tbls = cats_in_tbls[cats_in_tbls > 0]
    
    return cats_in_tbls.index.tolist()

def return_relevant_tables(categories_list = full_category_list, category_field_list = [], category_intersection = False):
    '''
    Reads through the Metadata csv to filter for tables which include the features included in categories_list

    INPUTS:
    categories_list: LIST of STRING objects - the list of categories you are interested in, e.g. Age, Sex, Place of Birth
    category_field_list: LIST of STRING objects - filters to pass over the feature list to subset by specified category values
    where possible, e.g. "65_74_years|Age" will subset the measures to *only* include those which have a "65_74_years" component where 
    there the category of "Age" applies to that feature. For example would keep all items under "Number of Commuting Methods" if these 
    were otherwise included in the selection because there is no age split for these features.
    category_intersection: BOOLEAN - whether to filter the selections based on an intersection (and) or union (or) of selected terms

    OUTPUTS:
    Pandas dataframe object with two columns, one for the Datapack table file reference and one of the table name itself
    '''

    df_cat_measure = pd.read_csv('{}\\Data\\Metadata\\Category_Measure_reference.csv'.format(env_path), dtype={'2016_Table':str})
    
    # filter for chosen categories + fields
    df_cat_measure = df_cat_measure[df_cat_measure['Category'].isin(categories_list)]
    
    if category_field_list != []:
        # category field value includes category after bar delimiter so need to split out just the measure
        category_field_list = [i[0] for i in [x.split("|") for x in category_field_list]]
        print(category_field_list)
        df_cat_measure = df_cat_measure[df_cat_measure['Measure'].isin(category_field_list)]

    # turn the table names into a simple list
    table_list = []
    for table_ls in [x.split("|") for x in df_cat_measure['2016_Table'].tolist()]:
        table_list.extend(table_ls)
        
    table_list = list(set(table_list))
    
    df_tbl = pd.read_csv('{}\\Data\\Metadata\\2016_table_reference.csv'.format(env_path), dtype=str)
    df_tbl = df_tbl[df_tbl['DataPack file'].isin(table_list)]
    return df_tbl[['DataPack file','Table name']]

def return_relevant_features(categories_list = full_category_list, tables_list = full_tables_list, category_field_subset = [], category_intersection = False):
    '''
    Reads through the Metadata csv to filter for features which make up the measures included 
    in measure_list, optionally filtered for prespecified tables and category values

    INPUTS:
    categories_list: LIST of STRING objects - the list of categories you are interested in, e.g. Age, Sex, Place of Birth
    tables_list: LIST of STRING objects - the list of tables (datapack files) which include the measures you are
    interested in, e.g. G01, G59A
    category_field_subset: LIST of STRING objects - filters to pass over the feature list to subset by specified category values
    where possible, e.g. "65_74_years|Age" will subset the measures to *only* include those which have a "65_74_years" component where 
    there the category of "Age" applies to that feature. For example would keep all items under "Number of Commuting Methods" if these 
    were otherwise included in the selection because there is no age split for these features. 
    category_intersection: BOOLEAN - whether to filter the selections based on an intersection (and) or union (or) of selected terms

    OUTPUTS:
    Pandas dataframe object with two columns, one for a user friendly measure description and one for the measure as referenced
    in the ABS documentation
    '''
    df_meta = pd.read_csv('{}\\Data\\Metadata\\Metadata_2016_w_Category_Values.csv'.format(env_path), dtype=cat_import_dtype)
    
    # sum the measure columns in order to create a quick and dirty "or" filter for multiple categories    
    df_meta['filter_col'] = (df_meta[categories_list])[categories_list].count(axis=1)

    # filter for category and table
    if category_intersection and len(categories_list)>1 and (categories_list != full_category_list):
        df_meta = df_meta[(df_meta['filter_col']==len(categories_list)) & df_meta['DataPack file'].str[:3].isin(tables_list)]
    else:
        df_meta = df_meta[(df_meta['filter_col']>0) & df_meta['DataPack file'].str[:3].isin(tables_list)]
    
    # filter for category subset
    if len(category_field_subset)>0:
        # transform the subset list into an iterable dataframe
        cfs_df = pd.DataFrame()
        cfs_df['Field'] = [i[0] for i in [x.split("|") for x in category_field_subset]]
        cfs_df['Cat'] = [i[1] for i in [x.split("|") for x in category_field_subset]]
        
        # iterate over each category to subset the applicable fields and filter the main df (still leaving nulls in place as well)
        for cfs_category in cfs_df['Cat'].drop_duplicates().tolist():
            cfs_fields = cfs_df[cfs_df['Cat']==cfs_category]['Field'].drop_duplicates().tolist()
            # because of certain features which contain two of the same feature (e.g. parent age for male and female parent)
            # need to filter for substrings by join the list with an or delimiter and applying a regex search
            pattern = '|'.join(cfs_fields)
            df_meta = df_meta[df_meta[cfs_category].str.contains(pattern, na=True)]
            
            # if this becomes a performance issue look into aho-corasick
            # https://stackoverflow.com/questions/48541444/pandas-filtering-for-multiple-substrings-in-series/48600345#48600345


    df_meta['Measure Desc'] = df_meta['Measures'].str.replace('_',' ')
    df_meta['Measure Desc'] = df_meta['Measure Desc'].str.replace('|',' | ')

    df_meta = df_meta[['Measures','Measure Desc']]

    return df_meta

def return_features_subsets(categories_list = full_category_list, tables_list = full_tables_list, category_intersection = False):
    '''
    Reads through the Category_Measure_reference csv to filter for values associated with the 
    categories included in categories_list and available in the tables specified in tables_list

    INPUTS:
    categories_list: LIST of STRING objects - the list of categories you are interested in (e.g. Age, Sex, Place of Birth)
                    output from "category" dropdown options.
    category_intersection: BOOLEAN - whether to filter the selections based on an intersection (and) or union (or) of selected terms

    OUTPUTS:
    Pandas dataframe object with index of user friendly measure description of the field 
    (e.g. Father only born overseas - Country of Birth of Parents)
    and column of raw measure|category text itself for use in later filtering
    (e.g. Father_only_born_overseas|Country of Birth of Parents)
    Might be better to replace this second output with a dict?
    '''

    # import Category_Measure_reference.csv as DF
    df_cat_measure = pd.read_csv('{}\\Data\\Metadata\\Category_Measure_reference.csv'.format(env_path))
    
    # filter measure column based on isin list
    df_cat_measure = df_cat_measure[df_cat_measure['Category'].isin(categories_list)]
    
    # filter tables for partial matches
    pattern = '|'.join(tables_list)
    df_cat_measure = df_cat_measure[df_cat_measure['2016_Table'].str.contains(pattern, na=True)]

    # create new column in format of "Measure - Category" with nice text formatting str.replace('_',' ')
    df_cat_measure['Desc'] = df_cat_measure['Measure'].str.replace('_', ' ') + ' - ' + df_cat_measure['Category']
    df_cat_measure['Measure|Category'] = df_cat_measure['Measure'] + '|' + df_cat_measure['Category']

    # sort descending by Category then measure
    df_cat_measure.sort_values(by=['Category', 'Desc'], ascending=True, inplace=True)

    df_cat_measure = df_cat_measure[['Measure|Category','Desc']].drop_duplicates(subset='Desc')

    df_cat_measure = df_cat_measure.set_index('Desc')

    return df_cat_measure


def return_selected_variables(tables_list, categories_list, category_values_list):
    '''
    Navigates the 2016 metadata file to select a list of features within the overall dataset population to use as 
    variables in building models. This is based on input references to tables, and refined/aggregated by a set of
    defined categories (e.g. age, sex, occupation, English proficiency, etc.) where available, and filtered/split for
    specific category values where possible (e.g. Female, Age 14 25, Speaks English Well or Very Well, etc.)

    INPUTS:
    tables_list: 
    categories_list:
    category_values_list:

    OUTPUTS:
    LIST of selected features as per their reference long description in the raw ABS data
    '''


    return None



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
    
    # from the "Categories" field, split an individual entry by the "|" character
    # to give the index of the measure you are interested in grouping by,
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

