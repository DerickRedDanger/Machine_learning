def add_title(df):
    df = df.copy()

    title_name = df['Name'].str.split(',').str[1]
    title = title_name.str.split('.').str[0]
    df['Title'] = title.str.strip()
    df['Title'] = df['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme': 'Mrs'})
    
    return df

# train_df['Title'].value_counts()
# Title
# Mr              517
# Miss            182
# Mrs             125
# Master           40
# Dr                7
# Rev               6
# Col               2
# Mlle              2
# Major             2
# Ms                1
# Mme               1
# Don               1
# Lady              1
# Sir               1
# Capt              1
# the Countess      1
# Jonkheer          1
# Name: count, dtype: int64

def group_rare_title(df):
    df = df.copy()

    rare_titles = [
    'Dr', 'Rev', 'Col', 'Major', 'Don', 'Lady',
    'Sir', 'Capt', 'the Countess', 'Jonkheer'
    ] 
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    
    return df

def add_full_title_feature(df):
    df = add_title(df)
    df = group_rare_title(df)

    return df