def add_title(df):
    df = df.copy()

    title_name = df['Name'].str.split(',').str[1]
    title = title_name.str.split('.').str[0]
    df['Title'] = title.str.strip()
    df['Title'] = df['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme': 'Mrs'})
    
    return df

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