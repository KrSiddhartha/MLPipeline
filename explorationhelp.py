import pandas as pd


## Distribution table
def distTab(df, clN=None, bins=None):
    """
    Generate distribution table with numbers and percentage.
    Parameters
    ----------
    df : pandas Dataframe or list. The one that will be checked and copied.
    clN : column name for which distribution needs to be generated.
    bins : number of bins required in a contineous variable.
    Raises
    ------
    TypeError
        If the input is not a Pandas DataFrame or a list
    Returns
    -------
    X : pandas Dataframe.
        
    """
    if type(df)==pd.core.series.Series: # When the input is a series
        tmpDf = pd.DataFrame(df.value_counts(bins=bins))
        tmpDf = pd.merge(tmpDf,
                         pd.DataFrame(round(df.value_counts(bins=bins, normalize=True)*100,1)),
                         on=tmpDf.index)
        tmpDf.columns = ["Value", "Frequency", "Percentage"]
    elif type(df)==list: # When the input is a list
        df = pd.Series(df)
        tmpDf = pd.DataFrame(df.value_counts(bins=bins))
        tmpDf = pd.merge(tmpDf,
                         pd.DataFrame(round(df.value_counts(bins=bins, normalize=True)*100,1)),
                         on=tmpDf.index)
        tmpDf.columns = ["Value", "Frequency", "Percentage"]
    elif type(df)==pd.core.frame.DataFrame: # When the input is a pandas dataframe
        tmpDf = pd.DataFrame(df[clN].value_counts(bins=bins))
        tmpDf = pd.merge(tmpDf,
                         pd.DataFrame(round(df[clN].value_counts(bins=bins, normalize=True)*100,1)),
                         on=tmpDf.index)
        tmpDf.columns = [clN, "Frequency", "Percentage"]
        
    return tmpDf