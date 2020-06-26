import pandas as pd

# returns value of index series generated by pandas
def l(df, i, string):
    ind = df.index
    return df.loc[ind[i], string]

# vehicle parameter class
class vehicle:
    # import function for class
    def __init__(self,df):
        df = df.set_index('Parameter')
        v = 'Value'
        ind = df.index
        self.t_k = l(df,0,v)
        self.f_k = l(df, 1, v)
        self.r_k = l(df, 2, v)
        self.f_damp = l(df, 3, v)
        self.r_damp = l(df, 4, v)
        self.f_usprung = l(df, 5, v)
        self.r_usprung = l(df, 6, v)
        self.f_sprung = l(df, 7, v)
        self.r_sprung = l(df, 8, v)

# generation of a vehicle
def vehicle_generator(file):
    param_df = pd.read_excel(file, header = 0, sheet_name = 'Driver')
    car = vehicle(param_df)
    return car

# write your test code here and execute
if __name__ == '__main__':
    car = vehicle_generator('driver.xlsx')
    print(car)