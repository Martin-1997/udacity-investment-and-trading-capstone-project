import matplotlib.pyplot as plt

def plot_data(title, df):
    x = df["Date"]
    for column in df.columns.drop(["Date"]):
        plt.plot(x, df[column])
    plt.ylabel('Price in $')
    plt.xlabel('Date')
    plt.figure().set_figwidth(8)
    plt.figure().set_figheight(4)
    plt.show()


 