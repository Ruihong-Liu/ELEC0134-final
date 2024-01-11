import json
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
def table(history,json_file):
    # store history to a dic
    history_dict = {
        "accuracy": history.history['accuracy'],
        "val_accuracy": history.history['val_accuracy'],
        "loss": history.history['loss'],
        "val_loss": history.history['val_loss']
    }

    # save as JSON
    with open(json_file, "w") as json_file:
        json.dump(history_dict, json_file)

    return history
def table_show(json_file,png_file):
    #load data from json file
    data = pd.read_json(json_file)

    # DataFrame
    df = pd.DataFrame(data)

    # print DataFrame
    print(df)
    # Save DataFrame as PNG image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12) 
    the_table.scale(1.2, 1.2)

    plt.savefig(png_file, dpi=300)