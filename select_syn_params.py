import pandas as pd


template = """
    "gmax"  : {g},
    "tau_d" : {tau_d},
    "tau_r" : {tau_r},
    "tau_f" : {tau_f},
    "Uinc"  : {U},
    "Erev"  : -75.0,
    "pconn" : 0.1,
"""

postsynaptic_neuron = 'CA1 Pyramidal (+)2223p'
filepath = '~/Data/Hippocampome/SynapsesParameters.csv'
syndata = pd.read_csv(filepath)


syndata = syndata[syndata['Postsynaptic Neuron'] == postsynaptic_neuron]
#syndata = syndata[ syndata['Presynaptic Neuron'].str.find("(+)") != -1 ]

file4saving = open("connections_as_code.txt", mode="w")


for idx, syn in syndata.iterrows():
    print(syn['Postsynaptic Neuron'], file=file4saving)
    print(syn['Presynaptic Neuron'], file=file4saving)

    code = template.format(**syn.to_dict())
    code = "{" + code + "},\n"
    print(code, file=file4saving)


    print("###################################", file=file4saving)

file4saving.close()
#syndata.to_csv('to_connections.csv')






