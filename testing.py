import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, root_mean_squared_error, max_error, explained_variance_score
import matplotlib.pyplot as plt
import joblib
import os

val_stats = []
aro_stats = []


acoustic = ["acoustic/mfcc_max.csv","acoustic/mfcc_mean.csv","acoustic/prosody_max.csv","acoustic/prosody_mean.csv","acoustic/compare_max.csv","acoustic/compare_mean.csv","acoustic/emobase_max.csv","acoustic/emobase_mean.csv"]
linguistic = ["linguistic/mini_384.csv","linguistic/mpnet_768","linguistic/fasttext_ave.csv","linguistic/fasttext_sum.csv","linguistic/glove_ave.csv","linguistic/glove_sum.csv","linguistic/word2vec_ave.csv","linguistic/word2vec_sum.csv"]

    
def load_data(acou,ling):
    acou_data = pd.read_csv(acou) 
    ling_data = pd.read_csv(ling)
    label_data = pd.read_csv("labels/val_aro.csv")
    acoustic_vectors = np.array(acou_data['embedding'].apply(eval).tolist())
    text_embeddings = np.array(ling_data['embedding'].apply(eval).tolist())
    labels = label_data[['valence', 'arousal']].values
    #print(acoustic_vectors.shape," ",text_embeddings.shape," ",labels.shape)
    return acoustic_vectors, text_embeddings, labels

def concatenate_features(acoustic_vectors, text_embeddings):
    return np.hstack((acoustic_vectors,text_embeddings))

def evaluate_model(true_labels, predictions, label_name,x,y):
    #print("\nPerformance Metrics: ",label_name," with ",x+"_"+y,)
    r2 = r2_score(true_labels, predictions)
    rootmeansq = root_mean_squared_error(true_labels, predictions)
    maxerr = max_error(true_labels, predictions)
    expvar = explained_variance_score(true_labels, predictions)

    """ print(f"R2 Score: {r2:.4f}")
    print(f"Root Mean Squared Error: {rootmeansq:.4f}")
    print(f"Max Error: {maxerr:.4f}")
    print(f"Explained Variance Score: {expvar:.4f}") """

    return x+"_"+y,r2, expvar, rootmeansq, maxerr

""" def plot_results(true_values, predicted_values, label_name,x,y):
    samples = np.arange(1,len(true_values)+1)
    plt.figure(figsize=(10, 6))
    plt.scatter(samples,true_values,label="True Values", marker='o')
    plt.scatter(samples,predicted_values, label="Predicted Values", marker='x')
    plt.title(label_name+" for "+x+"_"+y)
    plt.xlabel("Samples")
    plt.ylabel(label_name.capitalize())
    plt.legend()
    plt.grid()
    plt.savefig("plots/"+label_name[:3]+"_"+x+"_"+y+".png")
    plt.close() """

count = 0
for j in acoustic:
    for k in linguistic:
        xfull = j.split("/")
        x = xfull[1][:-4]
        yfull = k.split("/")
        y = yfull[1][:-4]
        count+=1
        print(f"Now working on {x} + {y} {count}/{len(acoustic)*len(linguistic)}")
        acoustic_vectors, text_embeddings, labels = load_data(j,k)

            # Concatenate acoustic vectors and text embeddings
        features = concatenate_features(acoustic_vectors, text_embeddings)

            # Create train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
            # Scale the datasets
        sc_x = StandardScaler()
        X_train = sc_x.fit_transform(X_train)
        X_test = sc_x.transform(X_test)
        

            # Iterate through the list of models to 
        for model in os.listdir("models/"):
            version = model.split("_")
            if x in model and y in model:
                if version[0] == "val":
                    svr = joblib.load(f"models/{model}")
                    y_pred = svr.predict(X_test)
                    newstats = evaluate_model(y_test[:, 1], y_pred,"valence",x,y)
                    val_stats.append(newstats)
                    #plot_results(y_test[:, i], y_pred,target,x,y)
                elif version[0] == "aro":
                    svr = joblib.load(f"models/{model}")
                    y_pred = svr.predict(X_test)
                    newstats = evaluate_model(y_test[:, 0], y_pred,"arousal",x,y)
                    aro_stats.append(newstats)
                    #plot_results(y_test[:, i], y_pred,target,x,y)

f = open("stats.txt","w")
metrics = ["R2","EVS","RME","Max"]
print("Metrics for Valence Models: ")
f.write("Metrics for Valence Models: ")
for i in range(4):
    print(f"\n\n{metrics[i]: >36}")
    f.write(f"\n\n{metrics[i]: >36}")
    if i<2:
        val_stats.sort(key=lambda x: x[i+1],reverse=True)
    else:
        val_stats.sort(key=lambda x: x[i+1])
    for j in val_stats:
        dataset = j[0]
        print(f"{dataset: >30} {j[i+1]:7.4f}")
        f.write(f"\n{dataset: >30} {j[i+1]:7.4f}")

print("\n\nMetrics for Arousal Models: ")
f.write("\n\nMetrics for Arousal Models: ")
for i in range(4):
    print(f"\n\n{metrics[i]: >36}")
    f.write(f"\n\n{metrics[i]: >36}")
    if i<2:
        aro_stats.sort(key=lambda x: x[i+1],reverse=True)
    else:
        aro_stats.sort(key=lambda x: x[i+1])
    for j in aro_stats:
        dataset = j[0]
        print(f"{dataset: >30} {j[i+1]:7.4f}")
        f.write(f"\n{dataset: >30} {j[i+1]:7.4f}")
