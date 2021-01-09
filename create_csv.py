# import os

# def to_csv(split):
#     with open(f"dataset_2/{split}.csv", "w") as data:
#         data.write("locutor,path,grupo,classe\n")
#         for classe in ["DISFARCE", "NORMAL"]:
#             for grupo in os.listdir(f"dataset_2/{split}/{classe}"):
#                 for arquivo in os.listdir(os.path.join("dataset_2", split, classe, grupo)):
#                     if os.path.splitext(arquivo)[-1] != ".wav":
#                         continue
#                     path = os.path.join("dataset_2", split, classe, arquivo)
#                     data.write(f"{arquivo.split('_')[0]},{path},{grupo.split('_')[0]},{classe}\n")

                    
# to_csv("train")     
# to_csv("test")


import os

def to_csv(split):
    with open(f"dataset_2_processed/{split}.csv", "w") as data:
        data.write("path,classe\n")
        for classe in ["DISFARCE", "NORMAL"]:               
            for folder in os.listdir(os.path.join("dataset_2_processed", split, classe)):
                for arquivo in os.listdir(os.path.join("dataset_2_processed", split, classe, folder)):    
                    path = os.path.join("dataset_2_processed", split, classe, folder, arquivo)
                    data.write(f"{path},{classe}\n")

                    
to_csv("train")     
to_csv("test")