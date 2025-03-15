## Statement ##

This code is a non-commercial improvement based on CR-FIQA (https://github.com/fdbtrs/CR-FIQA) (Copyright © 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt), licensed under CC BY-NC 4.0. Modifications © [yuying/2024].  



## Evaluate  LFW CFP-FP CPLFW CALFW AGEDB ##
## Take the LFW dataset as an example ##

##  Prepare Evaluation Dataset ##
1. Download lfw.bin from the trainging dataset folder from insightface (https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)

Then put lfw.bin in data/bin/

2. cd feature_extraction and run: python extract_bin.py 


##  Generate the Feature Embedding from a target Face Recognition (FR) Model ##
3. A pretrained R100-based ArcFace FR model can be download from insightface (https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

Change the name of "backbone.pth" to "ms1mv3-r100-arcface.pth" and put it in "pretrained" folder

4. cd feature_extraction and run: python extract_emb.py 


##  Generate the Quality Score ##
5. We recommend using the ElasticFace-Arc model (https://drive.google.com/drive/folders/1q3ws_BQLmgXyiy2msvHummXq4pRqc1rx), which can be replaced by other FR models as well.

Download and change the name of "295672backbone.pth" to "ElasticFace-Arc.pth"  and put it in "pretrained" folder

6. cd feature_extraction and run: python extract_qs.py 

## Calculate pAUC ##

7. cd ERC and run: python erc.py



######################################################################################
######################################################################################

## Evaluate XQLFW ##

##  Prepare Evaluation Dataset ##
1. Download the aligned XQLFW dataset (xqlfw_aligned_112.zip and xqlfw_pairs.txt) from https://martlgap.github.io/xqlfw/pages/download.html

2. mkdir "data/quality_data/XQLFW/", unzip xqlfw_aligned_112.zip, put it and xqlfw_pairs.txt to "data/quality_data/XQLFW/"

3. cd feature_extraction and run: python extract_xqlfw.py

##  Generate the Feature Embedding from a target Face Recognition (FR) Model ##
4. cd feature_extraction and run: python extract_emb.py --dataset_path "../data/quality_data/XQLFW"

## Generate the Quality Score ##
5. run: python extract_qs.py --dataset_name "XQLFW"

## Calculate pAUC ##
6. cd ERC and run: python erc.py --eval_db "XQLFW"


######################################################################################
######################################################################################

## Evaluate IJB-C ##

##  Prepare Evaluation Dataset ##
1. download IJB-C dataset from https://www.nist.gov/programs-projects/face-challenges or https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_

2. unzip IJB-C and put it in "your_IJB_path"

3. cd feature_extraction and run: python extract_IJB.py (change the path in line 10 to "your_IJB_path")

##  Generate the Feature Embedding from a target Face Recognition (FR) Model ##
4. cd feature_extraction and run: python extract_emb.py --dataset_path "../data/quality_data/IJBC"

## Generate the Quality Score ##
5. run: python extract_qs.py --dataset_name "IJBC"

## Aggregate Template Features ##
6. change the name of "../data/quality_embeddings/IJBC_ArcFaceModel" to "../data/quality_embeddings/IJBC_ArcFaceModel_raw"

7. run: python ijb_pair_file.py (change the path in line 13 to "your_IJB_path")

## Calculate pAUC ##
8. cd ERC and run: python erc_ijbc.py --eval_db "IJBC"

## 1:1 mixed verification on IJB-C ##
9. cd feature_extraction and run: python eval_ijbc_qs.py --image-path "your_IJB_path"

