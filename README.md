****ENGN8535 Group T0 Project Assignment ****

**Members**
* Zeji Hui    / u6987679
* Yingjia Cai / u6592822
* Di Li       / u7039884

**Topic**
* Movie Recommendation Implement a proof-of-concept system for movie recommendation
on the MovieLens-100K dataset using two different methods and compare their performances.

**Useful Link**

1. Proposal editor: https://docs.google.com/document/d/1YF-LDtBbpNGvLxNPKidinfN4GiT5beKBgm28k7tcqHY/edit?usp=sharing
2. Neural Collaborative Filtering : https://github.com/hexiangnan/neural_collaborative_filtering
3. svd++ : https://github.com/lxmly/recsyspy

**implement**
this project implement a practical  movie recommendation software systemfor MovieLens-100k dataset, using collaborative filtering techniques.
there are 2 methods： “FUNK SVD” and collaborative filtering based deep learning methods. 

**files**
* main(Yingcai Jia / Zeji Hui): run this to test the Movie RecommendationSystem, to run it 
'''
if __name__ == '__main__':
    recommendation()
'''

* svd_cf(Yingcai Jia):implement of funk SVD, return a array the rating for each of the moview, 
in 984*1682 shape. each element represent the rating for a movie
* neurlnet(Yingcai Jia / Zeji Hui):the MLP Movie Recommendation with  collaborative filtering return a array the rating for each of the moview, 
in 984*1682 shape. each element represent the rating for a movie. with MLP  structures, the hyperparameters been well toned
* data_reader(Yingcai Jia / Zeji Hui):load the data and eaxtract the data we want and from the datast and transform into array type
* visual(Zeji Hui): visualized 100 random movies, try to find the relationship 





