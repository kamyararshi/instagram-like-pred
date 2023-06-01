# Sources and Blah Blah
1. Main Challenge: https://equatorial-sternum-35b.notion.site/Instagram-Like-Prediction-270e4f0a8efd4e858be26e892bb25ed7
2. One Implementation from TDS: https://towardsdatascience.com/predict-the-number-of-likes-on-instagram-a7ec5c020203
3. Github repo: https://github.com/ralbertazzi/instagramlikeprediction/tree/master/IG


# Dataset Notes
1. We can get the top hashtags from [here](https://top-hashtags.com/instagram/)
2. GoogleVision API helps us get the labels from a given photo such as number of faces present, skyscrapers in the picture, ...
3. Instagram Influencer [Dataset](https://github.com/ksb2043/instagram_influencer_dataset)
4. Small dataset of top 200 influencers [kaggle](https://www.kaggle.com/datasets/syedjaferk/top-200-instagrammers-data-cleaned)


# Dataset Specifications

### Github Repo URL: https://github.com/ralbertazzi/instagramlikeprediction/tree/master

This repository contains two type of features in predicting likes:

 - A tabular data containing the metadata of each single post. Here is the full list of features in the ‘dataset.csv’ file. (The scraper code doesn’t exist !)

`numberPosts, website, urlProfile, username, numberFollowing, descriptionProfile,alias,numberFollowers,urlImgProfile,filename,date,urlImage,mentions,multipleImage,isVideo,localization,tags,numberLikes,url,description`


- A json file containing the objects detected in each image of the post. It capitalizes the google vision mode to extract the probabilities of objects in a specific image. (Extractor file exists!)


### Dataset URL: https://github.com/ksb2043/instagram_influencer_dataset

- Access has been requested from authors.

### Dataset LINK: [Kaggle](https://www.kaggle.com/datasets/syedjaferk/top-200-instagrammers-data-cleaned)

This dataset contains general data (e.g. Avg Views, Engagements, Comments, Topic, etc.) from top 200 influencers in instagram. (CURRENTLY NOTE APPLICABLE)

### Dataset used by TDS implementation [Link](https://github.com/gvsi/instagram-like-predictor/tree/master/profiles)
The dataset includes json files of each instagram user, in which there are various information such as user bio, number of posts, and all of the posts data. Image of each post is available via a given url.

### Papers
S. De, A. Maity, V. Goel, S. Shitole and A. Bhattacharya, "Predicting the popularity of instagram posts for a lifestyle magazine using deep learning," 2017 2nd International Conference on Communication Systems, Computing and IT Applications (CSCITA), Mumbai, 2017, pp. 174-177, doi: 10.1109/CSCITA.2017.8066548.

### Summary 
Overall, so far we have found a couple of repositories with their datasets. We can exploit the image url inside the datasets and use GoogelVision to find the labels in each image. We can use the labels alongside the post data, such as description and ..., to formulate our problem and predict the number of likes per post.
