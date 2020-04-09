# Programming Custom Transformers, Pipelines, and Ensemble Models to Predict Venue Ratings from the Yelp Academic Dataset

<span id='back-up'>The Yelp academic dataset </span> (included in this repository) consists of about 38,000 business listings and their relevant information. Each listing includes 15 properties, e.g. `business_id`, `full_address`, `hours`, `categories`, etc ([Click here to see all of a listing's properties (anchors don't work in Github)](#listing)).

The **goal of this project** is to demonstrate the use of custom transformer classes, chaining custom tranformers/estimators into pipelines for composite estimators, and using feature unions to develop an ensemble model for venues' ratings prediction.

Although I will optimize my models using cross-validation, the goal of the project is *not to develop the best* model to predict star ratings based on the features in the dataset. This dataset doesn't include actual user reviews. A much more accurate predictive model would likely require natural language processing (NLP) of the costumer review data. With that caveat, let's dive into the data.