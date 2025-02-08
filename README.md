# 1: Preparation of the training set (verified manually by you and then sent for verification to our Japanese colleagues).
# 2: The fine-tuned model, deployed on our servers and showing good results and performance in tests.

1-3 days: research, data collection, ETL, proofreading and data verification)

To conduct a proper research and fine-tune our model in the best possible way,
I am about to use following datasets in japanese language and additionally collect more data tweets, facebook posts, etc.:
    - https://dell-research-harvard.github.io/HJDataset/ (Historical Japanese Documents dataset);
    - https://github.com/SkelterLabsInc/JaQuAD (Japanese Question Answering Dataset);   
    - https://github.com/cl-tohoku/PheMT (PheMT);
    - https://huggingface.co/datasets/sergicalsix/Japanese_NER_Data_Hub 

Also, I am about to use data searched on my own from open resources 
    (mainly that consists of such named entities correlated to Ukraine and Japan, direct and indirect mentioning included).
For that matter, I am about to use following resources (these and not limited to): 
    - https://japan.mfa.gov.ua/ja/about-ukraine
    - search from as many japanese tweets as possible with ウクライナ and similar in it; 
    - https://www.instagram.com/ukraine_sunflower_sakura/ and similar. 


Deep learning would be the best approach when it comes to NER, yet it requires lots of data; thus, I am about to gather as much as it's possible to.
Also, I am considering to try supervised learning for NER, since it's mostly used for this purpose.

1-2 days: fine-tuning the model 

Working on developing assistants to withdraw named entities using LLama-3.
    - fine-tune the model;
    - test the model;
    - apply precision, recall, f-1 metrics to check the model accuracy. 

1-2 days: visualization and proper testing part 

As one of last and important stages might be visualization of the results and applying various techniques to check the accuracy of the model. 

And as i've mentioned before: we can apply manual review and/or crowdsourcing, apply double annotation (have two annotators per tweet, and resolve disagreements}.
To solve this issue automatically, we can apply automated consistency checks: flag tweets where the extracted result does not match title_p_countries and use language models to detect potential missing mentions.
Then after the first pass, refine the detection model and re-evaluate.


