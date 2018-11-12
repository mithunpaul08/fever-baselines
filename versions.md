
# Versions
These are the various versions in the fact verification code development cycle (and what they do) at University of Arizona. Note, there must be only one version of this document and preferably exists in the master branch

| Date of modification |name of the branch |git SHA | change made | New F1 score | New overall accuracy | New average Precision|  Merged with master? |Type of Classifier SVM or Decomp Attn | Notes |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Nov 8th 2018|   person_space_c1 | 9f20b8b8e3e79c6b3410b51c3905f58042d42d28  | Replaced PERSON_C1 with PERSON C1 in the NER replacement code   | 0.46  | 0.5062006200620062  | 0.73| Yes | Decomp Attn | email dated:Fri, Nov 9, 3:26 PM  | 
| Nov 11th 2018|   mrksic | 4c7257f62b755a4eec19f61e29b444792964200c  | Does IR retrieval using their FEVERReader instead of our custom function.. We are in the middle of adding mrksic vectors in this branch, so no results. But  This should be added to master main for hand crafted development channel|  NA | NA  | NA | No | SVM | | 
