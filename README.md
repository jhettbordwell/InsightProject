# InsightProject
A project to recommend psychiatric medications based on user preferences

A project completed for the Insight Data Science Health Data Science program, 3 weeks of work on an independent idea 
that culminated in a very beta-version webapp, called PsychedUp. The link for the webapp is associated with the repository,
but if you're interested, the slides associated with the presentation are also available online upon request.

Very briefly, this project was inspired by the nonadherence problem, specifically in relation to psychiatric medications. 
Between 30 and 60% of people by condition for many major psychiatric conditions don't follow their treatment regimen on 
a fully regular basis, and while some of this is unintentional, I believe strides can be made on the intentional side of things.
Two of the major causes of intentional nonadherence are:
* Negative opinions on the part of the patient about their medications
* The experience of negative side effects
PsychedUp is designed to allow a user to easily research medications in a way that should minimize their experience of 
side effects that they find intolerable, and hopefully the experience of taking charge in a conversation with their doctor,
and finding a medication that serves them better, will help them develop more positive feelings about the medication overall.

PsychedUp is a recommender system that ranks medications using an expected value framework to bring together the user concerns
(in the form of weights) with the experiences of people who have been on that medication
(in the form of side effect mention "frequencies", where in their current iteration, these frequencies are simply a boolean value). 
These experiences (as well as the relevant medications and condition information), are scraped from WebMD, in the form of reviews,
and Drugs.com (all the medication and side effect information), as well as FAERs/SIDER (all the rest of the side effect information).
NLP techniques are used to identify side effects in the text comments of the reviews (and the homebrew ID method was
validated against more standard classifiers using a hand-labeled dataset). Finally, medications are ranked and recommmended 
simply by comparing the medians of the expected value scores for each review, for each medication, dynamically on the website.

If I return to this project, the two major directions I would like to move in are to more seriously consider a method of 
minimizing the experience of any and all side effects AFTER the top three have been considered, and including weighting on the 
reviews based on the length of time the reviewer has been on the medication. An even more distant future step would involve
exploring the reviews and the performance of the my homebrew side effect ID method on the reviews present on other websites 
(like Drugs.com)...we'll see what the future brings. 

If you're interested in replicating this work, the order of execution of the scripts is:
1. ScrapeDrugsCom.py
2. ScrapeWebMD.py
3. ProcessWebMDReviews.py
4. (Run the routines in @mlbernauer's FAERs repository to create a sql database on all available FAERs data)
5. MakeMedsTables.py
6. Pull_sideeefects_byMed.py
7. IdentifySideEffects.py
8. CreateSideEffectClusters.py
9. CreateLabeledReviewSet.py
10. (Comparing homebrew algorithm against ML classifiers.ipynb)
11. CrossValidate.py

These steps correspond to:
1. Collecting all the medication and associated side effect information available on Drugs.com for ADHD, Bipolar Disorder, 
Generalized Anxiety Disorder, Major Depressive Episode, and Schizophrenia.
2. Collecting reviews for all of those medications from WebMD (HAS AN INTERACTIVE FEATURE)
3. Cleaning up the scraped reviews to be useable for analysis.
4. Creating a FAERs database to query to get more information on side effects for the medications.
5. Identifying medications and FAERs reports
6. Grabbing side effect information from SIDER based on FAERs reports
7. Combining side effect information from SIDER and Drugs.com to identify side effects in all of the reviews
8. Interactive clustering of the top 99.5% of side effect mentions into more general categories
9. Creating a labeled review set to evaluate the performance of the side effect algorithm against ML classifiers.
10. Actually doing that evaluation
11. Flagging useful reviews, creating "user" profiles, and comparing the "user" results in order to evaluate how effective 
the expected value framework is at recommending medications.
