This section is devoted to quantifying how Churn customers differentiate themselves. Out of the 21 features for each customer, 1 is an ID, 3 are continuous variables, and the rest are categorical variables. Out of those categoricals one is a label of whether or not they churned. So for each customer we have 16 categorical and 3 continuous variables of which we can use to describe why they churn.

### Categorical Variables
The best way to visualize the behavior of the average churned and retained customer customer is to plot each categorical feature in a polar plot. Looking at it this way makes it rather easy for us to decipher a pattern. When we get to the modeling section it's best to think that we are trying to train a computer to learn the patterns of the red blob from the blue blob. A few notes about this plot:

1) Most categorical features didn't arrive as binary vales - which is what is plotted in the polar plot. So many features, like Internet Service, had their 3 unique values of _Yes_, _No_, and _No Internet Service_ consolidated into just _Yes_ and _No_. Other features, such as Payment Method, had 4 features that were each turned into their own feature and plotted.

2) The pie and bar charts show the original percentage and composition respectively for each variable.

3) Just hover over a marker in the polar plot to examine it in the charts to the right. There's also a column description. Go ahead, give it a whack!
