# Ecommerce-Regression
Project Description:
You got some contract with an Ecommerce company based in New York City that sells clothing online but also have in-store style and clothing advice sessions.

Customers come in to the store, have sessions/meetings with a personal stylist,that they can go home and order either on a mobile app or website for the clothes they want.

The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you on contract to help them figure out!

Get the Data

We'll work with the Ecommerce Customers csv file from the company. It has Customer info, such as Email,Address and their color Avatar. Then it also has numerical value columns.

Avg.Session Length: Average session of in-store advice sessions.
Time on App: Average time spent on App in minutes.
Time on Website: Average time spent on Website in minutes.
Length of Membership:How many years the customer has been a member.

Exploratory Data analysis

Let's explore the data!
For the rest of the exercise we will only be using the numerical data of the csv file.

Training and Testing Data

Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.

Training the Model

Now its time to train our model on our trianing data!

Predicting Test Data

Now that we have fit our model, lets evaluate its performance by predicting off the test values!

Evaluating the model

Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score(R^2).

Residuals

You should have gotten a very good model with a good fit. Lets quickly explore the residuals to make sure everything was okay with our data.

Conclusion

We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important. Let's see if we can interpret the coefficients at all to get an idea.
