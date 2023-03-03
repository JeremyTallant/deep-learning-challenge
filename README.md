# Deep Learning Challenge
![image](https://user-images.githubusercontent.com/112406455/220229099-70f49c0b-e659-4de6-b8a0-f118eb0a7cfc.png)
## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special considerations for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively
## Instructions
### Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:

   * What variable(s) are the target(s) for your model?
    
    The target variable(s) for the model is `IS_SUCCESSFUL`, as it represents the binary classification outcome variable of whether a charity donation was successful or not.
  
    * What variable(s) are the feature(s) for your model?
    
    The feature variables for the model are all the other columns in the DataFrame, excluding `IS_SUCCESSFUL`.
    
<img width="1213" alt="Screenshot 2023-03-03 at 4 43 30 PM" src="https://user-images.githubusercontent.com/112406455/222846018-ec501cf0-b4cd-49f1-be39-4248d2a7de0d.png">    
  
2. Drop the `EIN` and `NAME` columns.
<img width="1212" alt="Screenshot 2023-03-03 at 4 48 55 PM" src="https://user-images.githubusercontent.com/112406455/222847360-df27e6cf-9924-428c-bc2c-9d153047f551.png">

3. Determine the number of unique values for each column.
<img width="1213" alt="Screenshot 2023-03-03 at 4 50 03 PM" src="https://user-images.githubusercontent.com/112406455/222848116-738ed474-768d-4f65-aaa5-a48c31b6ca68.png">

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

`APPLICATION_TYPE`:

<img width="1214" alt="Screenshot 2023-03-03 at 4 53 46 PM" src="https://user-images.githubusercontent.com/112406455/222849162-1ca5fbb6-ba78-46ba-b646-a7be76253544.png">

`CLASSIFICATION`:

<img width="1210" alt="Screenshot 2023-03-03 at 4 54 06 PM" src="https://user-images.githubusercontent.com/112406455/222849629-5518c218-25c9-45be-90c8-d18a946c7671.png">

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.

`APPLICATION_TYPE`:

<img width="1214" alt="Screenshot 2023-03-03 at 4 58 10 PM" src="https://user-images.githubusercontent.com/112406455/222850172-086d4ced-ba33-44e4-b964-1708ea3e9fbc.png">

`CLASSIFICATION`:

<img width="1212" alt="Screenshot 2023-03-03 at 4 58 18 PM" src="https://user-images.githubusercontent.com/112406455/222850365-251ab6f0-bc85-4c41-a45a-c4f24edd1184.png">

6. Use pd.get_dummies() to encode categorical variables.
<img width="1214" alt="Screenshot 2023-03-03 at 5 00 36 PM" src="https://user-images.githubusercontent.com/112406455/222850880-ae14a17d-21ec-49de-91bf-44ed62d25d37.png">

7. Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.
<img width="1212" alt="Screenshot 2023-03-03 at 5 00 51 PM" src="https://user-images.githubusercontent.com/112406455/222851080-239e1c10-9272-4814-8149-eaaa9b5a8c6e.png">

8. Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.
<img width="1210" alt="Screenshot 2023-03-03 at 5 00 44 PM" src="https://user-images.githubusercontent.com/112406455/222851265-afe941f8-547b-4d04-88c5-6464c59c2df0.png">

### Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the Jupyter Notebook in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
<img width="1213" alt="Screenshot 2023-03-03 at 5 04 55 PM" src="https://user-images.githubusercontent.com/112406455/222852003-8d5f2c7e-ea87-4cad-b7ca-671db5a8f2c2.png">

3. Create the first hidden layer and choose an appropriate activation function.
<img width="1214" alt="Screenshot 2023-03-03 at 5 04 56 PM" src="https://user-images.githubusercontent.com/112406455/222852149-adb85cd0-7a79-48a9-bee6-8caba0ae35ff.png">

4. If necessary, add a second hidden layer with an appropriate activation function.
<img width="1213" alt="Screenshot 2023-03-03 at 5 05 03 PM" src="https://user-images.githubusercontent.com/112406455/222852291-178c7256-1f1c-451c-928a-99a4bb2ec0ba.png">

5. Create an output layer with an appropriate activation function.
<img width="1213" alt="Screenshot 2023-03-03 at 5 05 09 PM" src="https://user-images.githubusercontent.com/112406455/222852383-2ce2a51f-180b-487e-83a7-6f745b8c2fb4.png">

6. Check the structure of the model.
<img width="1211" alt="Screenshot 2023-03-03 at 5 05 16 PM" src="https://user-images.githubusercontent.com/112406455/222852484-4893e0e8-094a-4c10-96c2-b5df0af56c6a.png">

7. Compile and train the model.
<img width="1210" alt="Screenshot 2023-03-03 at 5 05 41 PM" src="https://user-images.githubusercontent.com/112406455/222852580-6519bf66-ec9c-4918-811e-32bdda327534.png">
<img width="1215" alt="Screenshot 2023-03-03 at 5 05 49 PM" src="https://user-images.githubusercontent.com/112406455/222852637-45ac3cae-2091-45d6-abfa-b1b0ba00fc19.png">

8. Create a callback that saves the model's weights every five epochs.
<img width="1212" alt="Screenshot 2023-03-03 at 5 13 12 PM" src="https://user-images.githubusercontent.com/112406455/222852755-c7c44996-30d0-4ae9-b1b2-6a512a0f9738.png">

9. Evaluate the model using the test data to determine the loss and accuracy.
<img width="1212" alt="Screenshot 2023-03-03 at 5 06 02 PM" src="https://user-images.githubusercontent.com/112406455/222852814-323f7639-ee1b-4566-a1b9-09912fe51b93.png">

10. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.
<img width="1213" alt="Screenshot 2023-03-03 at 5 05 59 PM" src="https://user-images.githubusercontent.com/112406455/222852866-2cdb6af1-c3af-42ac-bc71-909ef8e95c0b.png">

### Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

* Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    * Dropping more or fewer columns.
    * Creating more bins for rare occurrences in columns.
    * Increasing or decreasing the number of values for each bin.
* Add more neurons to a hidden layer.
* Add more hidden layers.
* Use different activation functions for the hidden layers.
* Add or reduce the number of epochs to the training regimen.
**Note**: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimization.ipynb`.

2. Import your dependencies and read in the `charity_data.csv` to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

1. **Overview** of the analysis: Explain the purpose of this analysis.

2. **Results**: Using bulleted lists and images to support your answers, address the following questions:

* Data Preprocessing

    * What variable(s) are the target(s) for your model?
    * What variable(s) are the features for your model?
    * What variable(s) should be removed from the input data because they are neither targets nor features?
    
* Compiling, Training, and Evaluating the Model

    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * Were you able to achieve the target model performance?
    * What steps did you take in your attempts to increase model performance?
    
**Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
## References
IRS. Tax Exempt Organization Search Bulk Data Downloads. [https://www.irs.gov/](https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads)
